# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Architecture 

"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import math
import xml.etree.ElementTree as ET
from tqdm import tqdm
import torch.optim as optim
import pandas as pd # for debugging

from torchtext.vocab import FastText
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter

import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, Union

from datasetDevProp import DevPropData
from tokenizer import CustomTokenizer
from dataset import AntibodyDataset
import dataset

# Import Mamba
try:
    from mamba_ssm import Mamba, Mamba2
except ImportError:
    print("Warning: mamba_ssm not installed. Please install with: pip install mamba-ssm")
    # Fallback to a simple linear layer for demonstration
    class Mamba(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.linear = nn.Linear(d_model, d_model)
        
        def forward(self, x):
            return self.linear(x)


#debug functions
def saveTensToCsv(tensIn, fileName):
    # Convert to NumPy and then to DataFrame
    df = pd.DataFrame(tensIn.view(-1,tensIn.shape[-1]).cpu().detach().numpy())

    # Save to CSV
    df.to_csv(fileName, index=False)
    print(f"Tensor saved to {fileName}")


#%%
# Custom Embedding layer with scaling by sqrt(d_model)
class CustomTokenEmbedding(nn.Module):
    def __init__(self, tokenizerIn: CustomTokenizer, modelDim=8):
        super().__init__()
        # params (potentially to move to configuration file)
        self.modelDim = modelDim # dimension of the internal representation of the model
        self.tokenizer = tokenizerIn

        # embeddings for custom tokens
        self.tokEmbedding = nn.Embedding(tokenizerIn.vocab_size, self.modelDim, padding_idx=CustomTokenizer.customPad)

        # value projection
        self.valProj = nn.Linear(1, self.modelDim) # not used

        #=== plm sequence embedding
        # ESM2 model for sequence embeddings
        self.plmTokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")  # Using smaller model for demonstration
        self.plm = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")  # Using smaller model for demonstration
        self.plm.eval()  # set to evaluation mode if only extracting embeddings
        self.plmEmbProj = nn.Linear(self.plm.config.hidden_size, self.modelDim) # project 
        #===

        #=== text emb
        # see https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
        self.sentEmbMod = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.sentEmbMod.eval()
        self.sentEmbProj = nn.Linear(384, self.modelDim) # project 
        #===

        # Freeze PLM weights
        for param in self.plm.parameters():
            param.requires_grad = False

        # Freeze sentence embedding model weights  
        for param in self.sentEmbMod.parameters():
            param.requires_grad = False

    def forward(self, tokenDic, tokenExtraDic):
        x = tokenDic['token_ids']
        xSent = tokenDic['sent_token_ids']
        xPlm = tokenDic['plm_token_ids']

        # x shape: (batch_size, seq_len)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        # allocate output tensor (batch, )
        output = torch.ones(seq_len, batch_size, self.modelDim, device=x.device, dtype=torch.float)  * self.tokenizer.customPad     # set all values to self.padVal

        output = self.tokEmbedding(x)

        # reshape with view 
        xFlatBatch = x.view(-1)
        outFlatBatch = output.view( -1, self.modelDim )
        #==== Place sentence embeddings in the sequence
        with torch.no_grad():
            # get text embeddings
            xSentPerPrompt = xSent.reshape(xSent.shape[0]*xSent.shape[1],xSent.shape[2])
            outSentMod = self.sentEmbMod( input_ids=xSentPerPrompt, token_type_ids=torch.zeros_like(xSentPerPrompt), attention_mask=xSentPerPrompt!=self.tokenizer.sentPadIdx )
            sent_emb = self._sentence_mean_pooling(outSentMod, xSentPerPrompt!=self.tokenizer.sentPadIdx )

        # find embeddings that do NOT have all padding indexes 
        sent_emb = sent_emb[ torch.where((xSentPerPrompt == self.tokenizer.sentPadIdx ).all(axis=1)==False)[0] ]

        # project embedding to correct model size
        sent_emb = self.sentEmbProj(sent_emb) # project

        # reshape and identify locations of sentence placeholder in sequence
        assert(xFlatBatch.shape[0]==outFlatBatch.shape[0])
        sentLocs = torch.where(xFlatBatch==self.tokenizer.fullTok_to_idx['sentence-placeholder'])[0] # xFlatBatch stores 0s for embeddings with placeholders only 
        # assign sentence embeddings to correct locations
        outFlatBatch[sentLocs,:] = sent_emb
        #====

        #==== Plm embeddings in the sequence
        with torch.no_grad():
            # get plm embeddings
            xPlmPerPrompt = xPlm.reshape(xPlm.shape[0]*xPlm.shape[1],xPlm.shape[2])
            outPlmMod = self.plm(input_ids=xPlmPerPrompt, attention_mask=xPlmPerPrompt!=self.tokenizer.plmPadIdx)
            plm_emb = self._seq_mean_pooling(outPlmMod.last_hidden_state)
        # identify valid embeddings
        plm_emb = plm_emb[ torch.where((xPlmPerPrompt == self.tokenizer.plmPadIdx).all(axis=1)==False)[0], : ]

        # project embedding to correct model size
        plm_emb = self.plmEmbProj(plm_emb)
        
        # identify locations of plm placeholder in sequence
        plmLocs = torch.where(xFlatBatch==self.tokenizer.fullTok_to_idx['seq-placeholder'])[0]
        # assign plm embeddings to correct locations
        outFlatBatch[plmLocs,:] = plm_emb
        #====

        #==== val embeddings in the sequence
        # get value data from tokenExtraDic
        valData = tokenExtraDic['val_data']
        valDataFlat = torch.tensor([val for sublist in valData for val in sublist], device=x.device, dtype=torch.float)
        # project values to model dimension
        val_emb = self.valProj(valDataFlat.unsqueeze(-1))
        # identify locations of value placeholder in sequence
        valLocs = torch.where(xFlatBatch==self.tokenizer.fullTok_to_idx['value-placeholder'])[0]
        # assign value embeddings to correct locations
        outFlatBatch[valLocs,:] = val_emb
        #====

        return output
    
    def _sentence_mean_pooling(self,model_output, attention_mask):
        # Sentence Mean Pooling - Take attention mask into account for correct averaging
        # from https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _seq_mean_pooling(self,model_last_hidden_state):
        return model_last_hidden_state.mean(dim=1)




# Mamba-based Transformer model
class MambaModel(nn.Module):
    def __init__(self, embModel, n_layers=6, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.embModel = embModel
        d_model = embModel.modelDim
        
        # Stack of Mamba layers
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(n_layers)
        ])
        # self.mamba_layers = nn.ModuleList([
        #     Mamba2(
        #         d_model=d_model,
        #         d_state=d_state,
        #         d_conv=d_conv,
        #         expand=expand
        #     ) for _ in range(n_layers)
        # ])


        # Layer normalization for each Mamba layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        # self.pos_encoding = PositionalEncoding(d_model)
        # self.pos_encoding = PositionalEncoding1D(d_model)
        

        # add linear layer to be attached to the special prediction token and a single property
        self.predLayer = nn.Linear(d_model, 1) 

    def forward(self, tokenDic, tokenExtraDic):
        input_ids = tokenDic['token_ids']
        assert(torch.all(input_ids[:,0]==self.embModel.tokenizer.fullTok_to_idx['pred-tok'])) # first token needs to be prediction token

        batch_size, seq_len = input_ids.size()
        x = self.embModel(tokenDic, tokenExtraDic)
        
        #TODO: potentially put back fixed pos encodings
        # positional encoding not needed in Mamba 
        # x = self.pos_encoding(x)
        
        # Pass through Mamba layers with residual connections
        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            residual = x
            x = mamba_layer(x)
            x = layer_norm(x + residual)

        # predOut = self.predLayer(x[:,0,:]) # shape: (batch_size, seq, feat_dim) - first token across all batches
    
        
        # add linear layer to be attached to  last token's hidden states (this provide better performance for Mamba based architectures)
        last_hidden_state = x[:, -1, :] # shape: (batch_size, seq, feat_dim)

        predOut = self.predLayer(last_hidden_state) 

        
        return predOut


if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test input prompts
    promptLst = ["""
            <antibody>
                <property name="thermostability 2">0.94</property>
                <property name="thermostability">0.95</property>
                <property name="solubility">0.5</property>
                <property name="myprop">0.2</property>
                <seq>WVVNGNRICENWLGTFNYHS</seq>
                <seq-l>ARESTIHWSCIAAYIAPSKEICVITWN</seq-l>
            </antibody>
            <antibody>
                <property name="thermostability">0.51</property>
                <property name="hydrophobicity">0.2</property>
                <seq>MCWDHGEPQPNFAETMDWDIYEV</seq>
            </antibody>


            <query-antibody>
                <seq>GIDNPWRFNDISVKCWMIY</seq><seq-l>FKAQKNCQTW</seq-l>
                <property name="thermostability" />
            </query-antibody>
            """,
            """
            <antibody>
                <property name="thermostability">0.95</property>
                <property name="solubility">0.5</property>
                <seq>WVVNGNRICENWLGTFNYHS</seq>
                <seq-l>ARESTIHWSCIAAYIAPSKEICVITWN</seq-l>
            </antibody>
            <antibody>
                <property name="thermostability">0.51</property>
                <seq>MCWDHGEPQPNFAETMDWDIYEV</seq>
                <seq-l>ISQVRPMHSFIITWSDVSYI</seq-l>
            </antibody>


            <query-antibody>
                <seq>GIDNPWRFNDISVKCWMIY</seq><seq-l>FKAQKNCQTW</seq-l>
                <property name="thermostability" />
            </query-antibody>
            """,
            """
            <antibody>
                <property name="thermostability">0.95</property>
                <property name="solubility">0.5</property>
                <seq>WVVNGNRICENWLGTFNYHS</seq>
                <seq-l>ARESTIHWSCIAAYIAPSKEICVITWN</seq-l>
            </antibody>
            <antibody>
                <property name="thermostability">0.51</property>
                <seq>MCWDHGEPQPNFAETMDWDIYEV</seq>
                <seq-l>ISQVRPMHSFIITWSDVSYI</seq-l>
            </antibody>


            <query-antibody>
                <seq>GIDNPWRFNDISVKCWMIY</seq><seq-l>FKAQKNCQTW</seq-l>
                <property name="thermostability" />
            </query-antibody>
            """ ]
    ansLst = [ \
    """
    <ans-antibody>
        <seq>GIDNPWRFNDISVKCWMIY</seq><seq-l>FKAQKNCQTW</seq-l>
        <property name="thermostability">0.89</property>
    </ans-antibody>
    """,
        """
    <ans-antibody>
        <seq>GIDNPWRFNDISVKCWMIY</seq><seq-l>FKAQKNCQTW</seq-l>
        <property name="thermostability">0.89</property>
    </ans-antibody>
    """,
        """
    <ans-antibody>
        <seq>GIDNPWRFNDISVKCWMIY</seq><seq-l>FKAQKNCQTW</seq-l>
        <property name="thermostability">0.89</property>
    </ans-antibody>
    """ ]

    # Create tokenizer instance with empty params
    abTokenizer = CustomTokenizer({})

    abData = AntibodyDataset(promptLst, ansLst, abTokenizer)

    # Create data loaders
    train_loader = DataLoader(
        abData,
        batch_size=8,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
  
    # embedding model creating the full tokens 
    embMod = CustomTokenEmbedding(abTokenizer)

    vocab_size = abTokenizer.vocab_size
    modelDim = embMod.modelDim
    model = MambaModel(embMod, n_layers=2, d_state=16, d_conv=4, expand=2)
    model.to(device)

    l2e = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # perform an inference step
    with tqdm(total=len(train_loader), desc=f"test") as pbar:
        for batch in train_loader:
            # get prompt training data
            tokenDic = {
                    'token_ids': batch['prompt']['token_ids'].to(device),
                    'sent_token_ids': batch['prompt']['sent_token_ids'].to(device),
                    'plm_token_ids': batch['prompt']['plm_token_ids'].to(device)
            }
            tokenExtraDic = {
                    'sent_texts': batch['prompt']['sent_texts'],
                    'plm_seqs': batch['prompt']['plm_seqs'],
                    'val_data': batch['prompt']['val_data'],
                    'texts': batch['prompt']['texts']
            }
            # get answer training data
            targets = batch['answer']['token_ids'].to(device)
            targets_extra_info = batch['answer']['token_extra_info']

            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(tokenDic,tokenExtraDic)

            # find expected property output (assumes only a single output per query)
            expOut = torch.zeros_like(outputs)
            for bId in range(len(targets_extra_info)):
                # get number of properties (query antibody always have at least one property)
                propNum = len(targets_extra_info[bId]['ansExtraData'])
                for propId in range(propNum):
                    propInfo = targets_extra_info[bId]['ansExtraData'][propId]
                    if propInfo is None:
                        continue
                    if 'value' in propInfo:
                        expOut[bId] = propInfo['value']
                        break # only use the first one

            # mean absolute error loss
            loss = l2e(outputs, expOut)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            pbar.update(1)

    print('done')