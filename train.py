# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Training script

before running, check parameters. 
main parameters for an initial  training run  are: 
- expId: defining the ID of the experiment (which will be used to name output files, including weights)
- rawDataLocation: Dataset containing antibodies and properties. Follows the format in https://github.com/csi-greifflab/developability_profiling/blob/main/data/native/developability.csv

"""

import torch
import torch.nn as nn
import math
import numpy as np
import pickle
import os

import xml.etree.ElementTree as ET
from tqdm import tqdm
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from scipy.stats import spearmanr


from datasetDevProp import DevPropData
from tokenizer import CustomTokenizer
from dataset import AntibodyDataset
from model_mamba import CustomTokenEmbedding, MambaModel


# optimization suggested by pytorch
torch.set_float32_matmul_precision('high')
# suppress dynamo warnings and reduce recompilations
torch._dynamo.config.cache_size_limit = 256


def main():
    #======================================= parameters

    # Dataset containing antibodies and properties. Follows the format from https://github.com/csi-greifflab/developability_profiling/blob/main/data/native/developability.csv
    rawDataLocation = '../developability_profiling/data/native/developability.csv' 

    # propToUseLst = ['Hydrophobicity']
    propToUseLst = ['Hydrophobicity', "NegCh heterogeneity", "Solubility", "Stability 2 (aliph)"]
    # number of context antibodies to use for each prompt
    abNumRangePerPrompt = [10,15]

    # maxBias = 0
    maxBias = 0.3 # bias simulation for AB-context-aware training
    custRandMaxBiasPerProp=True #  add fixed random bias value, up until maxBias, for each individual property. i.e. each prompt will have N different biases, where N is the number of properties in the prompt
    # maxAbSamplesInTotal = 50000 # set to None for everything 
    maxAbSamplesInTotal = None # set to None for everything

    # optimizer params
    batchSize = 128
    learning_rate = 0.001
    weight_decay=0.01
    num_epochs = 20 # maximum number of epochs 
    
    # set experiment ID 
    expLbl = 'exp8'

    expId = expLbl + '-' + datetime.now().strftime("%Y%m%d-%H%M%S")

    # model weights out name
    modelOutFile = f'models/{expId}.pt'

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize TensorBoard writer
    writer = SummaryWriter(f'runs/training_info/{expId}')

   # Cache paths for data splits
    proc_data_cache_path = 'runs/tok/'+expLbl+'_proc_dataset.pkl'

    #=======================================
    
    # Try to load cached data splits
    if os.path.exists(proc_data_cache_path):
        print("Loading cached data splits...")
        with open(proc_data_cache_path, 'rb') as f:
            setA, setB, setC,trDataset,valDataset,tsDataset, tokenizer = pickle.load(f)


        print(f"Loaded setA: {len(setA)}, setB: {len(setB)}, setC: {len(setC)} samples + datasets ({len(trDataset)}, {len(valDataset)}, {len(tsDataset)}) + tokenizer from cache")
    else:
        #=== Load dataset using DevPropData
        dev_prop_data = DevPropData(inFile=rawDataLocation)


        # add normalization
        normDict = {
                    'Solubility': {'min': 0, 'max': 1}, 
                    'Immunogenicity': {'min': 0, 'max': 2}, 
                    'Stability 1 (insta)': {'min': 0, 'max': 80}, 
                    'Stability 2 (aliph)': {'min': 30, 'max': 100}, 
                    'Hydrophobicity': {'min': -0.2, 'max': 0.6}, 
                    'PosCh heterogeneity': {'min': 0, 'max': 10}, 
                    'NegCh heterogeneity': {'min': 0, 'max': 10}
                    }


        dev_prop_data.customNormalizationProp(normDict)


        setA, setB, setC = dev_prop_data.get_sets_exp1(maxSamples=maxAbSamplesInTotal)
        #===
        
        # Create tokenizer 
        tokenizer = CustomTokenizer({})

        # Generate training prompts and answers
        prompts, answers = dev_prop_data.gen_train_prompts(setA, 
                                                        propSubset=propToUseLst, 
                                                        abNumRangePerPrompt=abNumRangePerPrompt,
                                                        useRandMaxBias=maxBias, 
                                                        custRandMaxBiasPerProp=custRandMaxBiasPerProp)
        print(f"Generated {len(prompts)} training examples")
        trDataset = AntibodyDataset(prompts, answers, tokenizer)

        promptsVal, answersVal = dev_prop_data.gen_train_prompts(setB, 
                                                                propSubset=propToUseLst, 
                                                                abNumRangePerPrompt=abNumRangePerPrompt,
                                                                custRandMaxBiasPerProp=custRandMaxBiasPerProp)
        print(f"Generated {len(promptsVal)} val examples")
        valDataset = AntibodyDataset(promptsVal, answersVal, tokenizer)


        promptsTs, answersTs = dev_prop_data.gen_train_prompts(setC, 
                                                                propSubset=propToUseLst, 
                                                                abNumRangePerPrompt=abNumRangePerPrompt,
                                                                custRandMaxBiasPerProp=custRandMaxBiasPerProp)
        print(f"Generated {len(promptsTs)} test examples")
        tsDataset = AntibodyDataset(promptsTs, answersTs, tokenizer)


        
        # Save data splits to cache
        with open(proc_data_cache_path, 'wb') as f:
            pickle.dump((setA, setB, setC,trDataset,valDataset,tsDataset,tokenizer), f)


        print("Data splits+tokenizer cached successfully")



    # Create training data loader
    train_loader = DataLoader(
        trDataset,
        batch_size=batchSize,
        shuffle=True,
        collate_fn=trDataset.collate_fn
    )

    # Create validation data loader
    val_loader = DataLoader(
        valDataset,
        batch_size=batchSize,
        shuffle=False,
        collate_fn=valDataset.collate_fn
    )


    # Create model
    emb_model = CustomTokenEmbedding(tokenizer, modelDim=32)
    model = MambaModel(emb_model, n_layers=6, d_state=32, d_conv=8, expand=4)

    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    # # compilation
    # print("start compile")
    # model = torch.compile(model) # compile model to speed things up
    # print("compilation done")
    model.to(device)



    # Loss and optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # mae loss obj
    # mae = nn.L1Loss()
    l2e = nn.MSELoss()


    # Training loop
    global_step = 0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, batch in enumerate(train_loader):
                # Get data
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
                targets_extra_info = batch['answer']['token_extra_info']

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(tokenDic,tokenExtraDic)

                # find expected property output (assumes only a single output per query)
                expOut = torch.zeros_like(outputs)
                for bId in range(len(targets_extra_info)):
                    # get number of properties (query antibody always have at least one property)
                    propNum = len(targets_extra_info[bId]['ansExtraData'])
                    for propId in range(propNum):
                        # get property info
                        propInfo = targets_extra_info[bId]['ansExtraData'][propId]
                        if propInfo is None:
                            continue
                        if 'value' in propInfo:
                            expOut[bId] = propInfo['value']
                            break # only use the first one

                # mean absolute error loss
                loss = l2e(outputs, expOut)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Log to TensorBoard
                writer.add_scalar('Loss/TrainFull', loss.item(), global_step)
                
                # Log model weights periodically
                if global_step % 10 == 0:
                    # log model weights
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            writer.add_histogram(f'Weights/{name}', param.data, global_step)
                            writer.add_histogram(f'Gradients/{name}', param.grad, global_step)
                    


                epoch_loss += loss.item()
                global_step += 1
                
                pbar.update(1)
                pbar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/Epoch', avg_loss, epoch)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        

        #======================================= run inference for validation metrics
        model.eval()
        val_loss = 0.0
        predLst = []
        gtPropValLst = []

        
        with torch.no_grad():
            for batch in val_loader:
                # Get data
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
                targets_extra_info = batch['answer']['token_extra_info']
                
                outputs = model(tokenDic, tokenExtraDic)
                
                expOut = torch.zeros_like(outputs)
                for bId in range(len(targets_extra_info)):
                    propNum = len(targets_extra_info[bId]['ansExtraData'])
                    for propId in range(propNum):
                        # get property info
                        propInfo = targets_extra_info[bId]['ansExtraData'][propId]
                        if propInfo is None:
                            continue
                        if 'value' in propInfo:
                            expOut[bId] = propInfo['value']
                            break
                
                val_loss += l2e(outputs, expOut).item()
                

                predLst = np.concatenate((predLst, outputs.cpu().numpy().flatten())).tolist()
                gtPropValLst = np.concatenate((gtPropValLst, expOut.cpu().numpy().flatten())).tolist()

        
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        

        
        # Log prediction statistics
        if predLst:
            spearman_corr, spearman_p = spearmanr(gtPropValLst, predLst)

            
            writer.add_scalar('Predictions/spearman_corr_val', spearman_corr, epoch)

            print("validation on spearman_corr", spearman_corr)
        
        print(f"Validation loss: {avg_val_loss:.4f}")
        #=======================================


        
        # Save best model weights
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_file = modelOutFile.replace('.pt', '_best_weights.pt')
            torch.save(model.state_dict(), best_model_file)
            print(f"New best loss: {best_loss:.4f} - Model saved as '{best_model_file}'")

    # Save final model
    torch.save(model.state_dict(), modelOutFile)
    print(f"Final model saved as '{modelOutFile}'")
    print(f"Best model saved with loss: {best_loss:.4f}")
    
    writer.close()
    print("Training completed. TensorBoard logs saved in 'runs/training_info'")

if __name__ == "__main__":
    main()