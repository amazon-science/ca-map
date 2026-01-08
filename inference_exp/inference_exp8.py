# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Example inference experiments

"""


#%%

import sys
sys.path.append('..')


import os
print(os.getcwd())
assert('/inference_exp' in os.getcwd()) # make sure it is running in the right path 

#%%
import torch
import torch.nn as nn
import pickle
import numpy as np

import xml.etree.ElementTree as ET
from tqdm import tqdm
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd


from datasetDevProp import DevPropData
from tokenizer import CustomTokenizer
from dataset import AntibodyDataset
from model_mamba import CustomTokenEmbedding, MambaModel
# from model import CustomTokenEmbedding, DecoderTransformer

# optimization suggested by pytorch
torch.set_float32_matmul_precision('high')
# suppress dynamo warnings and reduce recompilations
torch._dynamo.config.cache_size_limit = 256
# torch._dynamo.config.suppress_errors = True
# # Reduce recompilations by allowing more shape variations
# torch._dynamo.config.automatic_dynamic_shapes = True




#======== Parameters

batchSize = 128
# set expId 
expLbl = 'exp8'

MODEL_DIR = '../models/'
OUT_DIR='../out/'

# Dataset containing antibodies and properties. Follows the format from https://github.com/csi-greifflab/developability_profiling/blob/main/data/native/developability.csv
rawDataLocation = '../../developability_profiling/data/native/developability.csv' 

# Cache paths for data splits
proc_data_cache_path = '../runs/tok/'+expLbl+'_proc_dataset.pkl'

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
    print(f"Created directory: {OUT_DIR}")

# use model weights with best_weights in file name
modelFile = None
for file in os.listdir(MODEL_DIR):
    if ('best_weights' in file) and (expLbl in file):
        modelFile = os.path.join(MODEL_DIR, file)
        break
    
assert(modelFile is not None)



# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
#========

def load_model(model_path,  modelIn):
    """Load trained model weights"""
    model = modelIn
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device,weights_only=True)
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            new_key = k.replace('_orig_mod.', '')
            new_state_dict[new_key] = v
        res = model.load_state_dict(new_state_dict)
        print(res)
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found. Using random weights.")
    
    return model






# Try to load cached data splits
if os.path.exists(proc_data_cache_path):
    print("Loading cached data splits...")
    with open(proc_data_cache_path, 'rb') as f:
        setA, setB, setC,trDataset,valDataset,tsDataset, tokenizer = pickle.load(f)


    print(f"Loaded setA: {len(setA)}, setB: {len(setB)}, setC: {len(setC)} samples + datasets ({len(trDataset)}, {len(valDataset)}, {len(tsDataset)}) + tokenizer from cache")
else:
    # assumes that the data splits have been already created during training 
    assert(False)

   


# Create test data loader
test_loader = DataLoader(
    tsDataset,
    batch_size=batchSize,
    shuffle=False,
    collate_fn=tsDataset.collate_fn
)


# Create model
emb_model = CustomTokenEmbedding(tokenizer, modelDim=32)
model = MambaModel(emb_model, n_layers=6, d_state=32, d_conv=8, expand=4)
print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

print("loading model from ", modelFile)
model = load_model(modelFile, model)


model.to(device)
model.eval()




#======================================= run inference



def runInference( inference_id, biasMaxVal=0 ):
    predLst = []
    gtPropValLst = []
    predNameLst = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
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
            # generate a random bias from 0 to biasMaxVal
            biasVal = 0
            if biasMaxVal > 0:
                # generate fixed bias
                biasVal = np.random.uniform(0, biasMaxVal)

                #===add fixed bias to tokenExtraDic['val_data']
                for bId in range(len(tokenExtraDic['val_data'])):
                    for sId in range(len(tokenExtraDic['val_data'][bId])):
                        tokenExtraDic['val_data'][bId][sId] += biasVal
                #===a

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
                        expOut[bId] = propInfo['value'] + biasVal # store value + bias
                        predNameLst.append(propInfo['name'])
                        break
            

            
            # Store predictions and targets 
            predLst = np.concatenate((predLst, outputs.cpu().numpy().flatten())).tolist()
            gtPropValLst = np.concatenate((gtPropValLst, expOut.cpu().numpy().flatten())).tolist()
            #predNameLst already stored

    # create "{OUT_DIR}/{expLbl} if does not exist
    if not os.path.exists(f"{OUT_DIR}/{expLbl}"):
        os.makedirs(f"{OUT_DIR}/{expLbl}")
        print(f"Created directory: {OUT_DIR}/{expLbl}")

    #=== Compute performance
    # find unique names in predNameLst
    uniqueNames = list(set(predNameLst))

    for propName in uniqueNames:
        print("Computing performance for ", propName)
        # get indices of all predictions with this name
        indices = [i for i, x in enumerate(predNameLst) if x == propName]
        # get predictions and ground truth for these indices
        pred_flat = [predLst[i] for i in indices]
        gt_flat = [gtPropValLst[i] for i in indices]

        # Calculate Spearman correlation and R²
        spearman_corr, spearman_p = spearmanr(gt_flat, pred_flat)
        r2 = r2_score(gt_flat, pred_flat)

        # Create regression plot
        plt.figure()
        plt.scatter(gt_flat, pred_flat, alpha=0.6)
        plt.xlabel('Ground Truth')
        plt.ylabel('Predictions')
        plt.title(f'Regression Plot (Spearman r = {spearman_corr:.3f}, R² = {r2:.3f})\n{inference_id}')
        plt.plot([min(gt_flat), max(gt_flat)], [min(gt_flat), max(gt_flat)], 'r--', alpha=0.8)

        # Save plot
        plt.savefig(f"{OUT_DIR}/{expLbl}/{inference_id}-{propName}-regression_plot.png")
        plt.close()

        # Save predictions and ground truth (in the same file with pandas)
        df = pd.DataFrame({'predictions': pred_flat, 'ground_truth': gt_flat})
        df.to_csv(f"{OUT_DIR}/{expLbl}/{inference_id}-{propName}-predictions.csv", index=False)
        # save performance as a text file
        with open(f"{OUT_DIR}/{expLbl}/{inference_id}--{propName}-performance.txt", 'w') as f:
            f.write('Performance of the model:\n')
            f.write(f"Inference ID: {inference_id}\n")
            f.write(f'Number of samples: {len(pred_flat)}\n')
            f.write(f'Property: {propName}\n')
            f.write(f'Spearman Correlation: {spearman_corr:.4f}\n')
            f.write(f'Spearman p-value: {spearman_p:.4e}\n')
            f.write(f'R² Score: {r2:.4f}\n')

        #===


def runInferenceNewProp( inference_id, propToUseLstIn, propAnsLstIn, abNumRangePerPrompt=[10,15],biasMaxVal=0 ):


    dev_prop_data = DevPropData(inFile=rawDataLocation)

    #===================== these should not be needed as everything is pre-loaded, but just in case, use the same matching the train.py for this experiment
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
    #===================== 
    # note that this will still use setC, i.e. the left out set of antibody generated in train.py
    promptsTs, answersTs = dev_prop_data.gen_train_prompts(setC, 
                                                            propSubset=propToUseLstIn, 
                                                            abNumRangePerPrompt=abNumRangePerPrompt,
                                                            propAnsSubset=propAnsLstIn,
                                                            includeQueryAbInCntx=True ) # if True, it will include  the query AB in the context but only for the set of properties set(propSubset)-set(propAnsSubset in the prompt) 
    print(f"Generated {len(promptsTs)} test examples")
    tsDataset = AntibodyDataset(promptsTs, answersTs, tokenizer)
    print("Data splits+tokenizer generated")

    # Create test data loader
    test_loader_custom = DataLoader(
        tsDataset,
        batch_size=batchSize,
        shuffle=False,
        collate_fn=tsDataset.collate_fn
        )


    predLst = []
    gtPropValLst = []
    predNameLst = []

    with torch.no_grad():
        for batch in tqdm(test_loader_custom):
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
            # generate a random bias from 0 to biasMaxVal
            biasVal = 0
            if biasMaxVal > 0:
                # generate fixed bias
                biasVal = np.random.uniform(0, biasMaxVal)

                #===add fixed bias to tokenExtraDic['val_data']
                for bId in range(len(tokenExtraDic['val_data'])):
                    for sId in range(len(tokenExtraDic['val_data'][bId])):
                        tokenExtraDic['val_data'][bId][sId] += biasVal
                #===a

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
                        expOut[bId] = propInfo['value'] + biasVal # store value + bias
                        predNameLst.append(propInfo['name'])
                        break
            

            
            # Store predictions and targets 
            predLst = np.concatenate((predLst, outputs.cpu().numpy().flatten())).tolist()
            gtPropValLst = np.concatenate((gtPropValLst, expOut.cpu().numpy().flatten())).tolist()
            #predNameLst already stored

    # create "{OUT_DIR}/{expLbl} if does not exist
    if not os.path.exists(f"{OUT_DIR}/{expLbl}"):
        os.makedirs(f"{OUT_DIR}/{expLbl}")
        print(f"Created directory: {OUT_DIR}/{expLbl}")

    #=== Compute performance
    # find unique names in predNameLst
    uniqueNames = list(set(predNameLst))

    for propName in uniqueNames:
        print("Computing performance for ", propName)
        # get indices of all predictions with this name
        indices = [i for i, x in enumerate(predNameLst) if x == propName]
        # get predictions and ground truth for these indices
        pred_flat = [predLst[i] for i in indices]
        gt_flat = [gtPropValLst[i] for i in indices]

        # Calculate Spearman correlation and R²
        spearman_corr, spearman_p = spearmanr(gt_flat, pred_flat)
        r2 = r2_score(gt_flat, pred_flat)

        # Create regression plot
        plt.figure()
        plt.scatter(gt_flat, pred_flat, alpha=0.6)
        plt.xlabel('Ground Truth')
        plt.ylabel('Predictions')
        plt.title(f'Regression Plot (Spearman r = {spearman_corr:.3f}, R² = {r2:.3f})\n{inference_id}')
        plt.plot([min(gt_flat), max(gt_flat)], [min(gt_flat), max(gt_flat)], 'r--', alpha=0.8)

        # Save plot
        plt.savefig(f"{OUT_DIR}/{expLbl}/{inference_id}-{propName}-regression_plot.png")
        plt.close()

        # Save predictions and ground truth (in the same file with pandas)
        df = pd.DataFrame({'predictions': pred_flat, 'ground_truth': gt_flat})
        df.to_csv(f"{OUT_DIR}/{expLbl}/{inference_id}-{propName}-predictions.csv", index=False)
        # save performance as a text file
        with open(f"{OUT_DIR}/{expLbl}/{inference_id}--{propName}-performance.txt", 'w') as f:
            f.write('Performance of the model:\n')
            f.write(f"Inference ID: {inference_id}\n")
            f.write(f'Number of samples: {len(pred_flat)}\n')
            f.write(f'Property: {propName}\n')
            f.write(f'Spearman Correlation: {spearman_corr:.4f}\n')
            f.write(f'Spearman p-value: {spearman_p:.4e}\n')
            f.write(f'R² Score: {r2:.4f}\n')

        #===


#%% Experiment without bias 
    
inference_id = expLbl + '-no-bias'
runInference( inference_id, biasMaxVal=0 )


#%% Experiment with bias 

inference_id = expLbl + '-max-0.3-bias'
runInference( inference_id, biasMaxVal=0.3 )



#%% Experiment including properties not used for training 
propToUseLstCust = [ 'Hydrophobicity', "NegCh heterogeneity", "Solubility", "Stability 2 (aliph)", "Immunogenicity", "PosCh heterogeneity", "Immunogenicity", "PosCh heterogeneity"]
propAnsLstCust = [ "Immunogenicity", "PosCh heterogeneity"]
inference_id = expLbl + '-new-prop-bias-max-0.3-more-cntx'

runInferenceNewProp( inference_id, propToUseLstCust, propAnsLstCust, abNumRangePerPrompt=[10,15],biasMaxVal=0.3 )
