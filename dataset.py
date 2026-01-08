# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Class wrapper to make DevPropData and CustomTokenizer compatible with pytorch
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import torch
import torch.nn as nn
import math
import xml.etree.ElementTree as ET

from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, Union, List

from datasetDevProp import DevPropData
from tokenizer import CustomTokenizer

# PAD_VALUE = CustomTokenizer.customPad

# PAD_PLM = None
# PAD_SENT = None

class AntibodyDataset(Dataset):
    def __init__(self, dataPrompts: List[str], 
                 dataAnswers: List[str], 
                 tokenizer: CustomTokenizer):
        """
        Initialize the dataset with a list of text examples.
        
        Args:
            data: List of text examples in the format:
                  "<antibody-start>...<antibody-end>
                   <antibody-start>...<antibody-end>
                   <query-antibody-start>...<query-antibody-end>"
            tokenizer: The CustomTokenizer instance
        """
        if len(dataPrompts) != len(dataAnswers):
            raise ValueError("Number of prompts and answers must be equal.")

        self.dataPrompts = dataPrompts
        self.dataAnswers = dataAnswers
        self.tokenizer = tokenizer
        # set padding in glob vars
        # global PAD_PLM, PAD_SENT
        self.PAD_VALUE = CustomTokenizer.customPad
        self.PAD_PLM = tokenizer.plmPadIdx
        self.PAD_SENT = tokenizer.sentPadIdx

        self.processedPrompts = [self.process_example_prompt_ans(currText) for currText in tqdm(self.dataPrompts, desc="Processing prompts")]
        self.processedAns = [self.process_example_prompt_ans(currText) for currText in tqdm(self.dataAnswers, desc="Processing answers")]
        
    def process_example_prompt_ans(self, currText: str):
        """
        Process a text example prompt into model inputs
        """
        # Parse the text into antibodies and query
        token_ids, token_extra_info = self.tokenizer(currText)
        
        # Encode the text into token IDs
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        
            
        return {
            'token_ids': token_ids,
            'token_extra_info': token_extra_info,
            'text': currText
        }
        
    def __len__(self):
        return len(self.dataPrompts)
        
    def __getitem__(self, idx):
        return {'prompt':self.processedPrompts[idx], 'answer':self.processedAns[idx]}

    def collate_fn(self, batch):
        """
        Custom collate function for batching examples.
        """
        # global PAD_PLM, PAD_SENT

        batchOut = {"prompt":{}, "answer":{} }
        for outType in ("prompt","answer"):

            #=== Pad main token IDs 
            # Get max sequence length for padding
            max_len = max([item[outType]['token_ids'].size(0) for item in batch])
            
            
            padded_token_ids = torch.ones(len(batch), max_len, dtype=torch.long) * self.PAD_VALUE
            for i, item in enumerate(batch):
                padded_token_ids[i, :item[outType]['token_ids'].size(0)] = item[outType]['token_ids']
            #===

            #=== Pad sentence embedding token IDs 
            dataKeyName = 'sentTokData'
            dataValName = 'sentTxt'
            # check if the sampels should not be parsed because empty
            doNotParseLst = [bSmp[outType]['token_extra_info'][dataKeyName][0] is None for bSmp in batch]
            # if any of the samples are empty, then skip parsing
            if any(doNotParseLst):
                sent_padded_token_ids = None
                sent_val_out = None
            else:
                # Get max sequence length for padding
                max_len = max([len(sSmp) for bSmp in batch \
                            for sSmpLst in bSmp[outType]['token_extra_info'][dataKeyName] \
                                for sSmp in sSmpLst['input_ids'] ] \
                            )
                # get max number of sentences
                max_sent_num = max([len(sSmpLst['input_ids']) for bSmp in batch \
                                    for sSmpLst in bSmp[outType]['token_extra_info'][dataKeyName] ] \
                                )
                
                
                sent_padded_token_ids = torch.ones(len(batch), max_sent_num, max_len, dtype=torch.long) * self.PAD_SENT
                sent_val_out = [None]*len(batch)
                for bi, bSmp in enumerate(batch):
                    sent_val_out[bi] = bSmp[outType]['token_extra_info'][dataValName]
                    for sli, sSmpLst in enumerate(bSmp[outType]['token_extra_info'][dataKeyName]):
                        for si, sSmp in enumerate(sSmpLst['input_ids']):
                            sent_padded_token_ids[bi, si, :len(sSmp)] = torch.tensor(sSmp, dtype=torch.long)
            #===

            #=== Pad plm embedding token IDs 
            dataKeyName = 'seqTokData'
            dataValName = 'seqLst'
            # check if the sampels should not be parsed because empty
            doNotParseLst = [bSmp[outType]['token_extra_info'][dataKeyName][0] is None for bSmp in batch]
            # if any of the samples are empty, then skip parsing
            if any(doNotParseLst):
                sent_padded_token_ids = None
                sent_val_out = None
            else:
                # Get max sequence length for padding
                max_len = max([len(sSmp) for bSmp in batch \
                            for sSmpLst in bSmp[outType]['token_extra_info'][dataKeyName] \
                                for sSmp in sSmpLst['input_ids'] ] \
                            )
                # get max number of sequences
                max_seq_num = max([len(sSmpLst['input_ids']) for bSmp in batch \
                                    for sSmpLst in bSmp[outType]['token_extra_info'][dataKeyName] ] \
                                )
                
                plm_padded_token_ids = torch.ones(len(batch), max_seq_num, max_len, dtype=torch.long) * self.PAD_PLM
                plm_val_out = [None]*len(batch)
                for bi, bSmp in enumerate(batch):
                    plm_val_out[bi] = bSmp[outType]['token_extra_info'][dataValName]
                    for sli, sSmpLst in enumerate(bSmp[outType]['token_extra_info'][dataKeyName]):
                        for si, sSmp in enumerate(sSmpLst['input_ids']):
                            plm_padded_token_ids[bi, si, :len(sSmp)] = torch.tensor(sSmp, dtype=torch.long)
            #===

            batchOut[outType] = {
                'token_ids': padded_token_ids,
                'sent_token_ids': sent_padded_token_ids,
                'sent_texts': sent_val_out,
                'plm_token_ids': plm_padded_token_ids,
                'plm_seqs': plm_val_out,
                'val_data': [item[outType]['token_extra_info']['valLst'] for item in batch],
                'token_extra_info': [item[outType]['token_extra_info'] for item in batch],
                'texts': [item[outType]['text'] for item in batch]
            }
        
        return batchOut
