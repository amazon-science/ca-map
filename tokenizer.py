# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Custom tokenizer v.3

Combines sentence + plm tokenizers and add additional custom tokens

"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import math
import re
import xml.etree.ElementTree as ET

from datasetDevProp import DevPropData
from torchtext.vocab import FastText
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from typing import Union

import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict


"""
Example prompt:

'<antibody>*
    <seq>QVHVKQSGPELVKPGASVKLSCKASGYTFTSYDINWVKQRPGQGLEWIGWIYPRDGSTKYNEKFKGKATLTVDTSSITAYMELHSLTSEDSAVDVCARLEFDGSSGDWYFDIWGTGTTVTVTS</seq>
    <property name="Hydrophobicity">0.30</property> * *
    <property name="NegCh heterogeneity">0.43</property> * *
    <property name="Solubility">0.88</property> * *
    <property name="Stability 2 (aliph)">0.52</property> * *
</antibody>*
<antibody>
    <seq>QVQLQQPGAELVKPGASVKLSCKASGYTFTNYWMHLVKQRPGQGLEWIGMMHPNGGSPDYNEKFKSEATLSVDKSSRTAYIELSSLTSEDSAVYYCARSYDYDDYTMDYWGQGTSVTVSS</seq>
    <property name="Hydrophobicity">0.24</property>
    <property name="NegCh heterogeneity">0.24</property>
    <property name="Solubility">0.94</property>
    <property name="Stability 2 (aliph)">0.40</property>
</antibody>
<query-antibody> *
    <seq>MSGGSLRLSCAASGFTFSSYDMHWVRQATGRGLEWVSAIGTAADSYYSGSVKGRFTVSRDNAKNSFYLQMNSLRAGDTAVYYCARVALPRECTSTSCSDSGYYFDYWGQGTLVTVSS</seq> * *
    <property name="Stability 2 (aliph)" /> * *
</query-antibody> *'
"""

"""
Example answer: 0.37 
"""


#%%
# # Define a custom vocabulary for trainable subToken and subtoken-to-index mapping
# subTokVocab = ['query-AB-tok', 'prop-tok', 'unknown-value-tok', 
#                'AB-tok', 'seq-tok', 'seq-l-tok', 'eos-tok',
#                 'ans-AB-tok' ]
# subTok_size = len(subTokVocab)
# subTok_to_idx = {token: idx for idx, token in enumerate(subTokVocab)}


class CustomTokenizer:
    # custom tokenizer padding index
    customPad = 0 

    # static variables for additional tokens
    customTokVocab = ['pad-tok', # (see custom tokenizer padding index)
                    'query-AB-st-tok', 'query-AB-end-tok',
                   'AB-st-tok','AB-end-tok',
                   'seq-st-tok','seq-end-tok',
                   'seq-l-st-tok','seq-l-end-tok',
                   'prop-name-st-tok','prop-name-end-tok', 
                   'prop-val-st-tok','prop-val-end-tok',
                   'sentence-placeholder', # placeholder to be substituted with sentence embedding
                   'seq-placeholder',  # placeholder to be substituted with sentence embedding
                   'value-placeholder', # placeholder to be substituted with value embedding
                   'eos-tok',
                   'pred-tok', 
                   'ans-AB-st-tok', 'ans-AB-end-tok' # not used if linear layer is used for prediction
                ]
    customTokVocab_size = len(customTokVocab)

    # types of tok sources (TODO: delete them if we only placeholder vocadulary are used)
    SRC_T_CUST = 0 
    SRC_T_PLM = 1 
    SRC_T_TXT = 2 
    SRC_T_VAL = 3 
    # tokSourceTypes = ['cust', 'plm', 'text', value ] 


    def __init__(self, paramDic: Dict[str, Union[int, float, str, bool]] ):
        self.globParam = paramDic
        
        self.plmTokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D") 
        # self.sentenceModel = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.sentenceTokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # find padding values
        self.plmPadIdx = self.plmTokenizer.pad_token_id
        self.sentPadIdx = self.sentenceTokenizer.pad_token_id

        # Create vocabulary
        self.fullTok_to_idx = {token: idx for idx, token in enumerate(self.customTokVocab)}
        self.vocab_size = len(self.customTokVocab)

    

    def __call__(self, textIn: str):
        """
        Tokenize the input text, return a list of tokens and extra data required for embeddings.

        """
        # init output
        tokIndices = []
        tokTypesLst = []
        sentTxtLst = []
        sentTokData = []
        seqLst = []
        seqTokData = []
        valLst = []
        ansExtraData = []

        # start to add special prediction token
        tokIndices.append(self.fullTok_to_idx['pred-tok'])
        tokTypesLst.append(self.SRC_T_CUST)


        # parse text
        abInfoDic = self.parseInputText(textIn)
        # if error return None, None
        if abInfoDic is None:
            return None, None
        

        #===== Query antibody tokens 
        if ('query-antibody' in abInfoDic.keys())  and (len(abInfoDic['query-antibody'])>0):
            # get query antibody info (for first antibody only)
            queryAbInfo = abInfoDic['query-antibody'][0]

            # append start query AB tok
            tokIndices.append(self.fullTok_to_idx['query-AB-st-tok'])
            tokTypesLst.append(self.SRC_T_CUST)


            # seq
            tokIndices.extend([self.fullTok_to_idx['seq-st-tok'], 
                                self.fullTok_to_idx['seq-placeholder'],
                                self.fullTok_to_idx['seq-end-tok']])
            tokTypesLst.extend([self.SRC_T_CUST,self.SRC_T_PLM,self.SRC_T_CUST])
            seqLst.append(queryAbInfo['seq'])

            if 'seq-l' in queryAbInfo.keys():
                # seq
                tokIndices.extend([self.fullTok_to_idx['seq-l-st-tok'], 
                                   self.fullTok_to_idx['seq-placeholder'],
                                   self.fullTok_to_idx['seq-l-end-tok']])
                tokTypesLst.extend([self.SRC_T_CUST,self.SRC_T_PLM,self.SRC_T_CUST])
                seqLst.append(queryAbInfo['seq-l'])
                

            # append end query AB tok
            tokIndices.append(self.fullTok_to_idx['query-AB-end-tok'])
            tokTypesLst.append(self.SRC_T_CUST)

            # get number of properties (query antibody always have at least one property)
            propNum = len(queryAbInfo['properties'])
            for propId in range(propNum):
                # get property info
                propInfo = queryAbInfo['properties'][propId]

                # get property name
                propName = propInfo['name']

                # create tokens
                # tokPrpNameTmp =  [self.fullTok_to_idx[tok] for tok in queryAbInfo['seq-l']]
                tokIndices.extend([self.fullTok_to_idx['prop-name-st-tok'], 
                                   self.fullTok_to_idx['sentence-placeholder'], 
                                   self.fullTok_to_idx['prop-name-end-tok']])
                tokTypesLst.extend([self.SRC_T_CUST, self.SRC_T_TXT,self.SRC_T_CUST])

                # add extra sentence info
                sentTxtLst.append(propName)
        #=====
        #===== Context antibody tokens
        if ('antibody' in abInfoDic.keys()) and (len(abInfoDic['antibody'])>0):
            # get number of antibodies
            abNum = len(abInfoDic['antibody'])
            for abId in range(abNum):
                # get antibody info
                abInfo = abInfoDic['antibody'][abId]
                # # pos value for embedding
                # posAb = abId / (abNum - 1) if abNum > 1 else 0


                # append start  AB tok
                tokIndices.append(self.fullTok_to_idx['AB-st-tok'])
                tokTypesLst.append(self.SRC_T_CUST)


                # seq
                tokIndices.extend([self.fullTok_to_idx['seq-st-tok'], 
                                    self.fullTok_to_idx['seq-placeholder'],
                                    self.fullTok_to_idx['seq-end-tok']])
                tokTypesLst.extend([self.SRC_T_CUST,self.SRC_T_PLM,self.SRC_T_CUST])
                seqLst.append(abInfo['seq'])

                if 'seq-l' in abInfo.keys():
                    # seq
                    tokIndices.extend([self.fullTok_to_idx['seq-l-st-tok'], 
                                    self.fullTok_to_idx['seq-placeholder'],
                                    self.fullTok_to_idx['seq-l-end-tok']])
                    tokTypesLst.extend([self.SRC_T_CUST,self.SRC_T_PLM,self.SRC_T_CUST])
                    seqLst.append(abInfo['seq-l'])
                    
                # append end AB tok
                tokIndices.append(self.fullTok_to_idx['AB-end-tok'])
                tokTypesLst.append(self.SRC_T_CUST)

                # get number of properties (context antibody always have at least one property)
                propNum = len(abInfo['properties'])
                for propId in range(propNum):
                    # get property info
                    propInfo = abInfo['properties'][propId]

                    # get property name
                    propName = propInfo['name']
                    # get property value
                    propValue = propInfo['value']

                    # create tokens
                    tokIndices.extend([self.fullTok_to_idx['prop-name-st-tok'], 
                                        self.fullTok_to_idx['sentence-placeholder'], 
                                        self.fullTok_to_idx['prop-name-end-tok'],
                                        self.fullTok_to_idx['prop-val-st-tok'], 
                                        self.fullTok_to_idx['value-placeholder'], 
                                        self.fullTok_to_idx['prop-val-end-tok']
                                        ])
                    tokTypesLst.extend([self.SRC_T_CUST, self.SRC_T_TXT,self.SRC_T_CUST,
                                        self.SRC_T_CUST, self.SRC_T_VAL,self.SRC_T_CUST])

                    # add extra sentence info
                    sentTxtLst.append(propName)
                    # add extra value info
                    valLst.append(propValue)


        #=====

        #===== Answer antibody (do not tokenize, only add answers)
        if ('ans-antibody' in abInfoDic.keys()) and (len(abInfoDic['ans-antibody'])>0):
            # get query antibody info (for the first antibody only)
            ansAbInfo = abInfoDic['ans-antibody'][0]

            # get number of properties (query antibody always have at least one property)
            propNum = len(ansAbInfo['properties'])
            for propId in range(propNum):
                # get property info
                propInfo = ansAbInfo['properties'][propId]

                # get property name
                propName = propInfo['name']
                propValue = propInfo['value']

                # tokIndices.append(self.fullTok_to_idx['ans-AB-st-tok'])
                ansExtraData.append({'name': propName, 'value': propValue})
                # tokExtraData.app
        #====
        #     ##====== Remove sequence tokens in answer to simplify the training 
        #     # # get sequence (answer AB always should always have a seq field)
        #     # tokIndices.append(self.fullTok_to_idx['ans-AB-st-tok'])
        #     # tokExtraData.append({'seq': ansAbInfo['seq']})

        #     # if 'seq-l' in ansAbInfo.keys():
        #     #     # get sequence-l
        #     #     tokIndices.append(self.fullTok_to_idx['seq-l-st-tok'])
        #     #     tokExtraData.append({'seq': ansAbInfo['seq-l']})

        #     # get number of properties (query antibody always have at least one property)
        #     propNum = len(ansAbInfo['properties'])
        #     for propId in range(propNum):
        #         # get property info
        #         propInfo = ansAbInfo['properties'][propId]

        #         # get property name
        #         propName = propInfo['name']
        #         propValue = propInfo['value']

        #         tokIndices.append(self.fullTok_to_idx['ans-AB-st-tok'])
        #         tokExtraData.append({'name': propName, 'value': propValue})
        #=====

        #===== add eos token if there is some data
        if len(tokIndices) == 0:
            print("no data found")
            return None, None
        
        tokIndices.append(self.fullTok_to_idx['eos-tok'])
        tokTypesLst.append(self.SRC_T_CUST)
        #=====

        # tokenize all sentences at once
        if len(sentTxtLst)>0:
            sentTokData.append(self.sentenceTokenizer( sentTxtLst ))
        else:
            sentTokData.append(None)


        # tokenize all sequences at once
        if len(seqLst)>0:
            seqTokData.append(self.plmTokenizer( seqLst ))
        else:
            seqTokData.append(None)



        fullExtraData = {'tokSourceTypes': tokTypesLst, 
                         'sentTxt': sentTxtLst,
                         'sentTokData': sentTokData,
                         'seqLst': seqLst,
                         'seqTokData': seqTokData,
                         'valLst': valLst,
                         'ansExtraData': ansExtraData }

        return tokIndices, fullExtraData

    def parseInputText(self, text: str):
        """
        Parse train/test data from text format.
        Returns the  sequence as full tok IDs, + other data required for embeddings/subtokens 
        group by prompt and expected answer (if present).
        """

        textSample = """
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

            <ans-antibody>
                <seq>GIDNPWRFNDISVKCWMIY</seq><seq-l>FKAQKNCQTW</seq-l>
                <property name="thermostability">0.89</property>
            </ans-antibody>
        """

        # #TODO: remove
        # text = textSample

        # add root element (to make it compatible with XML parser)
        text = "<root>" + text + "</root>"

        # dictionary of information of antibody (AB) found  by antibody tags. Tag names are used as keys
        # to distinguish between context AB,  query AB, expected AB
        abInfoDic = {"antibody": [], "query-antibody": [], "ans-antibody": []}



        #===== Parse the XML text
        # return None if it is not
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            print("Error: Invalid XML format")
            return None        
        
        for abTag, abInfo in abInfoDic.items():
            abLst = root.findall(abTag)
            for ab in abLst:
                antibody = {}

                # Extract properties
                properties = ab.findall('property')
                antibody['properties'] = []
                for prop in properties:
                    property_name = prop.get('name')
                    property_value = prop.text
                    property_value = float(property_value) if property_value else None
                    antibody['properties'].append({'name': property_name, 'value': property_value})

                
                # Extract sequence
                seq = ab.find('seq').text
                antibody['seq'] = seq

                # Extract light sequence (if available)
                resSeqL = ab.find('seq-l')
                if resSeqL is not None:
                    seq_l = resSeqL.text
                    antibody['seq-l'] = seq_l

                abInfo.append(antibody)
        #=====

        #===== Parse for validity of abInfoDic
        # all antibodies should have at least 'seq'
        for abTag, abInfo in abInfoDic.items():
            for ab in abInfo:
                if 'seq' not in ab:
                    print(f"Error: {abTag} does not have 'seq'")
                    return None
                
        # all antibodies should have at least one property
        for abTag, abInfo in abInfoDic.items():
            for ab in abInfo:
                if 'properties' not in ab or len(ab['properties']) == 0:
                    print(f"Error: {abTag} does not have 'properties'")
                    return None
        # query antibody should have all properties with value None
        for ab in abInfoDic['query-antibody']:
            for prop in ab['properties']:
                if prop['value'] is not None:
                    print(f"Error: query-antibody should not have property with value")
                    return None
        #=====
        
        return abInfoDic

if __name__ == "__main__":
    # Test input text
    test_text= """
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

            <ans-antibody>
                <seq>GIDNPWRFNDISVKCWMIY</seq><seq-l>FKAQKNCQTW</seq-l>
                <property name="thermostability">0.89</property>
            </ans-antibody>
    """


    # Create tokenizer instance with empty params
    tokenizer = CustomTokenizer({})

    # Test tokenization
    token_indices, token_extra_data = tokenizer(test_text)

    # Print results
    print("Token Indices:", token_indices)
    print("\nToken Extra Data:")
    for data in token_extra_data:
        print(data)


    test_prompt = '<query-antibody>' \
                '<seq>RYGESLKISCKASGYTFIGYYMHWVRQAPGRGLEWMGWINPDNGGTYYAEKFQGRIAMIRDTSINTVYMELSRLTSDDTAVYFCARGVGRTGIQARFFFWFDPWGQGTLVSVSS</seq>' \
                '<property name="Hydrophobicity" />' \
                '</query-antibody>'
    
    test_ans = '<ans-antibody>' \
                '<seq>RYGESLKISCKASGYTFIGYYMHWVRQAPGRGLEWMGWINPDNGGTYYAEKFQGRIAMIRDTSINTVYMELSRLTSDDTAVYFCARGVGRTGIQARFFFWFDPWGQGTLVSVSS</seq>' \
                '<property name="Hydrophobicity">0.07</property>' \
                '</ans-antibody>'