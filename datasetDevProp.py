# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Dataset classes able to generate prompts dynamically and add biases 

v. 7
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm import tqdm




#%% Parameters
# Dataset containing antibodies and properties. Follows the format from https://github.com/csi-greifflab/developability_profiling/blob/main/data/native/developability.csv
PROP_FILE = '../developability_profiling/data/native/developability.csv' 
SAMPLES_NUM = 50000
RND_SEED = 1234

# set numpy random seed for reproducibility
np.random.seed(RND_SEED)

# dataset Dev Prop Class
class DevPropData():
    def __init__(self, inFile=PROP_FILE):
        print(f"Loading file {inFile}")
        self.propFr = pd.read_csv(inFile, index_col=0)


        self.colsPropMapLbl = {"AbChain_solubility": "Solubility", 
                        "AbChain_min_rank": "Immunogenicity", 
                        'AbChain_instaindex': "Stability 1 (insta)", 
                        'AbChain_aliphindex': "Stability 2 (aliph)", 
                        'AbChain_hydrophobicity': "Hydrophobicity",
                        'AbStruc_pcharge_hetrgen': "PosCh heterogeneity", 
                        'AbStruc_ncharge_hetrgen': "NegCh heterogeneity"
                        }
        
        # invert keys values of self.colsPropMapLbl
        self.colsPropMapLblInv = {v: k for k, v in self.colsPropMapLbl.items()}
        
        self.colsPropOrig = list(self.colsPropMapLbl.keys())
        self.colsProp = list(self.colsPropMapLbl.values())

    def customNormalizationProp(self, normDict):
        """
        Normalize the properties using the provided normalization dictionary.
        normDict example: {"property name 1": {"min": minValue, "maxValue"}, 
                           "property name 2": {"min": minValue, "maxValue"},}

        property name is the user friendly name, such as "Hydrophobicity" rather than AbChain_hydrophobicity
        """


        for prTmp in normDict:
            pr = self.colsPropMapLblInv[prTmp]
            minVal = normDict[prTmp]['min']
            maxVal = normDict[prTmp]['max']
            self.propFr[pr] = (self.propFr[pr] - minVal) / (maxVal - minVal)

        print("Properties normalized")

        


    def get_sets_exp1(self, maxSamples=SAMPLES_NUM, rnd_seed=RND_SEED):
        """
        Get the sets for training and testing for experiment 1. i.e.

        * pre-train with AB-set-A and property-set-A
        * evaluate using
            * a variable number of AB context using  AB-set-B + property-set-A
            * querying property-set-A for unseen ABs (AB-set-C)
        """


        # filter heavy chain
        print(f"Select heavy chains only")
        propFiltFr = self.propFr[self.propFr['chain_type'] =='heavy']


        print(f"Sample n={maxSamples} from file")
        if maxSamples is None:
            # use all samples
            propFiltFr = propFiltFr
        else:
            # sample from the filtered dataframe
            propFiltFr = propFiltFr.sample(n=maxSamples, random_state=rnd_seed)

        # rename properties columns
        propFiltFr.rename(columns=self.colsPropMapLbl, inplace=True)


        # split into set A, B and C 
        setAFr = propFiltFr.sample(frac=0.7, random_state=rnd_seed)
        setBCFr = propFiltFr.drop(setAFr.index)
        setBFr = setBCFr.sample(frac=0.5, random_state=rnd_seed)
        setCFr = setBCFr.drop(setBFr.index)

        
        return setAFr, setBFr, setCFr

    def abTrTemplate(self, row, propSubset, isAnswer=False, biasDic=None):
        """
        Create context or answer antibody prompt text template

        biasDic (dictionary): prop_name -> bias_val containing the fixed bias to add to each property. if None do not add bias

        return template or None if none of the properties is valid
        """
        seq = row['sequence']

        abTagStart = '<antibody>'
        abTagEnd = '</antibody>'

        if isAnswer:
            abTagStart = '<ans-antibody>'
            abTagEnd = '</ans-antibody>'

        antibody_text = (
            f"{abTagStart}"
            f"<seq>{seq}</seq>"
            # f"<seq-l>{seq_l}</seq-l>" # do not use for now TODO: add later
        )
        numOfValProp = 0
        for pr in propSubset:
            property_name = pr
            property_value = row[pr]

            # check if property value can be converted to float
            # if not skip
            try:
                property_value = float(property_value)
                if np.isnan(property_value):
                    continue
                # add bias to property value
                if biasDic is not None:
                    property_value += biasDic[property_name]
            except (ValueError, TypeError):
                continue
            
            propNameEsc = property_name.replace('"','\\"')
            antibody_text = (
                f"{antibody_text}"
                f"<property name=\"{propNameEsc}\">{property_value:.3f}</property>"
            )

            numOfValProp += 1
        # return None if no properties are available
        if numOfValProp == 0:
            return None

        antibody_text = (
            f"{antibody_text}"
            f"{abTagEnd}"
        )

        return antibody_text

    def abQueryTemplate(self, row, propSubset):
        """
        Create "query" antibody prompt text template
        """

        seq = row['sequence']
        q_antibody_text = (
            f"<query-antibody>"
            f"<seq>{seq}</seq>"
            # f"<seq-l>{seq_l}</seq-l>" # do not use for now TODO: add later
        )
        numOfValProp = 0
        for pr in propSubset:
            property_name = pr
            property_value = row[pr]

            # check if property value can be converted to float
            # if not skip
            try:
                property_value = float(property_value)
                if np.isnan(property_value):
                    continue
            except (ValueError, TypeError):
                continue


            propNameEsc = property_name.replace('"','\\"')
            q_antibody_text = (
                f"{q_antibody_text}"
                f"<property name=\"{propNameEsc}\" />"
            )
            numOfValProp += 1

        # return None if no properties are available
        if numOfValProp == 0:
            return None
        
        q_antibody_text = (
            f"{q_antibody_text}"
            f"</query-antibody>"
        )


        return q_antibody_text

    def gen_train_prompts(self, setIn, propSubset='all', 
                          abNumRangePerPrompt=[5,50], 
                        #   biasVal=0, # removed, not useful
                          useRandMaxBias=0, 
                          propAnsSubset=None,
                          custRandMaxBiasPerProp=False,
                          includeQueryAbInCntx=False ): 
        """
        Generate prompts for training

        propSubset:  use a specific property subset for the context. Set to 'all' for using all properties.
        propAnsSubset: use a specific property subset for the query/answer. Set to None to use the same subset as the one used in propSubset
        useRandMaxBias (float): if true, add fixed random bias value, up until useRandMaxBias. Bias is constant at prompt level (including answer), if custRandMaxBiasPerProp=false, if custRandMaxBiasPerProp=True, see below 
        propAnsSubset (list): subset of properties to be used as answer/query. Note that each prompt will only have a single property as anaswer. If multiple properties are present a randoom one will be picked
        custRandMaxBiasPerProp (bool): if true,  add fixed random bias value, up until useRandMaxBias, for each individual property. i.e. each prompt will have N different biases, where N is the number of properties in the prompt
        includeQueryAbInCntx: # if True, it will include  the query AB in the context but only for the set of properties set(propSubset)-set(propAnsSubset in the prompt) ):
        """

        print("Generate prompts")

        setCurr = setIn.copy()
        # shuffle rows
        setCurr = setCurr.sample(frac=1, random_state=RND_SEED)

        if propSubset == 'all':
            propSubset = self.colsProp


        if propAnsSubset is None:
            propAnsSubset = propSubset
        

        # concatenate properties available
        propUsed = list(set(propSubset).union( set(propAnsSubset) ))
        # create a number of prompts IDs in range for num of AB used for each prompt 
        print("create a number of prompts IDs in range for num of AB used for each prompt ")

        ##======= Original
        # abIdxAvailable = setCurr.index.tolist()
        # abIdxBatches = []
        # with tqdm(total=len(abIdxAvailable)) as pbar:
        #     while(len(abIdxAvailable)>0):
        #         abIdxBatches.append([]) # add new batch
        #         abNumToUse = np.random.randint(abNumRangePerPrompt[0], abNumRangePerPrompt[1]+1)
        #         for i in range(abNumToUse):
        #             if len(abIdxAvailable) > 0:
        #                 abIdxToUse = np.random.choice(abIdxAvailable)
        #                 abIdxAvailable.remove(abIdxToUse)
        #                 abIdxBatches[-1].append(abIdxToUse)

        #                 pbar.update(1)
        ##======= 

        #====optimized version 
        abIdxAvailable = setCurr.index.tolist()
        abIdxBatches = []
        # Shuffle once at the beginning
        abIdxShuffled = list(abIdxAvailable)
        random.shuffle(abIdxShuffled)

        with tqdm(total=len(abIdxShuffled)) as pbar:
            idx = 0
            while idx < len(abIdxShuffled):
                abNumToUse = np.random.randint(abNumRangePerPrompt[0], abNumRangePerPrompt[1]+1)
                abNumToUse = min(abNumToUse, len(abIdxShuffled) - idx)

                # Slice the pre-shuffled list
                batch = abIdxShuffled[idx:idx + abNumToUse]
                abIdxBatches.append(batch)

                idx += abNumToUse
                pbar.update(abNumToUse)

        #====




        print("create prompt per batch")
        prompts = [] # initial prompts list
        expectedAns = [] # expected answer
        for bLst in tqdm(abIdxBatches):
            bLst = bLst.copy() # copy to avoid modifying the original list
            qIdxToUse = np.random.choice(bLst)
            bLst.remove(qIdxToUse)


            if useRandMaxBias:
                # Create dictionary from propUsed list with property name as key and empty list as value
                biasValDi = {prop: None for prop in propUsed}
                if custRandMaxBiasPerProp:
                    for p in propUsed:
                        # generate a random float number from 0 to useRandMaxBias
                        biasValTmp = np.random.uniform(0, useRandMaxBias)
                        # use only two decimal points
                        biasValTmp = np.round(biasValTmp, 2)
                        # store bias
                        biasValDi[p] = biasValTmp
                else:
                    # generate a random float number from 0 to useRandMaxBias
                    biasValAllPrompt = np.random.uniform(0, useRandMaxBias)
                    # use only two decimal points
                    biasValAllPrompt = np.round(biasValAllPrompt, 2)
                    for p in propUsed:
                        # store bias
                        biasValDi[p] = biasValAllPrompt
            else:
                # do not use any bias
                biasValDi=None

                
            # pick one random property to query
            propAnsSubsetSel = [propAnsSubset[np.random.randint(0,len(propAnsSubset))]]

            ansTxt = self.abTrTemplate(setCurr.loc[qIdxToUse], propAnsSubsetSel, isAnswer=True, biasDic=biasValDi)
            if ansTxt is None: # remove full prompt if the answer does not contain valid properties to predict
                continue

            # create prompt text
            antiBodyTxt = ''
            for abIdx in bLst:
                row = setCurr.loc[abIdx]
                currTempTxt = self.abTrTemplate(row, propSubset, biasDic=biasValDi)
                if currTempTxt is None: # do not include context antibody if no properties are available  
                    continue

                antiBodyTxt += currTempTxt



            # add query antibody to context (without the properties in the query)
            if includeQueryAbInCntx:
                propTmpLst = list(set(propSubset)-set(propAnsSubsetSel))
                tmpTxtAb = self.abTrTemplate(setCurr.loc[qIdxToUse], propTmpLst, biasDic=biasValDi)
                if tmpTxtAb is not None:
                    antiBodyTxt += tmpTxtAb


            promptTxt = (
                f"{antiBodyTxt}"
                f"{self.abQueryTemplate(setCurr.loc[qIdxToUse], propAnsSubsetSel)}"
            )
            

            prompts.append(promptTxt)
            expectedAns.append(ansTxt)

            pass

        return prompts, expectedAns


if __name__ == "__main__":
    # Example usage
    # # Set device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")

    #=== Load dataset using DevPropData
    devPropData = DevPropData()


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


    devPropData.customNormalizationProp(normDict)

    maxAbSamplesInTotal = 100
    propToUseLst = ['Solubility', 'Hydrophobicity']
    abNumRangePerPrompt = [5,5]
    maxBias = 0.5

    setA, setB, setC = devPropData.get_sets_exp1(maxSamples=maxAbSamplesInTotal)
    
    # Generate training prompts and answers
    prompts, answers = devPropData.gen_train_prompts(setA, 
                                                    propSubset=propToUseLst, 
                                                    abNumRangePerPrompt=abNumRangePerPrompt,
                                                    useRandMaxBias=maxBias, custRandMaxBiasPerProp=True)
    # prompts, answers = devPropData.gen_train_prompts(setA, 
    #                                                 propSubset=propToUseLst, 
    #                                                 abNumRangePerPrompt=abNumRangePerPrompt,
    #                                                 useRandMaxBias=maxBias, custRandMaxBiasPerProp=False)    

    ##============================= Unit test for generated bias 
    import re
    for i in range(len(prompts)):
        s = prompts[i]
        print(f'=========== new prompt')

        ansS = answers[i]
        # Find each antibody or query-antibody section
        blocks = re.findall(r"<antibody>(.*?)<\/antibody>", s)

        results = []
        for content in blocks:
            print(f'==seq')
            # extract sequence
            seq_match = re.search(r"<seq>(.*?)<\/seq>", content)
            sequence = seq_match.group(1) if seq_match else None
            
            # Extract property name & value
            prop_matches = re.finditer(r'<property name="([^"]+)">(.*?)<\/property>', content)
            for prop_match in prop_matches:
                prop_name, prop_value = (prop_match.group(1), prop_match.group(2))
                # results.append((sequence, prop_name, prop_value))          

                # find matching prop in set
                # propOrig = devPropData.propFr[devPropData.propFr['sequence']== sequence][devPropData.colsPropMapLblInv[prop_name]].values[0]
                propOrigArr = setA[setA['sequence']== sequence][prop_name].values
                assert(len(propOrigArr) == 1)
                propOrig = propOrigArr[0]

                # print bias
                print(f"bias - {prop_name}: {float(prop_value) - propOrig:.3f}")

        #== get answer property+value
        ans_prop_match = re.search(r'<property name="([^"]+)">(.*?)<\/property>', ansS)
        ans_prop_name, ans_prop_value = (ans_prop_match.group(1), ans_prop_match.group(2))

        #get answer antibody sequence
        print("==answer antibody")
        ans_seq_match = re.search(r"<ans-antibody>.*<seq>(.*?)<\/seq>.*<\/ans-antibody>", ansS)
        ans_sequence = ans_seq_match.group(1) if ans_seq_match else None
        # answer property value
        ansPropOrigArr = setA[setA['sequence']== ans_sequence][ans_prop_name].values
        assert(len(ansPropOrigArr) == 1)
        ansPropOrig = ansPropOrigArr[0]

        print(f"bias answer - {ans_prop_name}: {float(ans_prop_value) - ansPropOrig:.3f}")

        #==


    pass
