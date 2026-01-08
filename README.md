# CA-MAP 

This repository contains code to support the publication entitled "Context-aware Multi-Property Antibody Predictor: a Novel Framework Integrating Text and Protein Language Models".
The acronym of this repository stands for
CA-MAP: Context-aware Multi-Property Antibody Predictor 

## Overview

Recent advances in Machine Learning have transformed antibody development through in-silico models, accelerating therapeutic candidate identification. However, three critical challenges persist: rapid adaptation of property predictors to laboratory-specific assays with incomplete datasets; batch effects introducing systematic bias; assay costs necessitating efficient unseen property prediction.

We introduce a novel multi-modal architecture featuring specialized tokenization and embedding projection that integrates text and protein language models (pLM). Our framework enables prompting without dictionary merging across modalities, creating a compact model capable of in-context learning for multi-property prediction. The orchestrating model uses Mamba-based architecture with learnable tokens and projector layers, avoiding pLM-to-text projection while enabling inference-time adaptation without retraining.



This project implements a novel approach for antibody property prediction by combining:
- **Protein Language Models (PLM)**: ESM2 for sequence embeddings
- **Sentence Transformers**: For property name embeddings  
- **Mamba/Transformer Architecture**: For sequence modeling
- **In-Context Learning**: Few-shot prediction using context antibodies

The model predicts antibody developability properties such as hydrophobicity, solubility, stability, and charge heterogeneity from amino acid sequences. It aims learn correlation from existing properties provided in the context and predict new ones. It aims to to learn batch effect. 

## Architecture

### Multi-Modal Token Embedding
- **Custom Tokenizer**: Handles XML-formatted antibody data with placeholders
- **ESM2 Integration**: Protein sequence embeddings via facebook/esm2_t6_8M_UR50D
- **Sentence Embeddings**: Property name embeddings via paraphrase-multilingual-MiniLM-L12-v2
- **Value Embeddings**: Direct property value projections

### Model
Main model: State-space model implementation (`model_mamba.py`)

## Repository Structure

```
├── inference_exp/           # Directory containing experiments for inference and evaluation
├── models/                 # Trained model weights
├── runs/                   # TensorBoard logs and cached data (including processed dataset)
├── out/                    # Results of experiments 
├── train.py                 # Main training script
├── model_mamba.py          # Mamba model implementation + Embeddings 
├── dataset.py              # Dataset and data loading utilities
├── datasetDevProp.py       # Developability property data handling
├── tokenizer.py            # Custom tokenizer for multi-modal inputs

```

## Data Format
This codebased follows in-silico properties naming and data format  from https://github.com/csi-greifflab/developability_profiling/blob/main/data/native/developability.csv


It then simulates prompts representing antibodies with their properties:

```xml
<antibody>
    <property name="thermostability">0.95</property>
    <property name="solubility">0.5</property>
    <seq>WVVNGNRICENWLGTFNYHS</seq>
    <seq-l>ARESTIHWSCIAAYIAPSKEICVITWN</seq-l>
</antibody>
[...]
<antibody>
    <property name="thermostability">0.51</property>
    <seq>MCWDHGEPQPNFAETMDWDIYEV</seq>
    <seq-l>ISQVRPMHSFIITWSDVSYI</seq-l>
</antibody>


<query-antibody>
    <seq>GIDNPWRFNDISVKCWMIY</seq><seq-l>FKAQKNCQTW</seq-l>
    <property name="thermostability" />
</query-antibody>
```

## Dependencies

python environment with CUDA support can be created with conda as follows:
```
# create environment
conda env create -f requirements.yml
# alternatively if mamba package manager is installed, you can use (this has nothing to do with the mamba model architecture)
mamba env create -f requirements.yml

# activate environment with 
conda activate py-in-context-env
``` 

Note that: jupyter packages are not installed by default


### Training
edit parameters in train.py 
main parameters for an initial  training run  are: 
- expId: defining the ID of the experiment (which will be used to name output files, including weights)
- rawDataLocation: Dataset containing antibodies and properties. Follows the format in https://github.com/csi-greifflab/developability_profiling/blob/main/data/native/developability.csv


```bash
python train.py
```

weights will be saved in models/[expID]_[timestamp_start_training]_best_weights.pkl (best weights according to validation set) and models/[expID]_[timestamp_start_training].pkl (final weights after all epochs)
processed data splits and prompts will be saved in runs/tok/[expID]_proc_dataset.pkl
training information compatible with tensorboard be saved in runs/training_info/[expID]_[timestamp_start_training] . Tensorboard with training info and loss curves can be invoked while (or after) training as:
```
tensorboard --port [your_port] --logdir runs/training_info
```

### Inference/Experiments

The inference scripts can be divided by experiment and should be stored in the inference_exp directory. One set of experiments is provided in
inference_exp8.py 

note that it requires running the training, weights are not provided.

```bash
cd inference_exp
python inference_exp8.py
```

This will generate predictions with performance metrics and visualizations in the `out/[expID]` directory. Example outputs are included in `out/exp8`. Note that 'inference_exp8.py' will also create CSV files with raw scores and ground truth (not currently included in `out/exp8`) 


## Citation

TO BE ADDED

## License
This project is licensed under the [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en) License.