# Analyzing LLM Instruction Optimization for Tabular Fact Verification

This repository contains data and codes used for the experiments in the paper: [Analyzing LLM Instruction Optimization for Tabular Fact Verification
](https://aclanthology.org/2026.findings-eacl.161.pdf).

## Environment Setup
Clone the repository and install the required dependencies by the following commands. We use [DSPy framework](https://github.com/stanfordnlp/dspy) for instruction optimization. 
```
conda create -n tabfact python=3.10
conda activate tabfact
pip install -r requirements.txt 
```
Set your API keys, hyperparameters, and paths to the datasets in `config/config.yaml`. 

## Datasets
Scripts for data preprocessing can be found in `scripts/process_data` directory. We use the following table-based fact verification datasets for evaluation. The original data can be downloaded from the official repositories. 

- [TabFact](https://github.com/wenhuchen/Table-Fact-Checking)
- [PubHealthTab](https://github.com/mubasharaak/PubHealthTab)
- [SciTab](https://github.com/XinyuanLu00/SciTab)
- [MMSci-Table](https://github.com/Bernard-Yang/MMSci_Table)

We cleaned SciTab data by removing the special tags in the claims and tables. We created a hybrid train dataset with balanced label distribution by sampling instances from TabFact, PubHealthTab and SciTab. We optimize LLM instructions on the hybrid train data, and evaluate the performance of the optimized instructions on test data.

## Experiments
### 1. Instruction optimization
Optimize LLM instructions on hybrid train data:
```
python optimise.py --module [Method] \
    --model_name [Base model] \
    --port [vLLM port number] \
    --dataset hybrid \
    --optimiser [Optimization method] [Optimizer config]
```
The optimized instructions will be saved in `artifacts/models` directory.

### 2. Evaluate optimized instructions
Evaluate the model with optimized instructions on test data:
```
python eval.py --module [Method] \
    --model_name [Base model] \
    --port [vLLM port number] \
    --dataset [Dataset] \
    --split test \
    --load_path [Optimized model]
```
Set `--load_path` as the path to `optimised_program.json` file that stores optimized instructions and relevant meta data. The prediction results will be saved in `artifacts/results` directory. 

Check example usage in the bash scripts in `scripts/experiment` directory: 
```
cd scripts/
./experiment/optimise_single_parallel.sh
```

### 3. Experiment configuration
The following settings are supported in the current implementation:
- Base models: GPT-4o/4o-mini, Qwen3-8B/32B, Gemma3-12B/27B
- Prompting methods: Direct Prompting, CoT, ReAct, CodeAct
- Instruction optimizers: COPRO, MiPROv2, SIMBA

## Citation
```bibtex
@inproceedings{du-etal-2026-analyzing,
    title = "Analyzing {LLM} Instruction Optimization for Tabular Fact Verification",
    author = "Du, Xiaotang  and
      Hong, Giwon  and
      Kwan, Wai-Chung  and
      Saxena, Rohit  and
      Titov, Ivan  and
      Minervini, Pasquale  and
      Allaway, Emily",
    editor = "Demberg, Vera  and
      Inui, Kentaro  and
      Marquez, Llu{\'i}s",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EACL} 2026",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-eacl.161/",
    doi = "10.18653/v1/2026.findings-eacl.161",
    pages = "3078--3108",
    ISBN = "979-8-89176-386-9"
}
```