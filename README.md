# CaVepP53:Causal prediction of TP53 variant pathogenicity using a perturbation-informed protein language model with interpretable insights from an AI agent

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://github.com/HuangLab-Bioinformatics-zju/CaVepP53)


## Introduction
CaVepP53 is a TP53-specific variant effect prediction (VEP) model built on the ESMC protein language model and fine-tuned with perturbation-based functional assay data. It enables accurate and interpretable classification of TP53 missense variants by capturing gene-specific mutational signatures and supporting causal inference. The model significantly outperforms general predictors such as AlphaMissense and PrimateAI-3D, and has been experimentally validated with 22 mutations, including 7 novel functional variants.
<p align="center">
<img src="./figures/model framework.tif" alt="The framework" style="width:20cm; height:auto;"/>
</p>
CaVepP53-Agent is an AI-driven interpretation framework that integrates large language models (LLMs) with structured biological data sources (e.g., PubMed, TP53 mutation database, UniProt, structural annotations). It generates detailed variant analysis reports to assist in mechanistic understanding and hypothesis generation. Together, the model and agent provide a scalable, gene-focused solution for variant interpretation and precision medicine research.
<p align="center">
<img src="./figures/Agent.tif" alt="Agent" style="width:20cm; height:auto;"/>
</p>
<details open><summary><b>Table of contents</b></summary>

- [CaVepP53-Model](#CaVepP53-Model)
  - [Data](#data)
  - [Model Weights and Usage](#Usage)
  - [Example](#Example)
- [CaVepP53-Agent](#CaVepP53-Agent)
  - [Pre-trained Models](#available-models)
- [Citations](#citations)
- [License](#license)
</details>




## CaVepP53-Model <a name="CaVepP53-Model"></a>
### Data <a name="data"></a>

This repository contains the data and model for predicting the pathogenicity of TP53 variants using a our gene-specific protein language model, CaVepP53.
The datasets used for training and evaluating the model are provided"[.datas/train_dataset](.datas/train_dataset)".
We have conducted saturation mutagenesis on the TP53 sequence using the fine-tuned CaVepP53. The results of this analysis are included in this repository"[.datas/TP53_mutagenesis_predictions.csv](.datas/TP53_mutagenesis_predictions)".

### Model Weights and Usage <a name="Usage"></a>
To utilize the CaVepP53 model for predicting the pathogenicity of TP53 variants, you will first need to obtain the model weights and follow the steps below to run the model.
<pre>
#Clone the Repository
git clone https://github.com/HuangLab-Bioinformatics-zju/CaVepP53.git

# Create the CaVepP53 environment
conda create --name CaVepP53 python=3.8

# Activate the environment
conda activate CaVepP53

# Install the pip packages
pip install -r requirements.txt    
# You can add the other pip packages from the dyna.yml file one by one if needed
python score.py --checkpoint_path "chenyuhe/CaVepP53" --input_path './datas/example_pre_data.csv' --output_path './datas/example_outcome.csv'
#You may substitute the prediction data with your own CSV file

</pre>

If you want to fine-tune the model on your own data, we also provide an example"[examples/futune.ipynb](examples/futune.ipynb)".

## CaVepP53-Agent <a name="CaVepP53-Agent"></a>
### Setup
```bash
cd Agent
```
modify the `.env` file (src/.env) with your own API keys:
#### DeepSeek models (https://platform.deepseek.com/api_keys)
DeepSeek_API_KEY=<your-deepseek-api-key-here>
#### PubMed Email
ENTREZ_EMAIL=<your-email-here>

If you haven't installed the required packages yet, run the following command to install them:
<pre>
# Install the pip packages
pip install -r requirements.txt   
</pre>

### Usage
```bash
streamlit run launch.py
```
The app will open in your default web browser at the local host.

### Interact with the CaVepP53-Agent

You can start interacting with the agent by typing messages in the chat interface
Example prompts you can try:
```
"I am interested in investigating the TP53 R175H variant."
```
The intelligent agent will answer your questions and can perform analyses based on predefined workflows.

