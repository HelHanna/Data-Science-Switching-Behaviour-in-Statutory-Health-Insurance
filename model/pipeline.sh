#!/bin/bash
# set your Huggingface token here:
# export HF_TOKEN=
echo "Train prediction model and start SHAP generation..."
python train_model.py

echo "Starting LLM explanation generation..."
python llm_explainations.py meta-llama/Llama-3.1-8B-Instruct

echo "Starting evaluation..."
python evaluation.py meta-llama/Llama-3.1-8B-Instruct

echo "evaluation matrix saved."