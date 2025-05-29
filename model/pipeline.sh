#!/bin/bash

echo "Train prediction model and start SHAP generation..."
python train_model.py

echo "Starting LLM explanation generation..."
python llm_explainations.py

echo "Starting evaluation..."
python evaluation.py

echo "evaluation matrix saved."