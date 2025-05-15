#!/bin/bash

rm -rf ~/.cache/huggingface/datasets/
find . -type d -name "__pycache__" -exec rm -r {} + -o -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
rm -rf cached_datasets
rm -rf smollm-finetuned
rm -rf hcm_dataset
rm eval_outputs*
rm server_model_round*
rm server_eval_*
rm server_metrics*
rm client_eval_*
rm client_train_*