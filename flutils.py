import os
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from medpix2_dataset_preparation import medpix2_2050
import evaluate

def preprocess_function(example):
    if example["input"].strip():
        prompt = (
            f"### Instruction:\n{example['instruction']}\n"
            f"### Input:\n{example['input']}\n"
            "### Response:\n"
        )
    else:
        prompt = (
            f"### Instruction:\n{example['instruction']}\n"
            "### Response:\n"
        )
    full_text = prompt + example["output"]
    return {"text": full_text}

def tokenize_function(examples, tokenizer, max_length=512):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )

def load_dataset_fine(dataset_name, test_size=0.1, seed=42):
    if dataset_name == "alpaca":
        raw_dataset = load_dataset("tatsu-lab/alpaca")
    elif dataset_name == "hcm":
        raw_dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")
    elif dataset_name == "medpix2":
        raw_dataset = medpix2_2050()

    split_dataset = raw_dataset["train"].train_test_split(test_size=test_size, seed=seed)
    train_dataset = split_dataset["train"].map(preprocess_function)
    val_dataset = split_dataset["test"].map(preprocess_function)

    # sample only one example for validation
    # val_dataset = val_dataset.select([0])

    return train_dataset, val_dataset

def create_partitioned_datasets(tokenizer, dataset_name, partition_size, client_seed=None):
    cached_dir = "cached_datasets"
    os.makedirs(cached_dir, exist_ok=True)
    
    partitions_dir = os.path.join(cached_dir, "partitions")
    os.makedirs(partitions_dir, exist_ok=True)
    
    val_cache_path = os.path.join(cached_dir, "tokenized_val")
    
    partitions_created = False

    if not partitions_created or not os.path.exists(val_cache_path):

        for filename in os.listdir(partitions_dir):
            file_path = os.path.join(partitions_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        train_dataset_full, val_dataset = load_dataset_fine(dataset_name)
        
        train_dataset_full = train_dataset_full.map(
            lambda ex: tokenize_function(ex, tokenizer), batched=True
        )
        val_dataset = val_dataset.map(
            lambda ex: tokenize_function(ex, tokenizer), batched=True
        )
        
        val_dataset.to_json(val_cache_path)
        
        total_size = len(train_dataset_full)
        indices = list(range(total_size))
        random.shuffle(indices)
        
        partition_indices = [
            indices[i : i + partition_size] for i in range(0, total_size, partition_size)
        ]
        
        for i, indices in enumerate(partition_indices):
            partition = train_dataset_full.select(indices)
            partition_path = os.path.join(partitions_dir, f"partition_{i}")
            partition.to_json(partition_path)
            print(f"Partition {i} saved to {partition_path}")

    return partitions_dir, val_cache_path

def load_random_partition():
    cached_dir = "cached_datasets"
    partitions_dir = os.path.join(cached_dir, "partitions")
    partition_id = random.choice(os.listdir(partitions_dir))
    partition_path = os.path.join(partitions_dir, partition_id)
    print(f"Loading partition {partition_id}...\n")

    train_dataset = load_dataset("json", data_files=partition_path)["train"]
    
    print(f"Using partition {partition_id} with {len(train_dataset)} examples for this round\n")
    
    return train_dataset, partition_id

def get_model_tokenizer(modelname):
    if modelname == "llama":
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        model_dir = "tinyllama_model"
    elif modelname == "smol":
        model_name = "HuggingFaceTB/SmolLM2-135M"
        model_dir = "smollm_model"

    os.makedirs(model_dir, exist_ok=True)
    print(f"Using permanent model directory: {model_dir}\n")
    
    tokenizer_path = os.path.join(model_dir, "tokenizer")
    model_path = os.path.join(model_dir, "model")
    
    if os.path.exists(tokenizer_path) and os.path.exists(model_path):
        print("Loading model and tokenizer from local directory\n")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    else:
        print("Downloading model and tokenizer from HuggingFace\n")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        
        tokenizer.save_pretrained(tokenizer_path)
        model.save_pretrained(model_path)
        print("Model and tokenizer saved to local directory\n")
            
    tokenizer.pad_token = tokenizer.eos_token
    print(model)

    return model, tokenizer

def bleurt_metric(predictions, references):
    metric = evaluate.load("bleurt", "bleurt-base-128")
    scores = metric.compute(predictions=predictions, references=references)
    return scores
