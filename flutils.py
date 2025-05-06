import os
import random
import torch
import psutil
import json
import flwr as fl
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from medpix2_dataset_preparation import medpix2_2050
import evaluate

class CudaClearingTrainer(Trainer):
    def training_step(self, *args, **kwargs):
        output = super().training_step(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return output

    def evaluation_step(self, *args, **kwargs):
        output = super().evaluation_step(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return output

def preprocess_function(example):
    eos_token = "<eos>"
    texts = []
    if isinstance(example["input"], list):
        for i in range(len(example["input"])):
            input_val = example["input"][i]
            instruction_val = example["instruction"][i]
            output_val = example["output"][i]
            prompt = (
                f"### Instruction:\n{instruction_val.strip()}\n"
                f"### Input:\n{input_val.strip() if input_val.strip() else ''}\n"
                f"### Response:\n"
            )
            full_text = prompt + output_val.strip() + " " + eos_token
            texts.append(full_text)
    else:
        prompt = (
            f"### Instruction:\n{example['instruction'].strip()}\n"
            f"### Input:\n{example['input'].strip() if example['input'].strip() else ''}\n"
            f"### Response:\n"
        )
        full_text = prompt + example["output"].strip() + " " + eos_token
        texts.append(full_text)
    return {"text": texts}

def tokenize_function(examples, tokenizer, max_length=512):
    tokens = tokenizer(
        examples["text"], truncation=True, max_length=max_length,
        padding="max_length", return_tensors="pt"
    )
    labels = tokens["input_ids"].clone()
    vocab_size = len(tokenizer)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
    eos_token_id = tokenizer.eos_token_id

    for i, text in enumerate(examples["text"]):
        response_start = text.find("### Response:\n")
        if response_start == -1:
            continue
        prefix = text[:response_start + len("### Response:\n")]
        prefix_tokens = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        response_token_start = len(prefix_tokens)

        if response_token_start >= len(labels[i]):
            response_token_start = len(labels[i]) - 1

        labels[i][:response_token_start] = -100

        for j, token_id in enumerate(labels[i]):
            if token_id != -100 and (token_id < 0 or token_id >= vocab_size):
                labels[i][j] = pad_token_id

    tokens["labels"] = labels
    return tokens

def load_dataset_fine(dataset_name, test_size=0.1, seed=42):
    if dataset_name == "alpaca":
        raw_dataset = load_dataset("tatsu-lab/alpaca")
    elif dataset_name == "hcm":
        raw_dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")
    elif dataset_name == "medpix2":
        raw_dataset = medpix2_2050()

    split_dataset = raw_dataset["train"].train_test_split(test_size=test_size, seed=seed)
    train_dataset = split_dataset["train"].map(preprocess_function, batched=True)
    val_dataset = split_dataset["test"].map(preprocess_function, batched=True)
    return train_dataset, val_dataset

def create_partitioned_datasets(tokenizer, partition_config, client_seed=None):
    cached_dir = "cached_datasets"
    os.makedirs(cached_dir, exist_ok=True)
    partitions_dir = os.path.join(cached_dir, "partitions")
    os.makedirs(partitions_dir, exist_ok=True)
    
    val_cache_path = os.path.join(cached_dir, "tokenized_val")
    
    if not os.path.exists(os.path.join(cached_dir, "partitions_config.json")):
        with open(os.path.join(cached_dir, "partitions_config.json"), "w") as f:
            json.dump(partition_config, f)

    partition_size = partition_config["partition_size"]
    if not os.path.exists(val_cache_path):
        train_dataset_full, val_dataset = load_dataset_fine(partition_config["dataset_name"])
        train_dataset_full = train_dataset_full.map(lambda ex: tokenize_function(ex, tokenizer), batched=True)
        val_dataset = val_dataset.map(lambda ex: tokenize_function(ex, tokenizer), batched=True)

        val_dataset.to_json(val_cache_path)
        
        total_size = len(train_dataset_full)
        indices = list(range(total_size))
        random.shuffle(indices)
        partition_indices = [indices[i: i + partition_size] for i in range(0, total_size, partition_size)]

        for i, indices in enumerate(partition_indices):
            partition = train_dataset_full.select(indices)
            partition_path = os.path.join(partitions_dir, f"partition_{i}")
            partition.to_json(partition_path)

    return partitions_dir, val_cache_path

def load_random_partition():
    cached_dir = "cached_datasets"
    partitions_dir = os.path.join(cached_dir, "partitions")
    partition_id = random.choice(os.listdir(partitions_dir))
    partition_path = os.path.join(partitions_dir, partition_id)
    train_dataset = load_dataset("json", data_files=partition_path)["train"]
    return train_dataset, partition_id

def get_model_tokenizer(modelname):
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" if modelname == "llama" else "HuggingFaceTB/SmolLM2-360M"
    model_dir = "tinyllama_model" if modelname == "llama" else "smollm_model"
    os.makedirs(model_dir, exist_ok=True)
    
    tokenizer_path = os.path.join(model_dir, "tokenizer")
    model_path = os.path.join(model_dir, "model")
    
    if os.path.exists(tokenizer_path) and os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        tokenizer.padding_side = "left"  # Add this line
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer.save_pretrained(tokenizer_path)
        model.save_pretrained(model_path)
    
    if tokenizer.eos_token is None or tokenizer.eos_token != "<eos>":
        tokenizer.add_special_tokens({'eos_token': "<eos>"})
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({'pad_token': "<pad>"})
        model.resize_token_embeddings(len(tokenizer))

    special_tokens = ["<eos>"]
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def bleurt_metric(predictions, references):
    metric = evaluate.load("bleurt", "bleurt-base-128")
    return metric.compute(predictions=predictions, references=references)

def bert_metric(predictions, references):
    metric = evaluate.load("bertscore", "microsoft/deberta-v3-small")
    return metric.compute(predictions=predictions, references=references)

def get_device_capacity(device):
    if device.startswith("cuda") and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.device(device))
        cores_per_sm = 128  # typical for recent NVIDIA GPUs
        gpu_cores = props.multi_processor_count * cores_per_sm
        gpu_freq = props.clock_rate * 1e3  # Hz
        return gpu_cores * gpu_freq
    else:
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq().max
        return cpu_count * cpu_freq  # Hz

def get_partition_config(device):
    return {"dataset_name": "hcm", "partition_size": 128} if not device.startswith("cpu") else {"dataset_name": "hcm", "partition_size": 200}

class TokenWeightedFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        weights_results = []
        total_tokens = 0
        for _, fit_res in results:
            num_tokens = fit_res.metrics.get("num_tokens", 0)
            total_tokens += num_tokens
            weights_results.append((fit_res.parameters, num_tokens))

        if total_tokens == 0:
            return super().aggregate_fit(rnd, results, failures)

        aggregated = None
        for params, num_tokens in weights_results:
            weight = num_tokens / total_tokens
            params_ndarrays = fl.common.parameters_to_ndarrays(params)
            if aggregated is None:
                aggregated = [weight * p for p in params_ndarrays]
            else:
                for i, p in enumerate(params_ndarrays):
                    aggregated[i] += weight * p

        aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated)
        return aggregated_parameters, {}
