import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
import flwr as fl

class FileLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            with open("log.txt", "a") as log_file:
                log_file.write(f"Step {state.global_step} logs: {logs}\n")

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

def load_alpaca(test_size=0.1, seed=42):
            raw_dataset = load_dataset("tatsu-lab/alpaca")
            split_dataset = raw_dataset["train"].train_test_split(test_size=test_size, seed=seed)
            train_dataset = split_dataset["train"].map(preprocess_function)
            val_dataset = split_dataset["test"].map(preprocess_function)
            return train_dataset, val_dataset

def tinyllama_model_tokenizer():
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        model_dir = "tinyllama_model"
        
        os.makedirs(model_dir, exist_ok=True)
        with open("log.txt", "a") as log_file:
            log_file.write(f"Using permanent model directory: {model_dir}\n")
        
        # Check if model already exists locally
        tokenizer_path = os.path.join(model_dir, "tokenizer")
        model_path = os.path.join(model_dir, "model")
        
        if os.path.exists(tokenizer_path) and os.path.exists(model_path):
            # Load from local directory
            with open("log.txt", "a") as log_file:
                log_file.write("Loading model and tokenizer from local directory\n")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        else:
            # Download from HuggingFace and save locally
            with open("log.txt", "a") as log_file:
                log_file.write("Downloading model and tokenizer from HuggingFace\n")
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
            
            # Save to permanent location
            tokenizer.save_pretrained(tokenizer_path)
            model.save_pretrained(model_path)
            with open("log.txt", "a") as log_file:
                log_file.write("Model and tokenizer saved to local directory\n")
                
        tokenizer.pad_token = tokenizer.eos_token

        return tokenizer, model

def get_tokenized_datasets(tokenizer):
    cached_dir = "cached_datasets"
    os.makedirs(cached_dir, exist_ok=True)
    train_cache_path = "tokenized_train"
    val_cache_path = "tokenized_val"
    
    # Check if cached datasets exist
    if os.path.exists(train_cache_path) and os.path.exists(val_cache_path):
        with open("log.txt", "a") as log_file:
            log_file.write("Loading tokenized datasets from cache...\n")
        train_dataset = load_dataset("json", data_files=train_cache_path)["train"]
        val_dataset = load_dataset("json", data_files=val_cache_path)["train"]
    else:
        with open("log.txt", "a") as log_file:
            log_file.write("Tokenizing datasets and saving to cache...\n")
        train_dataset, val_dataset = load_alpaca()
        train_dataset = train_dataset.map(
            lambda ex: tokenize_function(ex, tokenizer), batched=True
        )
        val_dataset = val_dataset.map(
            lambda ex: tokenize_function(ex, tokenizer), batched=True
        )
        
        # Save processed datasets to disk
        train_dataset.to_json(train_cache_path)
        val_dataset.to_json(val_cache_path)
        with open("log.txt", "a") as log_file:
            log_file.write(f"Datasets cached to {cached_dir}\n")

    return train_dataset, val_dataset

class LLMFlowerClient(fl.client.NumPyClient):
    def __init__(self):
        with open("log.txt", "w") as log_file:
            log_file.write("Starting federated fine-tuning client...\n")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        with open("log.txt", "a") as log_file:
            log_file.write(f"Using device: {self.device}\n")

        self.tokenizer, self.model = tinyllama_model_tokenizer()
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        with open("log.txt", "a") as log_file:
            log_file.write("Model and tokenizer loaded successfully.\n")

        self.train_dataset, self.val_dataset = get_tokenized_datasets(self.tokenizer)
        # downsample the val_dataset to 1 example
        self.val_dataset = self.val_dataset.select(range(1))

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        self.training_args = TrainingArguments(
            output_dir="./tinyllama-alpaca-finetuned",
            eval_strategy="no",
            learning_rate=1e-4,
            per_device_train_batch_size=1,
            num_train_epochs=0.0001,
            weight_decay=0.0,
            logging_steps=5,
            save_strategy="no",
            fp16=False,
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.data_collator,
            callbacks=[FileLogCallback()]
        )

        # ...existing code...
    def get_parameters(self, config=None):
        # Only get the trainable parameters (LoRA parameters)
        trainable_params = {
            k: v.cpu().detach().numpy()  # Add detach() before numpy()
            for k, v in self.model.named_parameters()
            if v.requires_grad
        }
        return [trainable_params[k] for k in sorted(trainable_params.keys())]

    def set_parameters(self, parameters):
        trainable_param_names = [
            k for k, v in self.model.named_parameters() if v.requires_grad
        ]
        trainable_param_names = sorted(trainable_param_names)
        
        with torch.no_grad():
            for param_name, param_data in zip(trainable_param_names, parameters):
                module_path = param_name.split('.')
                curr_module = self.model
                for path in module_path[:-1]:
                    curr_module = getattr(curr_module, path)
                param = getattr(curr_module, module_path[-1])
                param.copy_(torch.tensor(param_data).to(param.device))

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.trainer.train()
        updated_parameters = self.get_parameters()
        num_examples = len(self.train_dataset)
        return updated_parameters, num_examples, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        eval_result = self.trainer.evaluate()
        loss = eval_result.get("eval_loss", 0.0)
        num_examples = len(self.val_dataset)
        return float(loss), num_examples, {"loss": float(loss)}

def main():
    client = LLMFlowerClient()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()