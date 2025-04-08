import os
import torch
import numpy as np
import evaluate
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
import random

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
            # raw_dataset = load_dataset("tatsu-lab/alpaca")
            raw_dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")
            split_dataset = raw_dataset["train"].train_test_split(test_size=test_size, seed=seed)
            train_dataset = split_dataset["train"].map(preprocess_function)
            val_dataset = split_dataset["test"].map(preprocess_function)

            return train_dataset, val_dataset

def tinyllama_model_tokenizer():
        # model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        # model_dir = "tinyllama_model"
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

        return tokenizer, model

def create_partitioned_datasets(tokenizer, num_partitions=5, client_seed=None):
    cached_dir = "cached_datasets"
    os.makedirs(cached_dir, exist_ok=True)
    
    partitions_dir = os.path.join(cached_dir, "partitions")
    os.makedirs(partitions_dir, exist_ok=True)
    
    val_cache_path = os.path.join(cached_dir, "tokenized_val")
    
    partitions_created = True
    for i in range(num_partitions):
        partition_path = os.path.join(partitions_dir, f"partition_{i}")
        if not os.path.exists(partition_path):
            partitions_created = False
            break
    
    num_partitions_created = len(os.listdir(partitions_dir))
    if num_partitions_created != num_partitions:
        partitions_created = False

    if not partitions_created or not os.path.exists(val_cache_path):

        for filename in os.listdir(partitions_dir):
            file_path = os.path.join(partitions_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        print(f"Creating {num_partitions} dataset partitions...\n")
        train_dataset_full, val_dataset = load_alpaca()
        
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
        
        partition_size = total_size // num_partitions
        for i in range(num_partitions):
            start_idx = i * partition_size
            end_idx = start_idx + partition_size if i < num_partitions - 1 else total_size
            partition_indices = indices[start_idx:end_idx]
            
            partition = train_dataset_full.select(partition_indices)
            partition_path = os.path.join(partitions_dir, f"partition_{i}")
            partition.to_json(partition_path)
            
            print(f"Created partition {i} with {len(partition)} examples\n")
    
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

class LLMFlowerClient(fl.client.NumPyClient):
    def __init__(self):
        print("Starting federated fine-tuning client...\n")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}\n")

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
        print("Model and tokenizer loaded successfully.\n")

        self.metric = evaluate.load("rouge")

        _, val_cache_path = create_partitioned_datasets(self.tokenizer, num_partitions=8000)
        # sample only one example for validation
        self.val_dataset = load_dataset("json", data_files=val_cache_path)["train"]  
        self.val_dataset = self.val_dataset.select(range(1))

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        self.training_args = TrainingArguments(
            output_dir="./tinyllama-alpaca-finetuned",
            eval_strategy="epoch",
            learning_rate=1e-4,
            num_train_epochs=1,
            weight_decay=0.0,
            logging_steps=5,
            save_strategy="no",
            fp16=False,
            greater_is_better=True,
            metric_for_best_model="rouge1"
        )

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        
        # Get logits from predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Convert predictions from logits to token IDs
        predictions_ids = np.argmax(predictions, axis=-1)
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions_ids, skip_special_tokens=True)
        
        # For labels, we need to handle potential padding
        # Ignore -100 which is used as padding token ID in many cases
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute ROUGE scores
        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
        
        # Return metrics - directly use the result values without accessing .mid.fmeasure
        return {
            "rouge1": float(result["rouge1"]),
            "rougeL": float(result["rougeL"]),
        }

    def get_parameters(self, config=None):
        trainable_params = {
            k: v.cpu().detach().numpy()
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

        train_dataset, partition_id = load_random_partition()

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )

        self.trainer.train()
        updated_parameters = self.get_parameters()
        num_examples = len(train_dataset)
        return updated_parameters, num_examples, {"partition": partition_id}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        batch_size = config.get("eval_batch_size", 1)
        
        dataset_len = len(self.val_dataset)
        num_batches = (dataset_len + batch_size - 1) // batch_size
        
        total_rouge1 = 0.0
        total_rougeL = 0.0
        total_examples = 0
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, dataset_len)
            current_batch_size = end_idx - start_idx
            
            batch_dataset = self.val_dataset.select(range(start_idx, end_idx))
            
            temp_trainer = Trainer(
                model=self.model,
                args=self.training_args,
                eval_dataset=batch_dataset,
                data_collator=self.data_collator,
                compute_metrics=self.compute_metrics  # Add compute_metrics function
            )
            
            batch_result = temp_trainer.evaluate()
            batch_rouge1 = batch_result.get("eval_rouge1", 0.0)
            batch_rougeL = batch_result.get("eval_rougeL", 0.0)
            
            total_rouge1 += batch_rouge1 * current_batch_size
            total_rougeL += batch_rougeL * current_batch_size
            total_examples += current_batch_size
        
        # Calculate average metrics
        avg_rouge1 = total_rouge1 / total_examples if total_examples > 0 else 0.0
        avg_rougeL = total_rougeL / total_examples if total_examples > 0 else 0.0
        
        return float(avg_rouge1), total_examples, {
            "rouge1": float(avg_rouge1),
            "rougeL": float(avg_rougeL)
        }

def main():
    client = LLMFlowerClient()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())


if __name__ == "__main__":
    main()