import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
import flwr as fl
from flutils import *
from medpix2_dataset_preparation import medpix2_2050, medpix2_671

# export CUDA_VISIBLE_DEVICES = 0

def tokenize_function(examples, tokenizer, max_length=512):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )

class LLMFlowerClient(fl.client.NumPyClient):
    def __init__(self):
        print("Starting federated fine-tuning client...\n")

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}\n")

        self.model, self.tokenizer = get_model_tokenizer("smol")
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "o_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        print("Model and tokenizer loaded successfully.\n")

        self.metric = evaluate.load("bleurt", 'bleurt-large-512')

        _, val_cache_path = create_partitioned_datasets(
            tokenizer=self.tokenizer,
            partition_config=get_partition_config(self.device)
        )
        self.val_dataset = load_dataset("json", data_files=val_cache_path)["train"]  

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        self.training_args = TrainingArguments(
            output_dir="./tinyllama-alpaca-finetuned",
            eval_strategy="no",  # Disabilita valutazione durante il training
            learning_rate=2e-6,
            num_train_epochs=1,
            weight_decay=0.0,
            logging_steps=5,
            save_strategy="no",
            fp16=False,
            greater_is_better=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            metric_for_best_model="bleurt",
            dataloader_num_workers=0  # Riduce l'uso di memoria
        )
    


    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        predictions_ids = np.argmax(predictions, axis=-1)
        
        decoded_preds = self.tokenizer.batch_decode(predictions_ids, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        result = bleurt_metric(predictions=decoded_preds, references=decoded_labels)
        for i in range(len(decoded_preds)):
            print(f"\nDecoded prediction:\n{decoded_preds[i]}")
            print(f"\nDecoded label:\n{decoded_labels[i]}")

        print(f"BLEURT result keys: {list(result.keys())}")
        
        if 'scores' in result:
            bleurt_scores = result['scores']
            print(f"BLEURT scores: {bleurt_scores}")
        else:
            first_key = next(iter(result.keys())) if result else None
            bleurt_scores = result.get(first_key, [0.0])
            print(f"Using fallback key '{first_key}' for BLEURT scores")
        
        avg_bleurt = float(np.mean(bleurt_scores))
        
        return {"bleurt": avg_bleurt}
    
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
        batch_size = 10  # Imposta la dimensione del batch a 10 esempi
        
        # Modifica le training_args per usare batch piccoli e per evitare OOM
        self.training_args.per_device_train_batch_size = 1
        self.training_args.gradient_accumulation_steps = 4
        self.training_args.dataloader_num_workers = 0
        
        # Prepara il modello e ottimizzatore
        self.model.to(self.device)
        
        # Suddivide il dataset in batch più piccoli
        num_examples = len(train_dataset)
        num_batches = (num_examples + batch_size - 1) // batch_size
        
        # Counters
        num_tokens = 0
        for example in train_dataset:
            num_tokens += len(example["input_ids"])
        
        # Addestra su batch piccoli
        print(f"Training on {num_examples} examples in {num_batches} batches")
        
        for i in range(num_batches):
            # Libera memoria CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Crea sotto-dataset per questo batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_examples)
            current_batch_size = end_idx - start_idx
            
            batch_dataset = train_dataset.select(range(start_idx, end_idx))
            
            print(f"Training batch {i+1}/{num_batches} with {len(batch_dataset)} examples")
            
            # Crea un nuovo trainer per ogni batch
            temp_trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=batch_dataset,
                eval_dataset=None,  # Non valutare durante il training a batch
                data_collator=self.data_collator,
                compute_metrics=None  # Non calcolare metriche durante il training a batch
            )
            
            temp_trainer.train()
            
            # Libera memoria dopo ogni batch
            del temp_trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print(f"Batch {i+1}/{num_batches} completed successfully")

        
        # Ottiene i parametri aggiornati dopo aver completato tutti i batch
        updated_parameters = self.get_parameters()
        
        return updated_parameters, num_examples, {
            "partition": partition_id,
            "num_tokens": num_tokens
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        batch_size = config.get("eval_batch_size", 5)
        
        dataset_len = len(self.val_dataset)
        num_batches = (dataset_len + batch_size - 1) // batch_size
        
        total_bleurt = 0.0
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
                compute_metrics=self.compute_metrics
            )
            
            batch_result = temp_trainer.evaluate()
            batch_bleurt = batch_result.get("eval_bleurt", 0.0)
            
            total_bleurt += batch_bleurt * current_batch_size
            total_examples += current_batch_size
        
        avg_bleurt = total_bleurt / total_examples if total_examples > 0 else 0.0
        
        return float(avg_bleurt), total_examples, {
            "bleurt": float(avg_bleurt)
        }

def main():
    client = LLMFlowerClient()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())


if __name__ == "__main__":
    main()