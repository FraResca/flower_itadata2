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

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
            dataset_name="medpix2",
            partition_size=300
        )
        self.val_dataset = load_dataset("json", data_files=val_cache_path)["train"]  

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        self.training_args = TrainingArguments(
            output_dir="./tinyllama-alpaca-finetuned",
            eval_strategy="epoch",
            learning_rate=2e-6,
            num_train_epochs=1,
            weight_decay=0.0,
            logging_steps=5,
            save_strategy="no",
            fp16=False,
            greater_is_better=True,
            metric_for_best_model="bleurt"
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

        print(f"BLEURT result keys: {list(result.keys())}")
        
        if 'scores' in result:
            bleurt_scores = result['scores']
            print(f"BLEURT scores: {bleurt_scores}")
        else:
            first_key = next(iter(result.keys())) if result else None
            bleurt_scores = result.get(first_key, [0.0])
            print(f"Using fallback key '{first_key}' for BLEURT scores")
        
        avg_bleurt = float(np.mean(bleurt_scores)) if len(bleurt_scores) > 0 else 0.0
        
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