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

def clean_response(text):
    # Remove unwanted headers from model output
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        if not line.strip().lower().startswith(("instruction:", "answer:")):
            cleaned.append(line)
    return "\n".join(cleaned).strip()

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

        # Real training parameters
        self.training_args = TrainingArguments(
            output_dir="./smollm-finetuned",
            evaluation_strategy="epoch",
            learning_rate=5e-5,              # Typical for LLM finetuning
            num_train_epochs=3,              # More epochs for real training
            weight_decay=0.01,
            logging_steps=50,
            save_strategy="epoch",           # Save at the end of each epoch
            fp16=torch.cuda.is_available(),  # Use fp16 if possible
            greater_is_better=True,
            per_device_train_batch_size=8,   # Increase batch size if possible
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,   # Accumulate gradients for larger effective batch
            metric_for_best_model="bleurt",
            dataloader_num_workers=2
        )
    
    @staticmethod
    def extract_prompt_and_response(text):
        marker = "### Response:\n"
        if marker in text:
            prompt, response = text.split(marker, 1)
            return prompt.strip(), response.strip()
        return "", text.strip()

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        predictions_ids = np.argmax(predictions, axis=-1)
        decoded_preds = self.tokenizer.batch_decode(predictions_ids, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        extracted_preds = []
        extracted_labels = []
        for i in range(len(decoded_preds)):
            prompt, gold_output = self.extract_prompt_and_response(decoded_labels[i])
            _, pred_output = self.extract_prompt_and_response(decoded_preds[i])
            pred_output = clean_response(pred_output)
            gold_output = clean_response(gold_output)
            extracted_preds.append(pred_output)
            extracted_labels.append(gold_output)
            print(f"\nInput (Prompt):\n{prompt}")
            print(f"Predicted Output:\n{pred_output}")
            print(f"Original Output:\n{gold_output}")

        result = bleurt_metric(predictions=extracted_preds, references=extracted_labels)
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.set_parameters(parameters)
        
        train_dataset, partition_id = load_random_partition()
        
        # Use real training args (already set in __init__)
        self.model.to(self.device)

        num_tokens = sum(len(example["input_ids"]) for example in train_dataset)
        num_examples = len(train_dataset)

        # Debug: Check for out-of-vocab token ids
        vocab_size = len(self.tokenizer)
        for example in train_dataset:
            for key in ["input_ids", "labels"]:
                if key in example:
                    ids = example[key]
                    for idx in ids:
                        if idx != -100 and (idx < 0 or idx >= vocab_size):
                            print(f"Invalid token id {idx} in {key} (vocab size {vocab_size})")

        print(f"Training on {num_examples} examples (single Trainer, automatic batching)")

        eval_len = min(32, len(self.val_dataset))
        trainer = CudaClearingTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=self.val_dataset.shuffle().select(range(eval_len)),
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )

        trainer.train()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        updated_parameters = self.get_parameters()

        return updated_parameters, num_examples, {
            "partition": partition_id,
            "num_tokens": num_tokens
        }
    '''
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        batch_size = config.get("eval_batch_size", 8)
        self.val_dataset = self.val_dataset.shuffle().select(range(100))

        dataset_len = len(self.val_dataset)
        num_batches = (dataset_len + batch_size - 1) // batch_size
        
        total_bleurt = 0.0
        total_examples = 0
        
        for i in range(num_batches):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_bleurt = total_bleurt / total_examples if total_examples > 0 else 0.0

        return float(avg_bleurt), total_examples, {
            "bleurt": float(avg_bleurt)
        }
    '''

def main():
    client = LLMFlowerClient()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())

if __name__ == "__main__":
    main()