import torch
import flwr as fl
import numpy as np
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from flutils import *
import evaluate

def extract_prompt_and_response(text):
    marker = "### Response:\n"
    if marker in text:
        prompt, response = text.split(marker, 1)
        return prompt.strip(), response.strip()
    return "", text.strip()

class ServerEvaluator:
    def __init__(self, model, tokenizer, val_dataset, device):
        self.model = model
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.device = device
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        self.training_args = TrainingArguments(
            output_dir="./server_evaluation",
            per_device_eval_batch_size=8,
            fp16=torch.cuda.is_available(),
            report_to="none"
        )

    def set_parameters(self, parameters):
        param_names = sorted(k for k, v in self.model.named_parameters() if v.requires_grad)
        with torch.no_grad():
            for param_name, param_data in zip(param_names, parameters):
                param = self.model
                for part in param_name.split('.')[:-1]:
                    param = getattr(param, part)
                param_part = getattr(param, param_name.split('.')[-1])
                param_part.copy_(torch.tensor(param_data, device=self.device))

    def evaluate_fn(self, server_round, parameters, config):
        if server_round == 0:
            print("Skipping evaluation before first training round.")
            return None, {}

        self.set_parameters(parameters)
        
        # Prepare validation data
        self.val_dataset = self.val_dataset.shuffle(seed=server_round).select(range(min(100, len(self.val_dataset))))
        
        batch_size = config.get("eval_batch_size", 8)
        num_batches = len(self.val_dataset) // batch_size + (len(self.val_dataset) % batch_size != 0)
        total_bleurt, total_examples = 0.0, 0

        # Evaluation loop
        for i in range(num_batches):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            start_idx, end_idx = i * batch_size, min((i + 1) * batch_size, len(self.val_dataset))
            batch_dataset = self.val_dataset.select(range(start_idx, end_idx))

            # Token ID validation
            vocab_size = len(self.tokenizer)
            for example in batch_dataset:
                for key in ["input_ids", "labels"]:
                    if key in example:
                        example[key] = [self.tokenizer.pad_token_id if (x != -100 and (x < 0 or x >= vocab_size)) else x for x in example[key]]

            # Temporary trainer for evaluation
            temp_trainer = Trainer(
                model=self.model,
                args=self.training_args,
                eval_dataset=batch_dataset,
                data_collator=self.data_collator
            )

            # Evaluate the batch
            prediction_output = temp_trainer.predict(batch_dataset)
            predictions, label_ids = prediction_output.predictions, prediction_output.label_ids

            # Decode predictions and labels
            predictions_ids = torch.argmax(torch.tensor(predictions).to(self.device), dim=-1)
            decoded_preds = self.tokenizer.batch_decode(predictions_ids.cpu().numpy(), skip_special_tokens=True)
            
            label_ids_np = np.array(label_ids)
            label_ids_np[label_ids_np == -100] = self.tokenizer.pad_token_id
            decoded_labels = self.tokenizer.batch_decode(label_ids_np, skip_special_tokens=True)

            # Extract responses and calculate BLEURT
            extracted_preds, extracted_labels = [], []
            for pred, label in zip(decoded_preds, decoded_labels):
                prompt, gold_output = extract_prompt_and_response(label)
                _, pred_output = extract_prompt_and_response(pred)
                extracted_preds.append(pred_output)
                extracted_labels.append(gold_output)

            bleurt_result = bleurt_metric(predictions=extracted_preds, references=extracted_labels)
            batch_bleurt = np.mean(bleurt_result.get("scores", [0.0])) if bleurt_result else 0.0

            total_bleurt += batch_bleurt * (end_idx - start_idx)
            total_examples += (end_idx - start_idx)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_bleurt = total_bleurt / total_examples if total_examples else 0.0
        return avg_bleurt, {"bleurt": avg_bleurt}

def main():
    num_rounds = 10
    min_clients = 1

    # Server configuration
    server_config = fl.server.ServerConfig(num_rounds=num_rounds)

    # Initialize model and tokenizer
    model_name = "smol"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    model, tokenizer = get_model_tokenizer(model_name)
    lora_config = LoraConfig(
        r=8, lora_alpha=32, target_modules=["q_proj", "v_proj", "o_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config).to(device)

    # Load validation dataset
    _, val_cache_path = create_partitioned_datasets(
        tokenizer=tokenizer,
        partition_config=get_partition_config(device)
    )
    val_dataset = load_dataset("json", data_files=val_cache_path)["train"]

    # Set up evaluator and strategy
    evaluator = ServerEvaluator(model, tokenizer, val_dataset, device)
    strategy = TokenWeightedFedAvg(
        min_available_clients=min_clients,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        evaluate_fn=evaluator.evaluate_fn
    )

    # Start federated learning server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=server_config
    )

if __name__ == "__main__":
    main()
