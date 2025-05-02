import torch
from peft import LoraConfig, get_peft_model
import flwr as fl
from flutils import *
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import numpy as np

# export CUDA_VISIBLE_DEVICES = 1

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
            per_device_eval_batch_size=1,
            fp16=False,
            report_to="none"
        )

    def set_parameters(self, parameters):
        trainable_param_names = [k for k, v in self.model.named_parameters() if v.requires_grad]
        trainable_param_names = sorted(trainable_param_names)
        with torch.no_grad():
            for param_name, param_data in zip(trainable_param_names, parameters):
                module_path = param_name.split('.')
                curr_module = self.model
                for path in module_path[:-1]:
                    curr_module = getattr(curr_module, path)
                param = getattr(curr_module, module_path[-1])
                param.copy_(torch.tensor(param_data, device=self.device))

    def evaluate_fn(self, server_round, parameters, config):
        self.set_parameters(parameters)

        batch_size = config.get("eval_batch_size", 5)
        
        # RIGA DA TOGLIERE QUANDO FACCIAMO I SERI
        self.val_dataset = self.val_dataset.shuffle(seed=server_round).select(range(5))
        
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
                data_collator=self.data_collator
            )

            _ = temp_trainer.evaluate()

            prediction_output = temp_trainer.predict(batch_dataset)
            predictions, label_ids = prediction_output.predictions, prediction_output.label_ids

            predictions_tensor = torch.tensor(predictions).to(self.device)
            predictions_ids = torch.argmax(predictions_tensor, dim=-1)

            decoded_preds = self.tokenizer.batch_decode(predictions_ids.cpu().numpy(), skip_special_tokens=True)

            label_ids_np = np.array(label_ids)
            label_ids_np[label_ids_np == -100] = self.tokenizer.pad_token_id
            decoded_labels = self.tokenizer.batch_decode(label_ids_np, skip_special_tokens=True)

            bleurt_result = bleurt_metric(decoded_preds, decoded_labels)
            batch_scores = bleurt_result.get("scores", [])
            batch_bleurt = float(np.mean(batch_scores)) if batch_scores else 0.0

            total_bleurt += batch_bleurt * current_batch_size
            total_examples += current_batch_size

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            

        avg_bleurt = total_bleurt / total_examples if total_examples else 0.0
        return avg_bleurt, {"bleurt": avg_bleurt}

def main():
    num_rounds = 5
    min_clients = 1

    server_config = fl.server.ServerConfig(num_rounds=num_rounds)
    modelname = "smol"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}\n")

    model, tokenizer = get_model_tokenizer(modelname)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "o_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model = model.to(device)

    _, val_cache_path = create_partitioned_datasets(
        tokenizer=tokenizer,
        partition_config=get_partition_config(device),
    )

    val_dataset = load_dataset("json", data_files=val_cache_path)["train"]

    evaluator = ServerEvaluator(model, tokenizer, val_dataset, device)

    '''
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=min_clients,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        evaluate_fn=evaluator.evaluate_fn,
    )
    '''
    strategy = TokenWeightedFedAvg(
        min_available_clients=min_clients,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        evaluate_fn=evaluator.evaluate_fn,
    )
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=server_config
    )

if __name__ == "__main__":
    main()