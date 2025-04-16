import torch
from peft import LoraConfig, get_peft_model
import flwr as fl
from flutils import get_model_tokenizer, create_partitioned_datasets
from datasets import load_dataset
from flwr.common import parameters_to_ndarrays
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import numpy as np
import evaluate

class ServerEvaluator:
    def __init__(self, model, tokenizer, val_dataset, device):
        self.model = model
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.device = device
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        # self.bleurt = evaluate.load("bleurt", "bleurt-large-512")  # or "bleurt-base-128"
        self.bleurt = evaluate.load("bertscore")  # Specify the language or model_type

        self.training_args = TrainingArguments(
            output_dir="./server_evaluation",
            per_device_eval_batch_size=1,
            fp16=False,
            report_to="none"
        )

    def set_parameters(self, parameters):
        # Convert parameters to torch and load into model
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
        # Typical Flower signature: evaluate_fn(server_round, parameters, config), but we only need parameters, config
        self.set_parameters(parameters)

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            eval_dataset=self.val_dataset,
            data_collator=self.data_collator
        )

        result = trainer.evaluate()
        
        prediction_output = trainer.predict(self.val_dataset)
        predictions, label_ids = prediction_output.predictions, prediction_output.label_ids

        # Ensure on GPU for torch ops, then move to CPU for decoding
        predictions_tensor = torch.tensor(predictions).to(self.device)
        predictions_ids = torch.argmax(predictions_tensor, dim=-1)

        # Decoding expects CPU
        decoded_preds = self.tokenizer.batch_decode(predictions_ids.cpu().numpy(), skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(torch.tensor(label_ids).cpu().numpy(), skip_special_tokens=True)

        bleurt_result = self.bleurt.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
        avg_bleurt = float(np.mean(bleurt_result["scores"])) if "scores" in bleurt_result else 0.0


        return avg_bleurt, {"bleurt": avg_bleurt}

def main():
    num_rounds = 3
    min_clients = 1
    partition_size = 5

    server_config = fl.server.ServerConfig(num_rounds=num_rounds)
    modelname = "smol"
    datasetname = "medpix2"
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
        dataset_name=datasetname,
        partition_size=300
    )

    val_dataset = load_dataset("json", data_files=val_cache_path)["train"]

    evaluator = ServerEvaluator(model, tokenizer, val_dataset, device)

    strategy = fl.server.strategy.FedAvg(
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