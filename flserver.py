import os
import flwr as fl
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import numpy as np
from flwr.common import parameters_to_ndarrays

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        save_dir="./aggregated_models",
        save_frequency=1,
        save_final=True,
        num_rounds=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.save_dir = save_dir
        self.save_frequency = save_frequency
        self.save_final = save_final
        self.num_rounds = num_rounds
        os.makedirs(save_dir, exist_ok=True)
        self.init_model()
    
    def init_model(self):
        model_name = "HuggingFaceTB/SmolLM2-135M"
        
        print("Initializing model for parameter aggregation...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.base_model, lora_config)
        print("Model initialized successfully")

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            parameters, _ = aggregated_parameters
            should_save = self.save_final and server_round == self.num_rounds
            
            if should_save:
                self.apply_parameters_to_model(parameters)
                round_dir = os.path.join(self.save_dir, f"round_{server_round}")
                os.makedirs(round_dir, exist_ok=True)

                adapters_path = os.path.join(round_dir, "lora_adapters")
                self.model.save_pretrained(adapters_path)

                tokenizer_path = os.path.join(round_dir, "tokenizer")
                self.tokenizer.save_pretrained(tokenizer_path)

                print(f"Saved aggregated model for round {server_round} to {round_dir}")

        return aggregated_parameters
    
    def apply_parameters_to_model(self, param_tensors):
        param_tensors_list = parameters_to_ndarrays(param_tensors)
        trainable_param_names = [
            k for k, v in self.model.named_parameters() if v.requires_grad
        ]
        trainable_param_names = sorted(trainable_param_names)

        with torch.no_grad():
            for param_name, param_data in zip(trainable_param_names, param_tensors_list):
                module_path = param_name.split('.')
                curr_module = self.model
                for path in module_path[:-1]:
                    curr_module = getattr(curr_module, path)
                param = getattr(curr_module, module_path[-1])

                tensor_data = torch.tensor(
                    param_data,
                    dtype=param.dtype,
                    device=param.device
                )
                param.copy_(tensor_data)

def main():
    num_rounds = 3
    min_clients = 2
    server_config = fl.server.ServerConfig(num_rounds=num_rounds)

    strategy = SaveModelStrategy(
        save_dir="./aggregated_models",
        save_frequency=1,
        save_final=True,
        min_available_clients=min_clients,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        num_rounds=num_rounds
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=server_config
    )

if __name__ == "__main__":
    main()