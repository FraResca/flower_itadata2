import flwr as fl
import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from newflutils import load_train_partition, create_hcm_dataset, empty_gpu_cache
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from dataset_manager import load_processed_dataset, preprocess_all, save_all_train_test
from datasets import Dataset

def get_client_config_param(param_name, default_value):
    with open("client_config.json", "r") as f:
        data = json.load(f)
        param_value = data.get(param_name, default_value)
    return param_value

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model_name="HuggingFaceTB/SmolLM2-135M"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "o_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set pad_token if not present
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=False)

    '''
    # con autocast
    def fit(self, parameters, config):
        empty_gpu_cache()

        self.set_parameters(parameters)
        partition_index = config.get("partition_index", 0)
        train_data = load_train_partition(partition_index)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.model.train()
        total_tokens = 0

        batch_size = 2
        dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
        for batch in tqdm(dataloader, desc="Training", unit="batch"):
            prompts = [s["prompt"] for s in batch]
            answers = [s["answer"] for s in batch]

            sequences = [p + a for p, a in zip(prompts, answers)]
            encodings = self.tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)
            labels = input_ids.clone()

            for i, (prompt, answer) in enumerate(zip(prompts, answers)):
                prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True)["input_ids"][0]
                prompt_len = prompt_ids.size(0)
                labels[i, :prompt_len] = -100

            with torch.amp.autocast("cuda"):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
            total_tokens += (labels != -100).sum().item()

            empty_gpu_cache()

        return self.get_parameters(), len(train_data), {"num_tokens": total_tokens}
    '''
    # '''
    # Senza autocast
    def fit(self, parameters, config):
        empty_gpu_cache()

        dataset_folder_name = "datasets"
        dataset_name = get_client_config_param("dataset_name", "chatdoctor_icliniq_7k")

        self.set_parameters(parameters)
        train_data = load_processed_dataset(f"{dataset_folder_name}/{dataset_name}_train_set.jsonl")
        num_examples = get_client_config_param("train_examples", 1024)
        train_data = Dataset.from_list(train_data)
        train_data = train_data.shuffle().select(range(num_examples))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.model.train()
        total_tokens = 0

        batch_size = get_client_config_param("train_batch_size", 2)
        dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
        for batch in tqdm(dataloader, desc="Training", unit="batch"):
            try:
                prompts = [s["prompt"] for s in batch]
                answers = [s["answer"] for s in batch]

                sequences = [p + a for p, a in zip(prompts, answers)]
                encodings = self.tokenizer(
                    sequences,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )

                input_ids = encodings["input_ids"].to(self.device)
                attention_mask = encodings["attention_mask"].to(self.device)
                labels = input_ids.clone()

                for i, (prompt, _) in enumerate(zip(prompts, answers)):
                    prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True)["input_ids"][0]
                    prompt_len = prompt_ids.size(0)
                    labels[i, :prompt_len] = -100

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_tokens += (labels != -100).sum().item()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("OOM detected, skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            empty_gpu_cache()
        
        return self.get_parameters(), len(train_data), {"num_tokens": total_tokens}    # '''
    '''
    def evaluate(self, parameters, config):
        empty_gpu_cache()

        self.set_parameters(parameters)
        val_dataset = load_test_data()
        val_dataset = Dataset.from_list(val_dataset).shuffle().select(range(50))
        # Remove Dataset.from_list(val_dataset) and use val_data directly
        self.model.eval()
        total_loss = 0.0
        batch_size = 2
        dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=lambda x: x)
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
                prompts = [s["prompt"] for s in batch]
                answers = [s["answer"] for s in batch]
                inputs = [p + a for p, a in zip(prompts, answers)]
                encodings = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
                input_ids = encodings["input_ids"].to(self.device)
                attention_mask = encodings["attention_mask"].to(self.device)
                labels = input_ids.clone()
                for i, (prompt, answer) in enumerate(zip(prompts, answers)):
                    prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True)["input_ids"][0]
                    prompt_len = prompt_ids.size(0)
                    labels[i, :prompt_len] = -100
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item() * len(batch)
        avg_loss = total_loss / len(val_dataset)

        empty_gpu_cache()
        return float(avg_loss), len(val_dataset), {}
    '''
        
if __name__ == "__main__":
    # if there is no dataset folder, create it
    dataset_folder_name = "datasets"
    if not os.path.exists(dataset_folder_name):
        os.makedirs(dataset_folder_name)
        preprocess_all()
        save_all_train_test()
    fl.client.start_client(
        server_address="0.0.0.0:8080",
        client=FlowerClient(model_name=get_client_config_param("modelname", "HuggingFaceTB/SmolLM2-135M")).to_client()
    )