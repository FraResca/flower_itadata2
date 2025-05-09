import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import flwr as fl
from flutils import *
from medpix2_dataset_preparation import medpix2_2050, medpix2_671

def clean_response(text):
    return "\n".join([line for line in text.splitlines() if not line.strip().lower().startswith(("instruction:", "answer:"))]).strip()

class LLMFlowerClient(fl.client.NumPyClient):
    def __init__(self):
        print("Starting federated fine-tuning client...\n")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}\n")
        self.model, self.tokenizer = get_model_tokenizer("smol")
        
        lora_config = LoraConfig(
            r=8, lora_alpha=32, target_modules=["q_proj", "v_proj", "o_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1, bias="none", task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        print("Model and tokenizer loaded successfully.\n")

        self.metric = get_metric("bleurt", "bleurt-large-512")
        _, val_cache_path = create_partitioned_datasets(tokenizer=self.tokenizer, partition_config=get_partition_config(self.device))
        self.val_dataset = load_dataset("json", data_files=val_cache_path)["train"]
        
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        
        self.training_args = TrainingArguments(
            output_dir="./smollm-finetuned",
            evaluation_strategy="epoch",
            learning_rate=5e-5,
            num_train_epochs=1,
            weight_decay=0.01,
            logging_steps=50,
            save_strategy="epoch", 
            fp16=False,
            greater_is_better=False,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            # metric_for_best_model="bleurt",
            dataloader_num_workers=0
        )

    def get_parameters(self, config=None):
        return [param.cpu().detach().numpy() for param in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=False)

    def move_model_to_cpu(self):
        self.model.to("cpu")
        self.device = "cpu"

    def move_model_to_gpu(self):
        if torch.cuda.is_available():
            self.model.to("cuda")
            self.device = "cuda"
        else:
            self.device = "cpu"

    @staticmethod
    def extract_prompt_and_response(text):
        marker = "### Response:\n"
        if marker in text:
            prompt, response = text.split(marker, 1)
            return prompt.strip(), response.strip()
        return "", text.strip()

    def compute_bleurt(self, preds, labels):
        extracted_preds, extracted_labels = [], []
        for pred, label in zip(preds, labels):
            _, pred_output = self.extract_prompt_and_response(pred)
            _, gold_output = self.extract_prompt_and_response(label)
            extracted_preds.append(clean_response(pred_output))
            extracted_labels.append(clean_response(gold_output))
        result = self.metric.compute(predictions=extracted_preds, references=extracted_labels)
        bleurt_scores = result.get("scores", [0.0])
        avg_bleurt = np.mean(bleurt_scores)
        return avg_bleurt

    def fit(self, parameters, config):
        # self.move_model_to_gpu()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.set_parameters(parameters)
        train_dataset, partition_id = load_random_partition()
        vocab_size = len(self.tokenizer)
        for example in train_dataset:
            for key in ["input_ids", "labels"]:
                if key in example:
                    example[key] = [self.tokenizer.pad_token_id if (x != -100 and (x < 0 or x >= vocab_size)) else x for x in example[key]]
        eval_len = min(8, len(self.val_dataset))

        trainer = CudaClearingTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=self.val_dataset.shuffle().select(range(eval_len)),
            data_collator=self.data_collator
        )
        trainer.train()
        
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        updated_parameters = self.get_parameters(config)
        num_examples = len(train_dataset)
        num_tokens = sum(len(example["input_ids"]) for example in train_dataset)
        
        # self.move_model_to_cpu()
        print("Arriva qui")
        return updated_parameters, num_examples, {"partition": partition_id, "num_tokens": num_tokens}

    '''
    def evaluate(self, parameters, config):
        self.move_model_to_gpu()
        self.set_parameters(parameters)
        batch_size = config.get("eval_batch_size", 8)
        eval_dataset = self.val_dataset.shuffle().select(range(min(100, len(self.val_dataset))))
        dataset_len = len(eval_dataset)
        num_batches = (dataset_len + batch_size - 1) // batch_size
        total_bleurt = 0.0
        total_examples = 0
        for i in range(num_batches):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, dataset_len)
            batch = eval_dataset.select(range(start_idx, end_idx))
            input_ids = torch.tensor([ex["input_ids"] for ex in batch]).to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=32,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            decoded_preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            label_ids = [ex["labels"] for ex in batch]
            label_ids_np = np.array(label_ids)
            label_ids_np[label_ids_np == -100] = self.tokenizer.pad_token_id
            decoded_labels = self.tokenizer.batch_decode(label_ids_np, skip_special_tokens=True)
            batch_bleurt = self.compute_bleurt(decoded_preds, decoded_labels)
            total_bleurt += batch_bleurt * (end_idx - start_idx)
            total_examples += (end_idx - start_idx)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        avg_bleurt = total_bleurt / total_examples if total_examples > 0 else 0.0
        self.move_model_to_cpu()
        return float(avg_bleurt), total_examples, {"bleurt": float(avg_bleurt)}
    '''

def main():
    client = LLMFlowerClient()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())

if __name__ == "__main__":
    main()
