import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from peft import LoraConfig, get_peft_model

class FileLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            with open("log.txt", "a") as log_file:
                log_file.write(f"Step {state.global_step} logs: {logs}\n")

def preprocess_function(example):
    if example["input"].strip():
        prompt = f"### Instruction:\n{example['instruction']}\n### Input:\n{example['input']}\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{example['instruction']}\n### Response:\n"
    full_text = prompt + example["output"]
    return {"text": full_text}

def tokenize_function(examples, tokenizer, max_length=512):
    return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")

def main():
    with open("log.txt", "w") as log_file:
        log_file.write("Starting the fine-tuning process...\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open("log.txt", "a") as log_file:
        log_file.write(f"Using device: {device}\n")

    raw_dataset = load_dataset("tatsu-lab/alpaca")
    split_dataset = raw_dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    train_dataset = train_dataset.map(preprocess_function)
    val_dataset = val_dataset.map(preprocess_function)

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    with open("log.txt", "a") as log_file:
        log_file.write("Model and tokenizer loaded successfully.\n")

    train_dataset = train_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    '''
    # Set up training arguments.
    training_args = TrainingArguments(
        output_dir="./tinyllama-alpaca-finetuned",
        evaluation_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,  # Enable mixed-precision if supported.
    )
    '''

    training_args = TrainingArguments(
        output_dir="./tinyllama-alpaca-finetuned",
        evaluation_strategy="no",              
        learning_rate=1e-4,                    
        per_device_train_batch_size=1,         
        num_train_epochs=1,                    
        weight_decay=0.0,                      
        logging_steps=5,                       
        save_strategy="no",                    
        fp16=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[FileLogCallback()]
    )

    trainer.train()
    eval_results = trainer.evaluate()

    with open("log.txt", "a") as log_file:
        log_file.write(f"Evaluation results: {eval_results}\n")

    model.save_pretrained("./tinyllama-alpaca-finetuned")
    tokenizer.save_pretrained("./tinyllama-alpaca-finetuned")

    with open("log.txt", "a") as log_file:
        log_file.write("Fine-tuning completed and model saved.\n")

if __name__ == "__main__":
    main()
