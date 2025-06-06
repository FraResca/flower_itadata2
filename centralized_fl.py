import torch
import os
import time
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate
from bert_score import score as bert_score_fn
from dataset_manager import load_processed_dataset, preprocess_all, save_all_train_test, create_balanced_test_set
from newflutils import empty_gpu_cache
import gc
import sys

def get_config_param(param_name, default_value):
    config_path = f"config_files/centralized{sys.argv[1]}.json"
    if not os.path.exists(config_path):
        return default_value
    with open(config_path, "r") as f:
        data = json.load(f)
        return data.get(param_name, default_value)

def main():
    start_time = time.time()
    dataset_folder_name = "datasets"
    if not os.path.exists(dataset_folder_name):
        os.makedirs(dataset_folder_name)
        preprocess_all()
        save_all_train_test(get_config_param("seed", 42))
        create_balanced_test_set()

    model_name = get_config_param("modelname", "HuggingFaceTB/SmolLM2-135M")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and LoRA
    model = AutoModelForCausalLM.from_pretrained(model_name)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "o_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

    # Load and shuffle train set
    '''
    pub211_data = load_processed_dataset(f"{dataset_folder_name}/pubmed_qa_211k_train_set.jsonl")
    pub211_data = Dataset.from_list(pub211_data).shuffle().select(range(32768))

    med34_data = load_processed_dataset(f"{dataset_folder_name}/medical_meadow_medical_flashcards_34k_train_set.jsonl")
    med34_data = Dataset.from_list(med34_data).shuffle().select(range(4096))

    smalls_data = load_processed_dataset(f"{dataset_folder_name}/small_sets_united_train_set.jsonl")
    smalls_data = Dataset.from_list(smalls_data).shuffle().select(range(4096))

    train_data = concatenate_datasets([pub211_data, med34_data, smalls_data])
    '''

    short_data = load_processed_dataset(f"{dataset_folder_name}/short_train_set.jsonl")
    short_data = Dataset.from_list(short_data).shuffle(get_config_param("seed", 42)).select(range(16384))

    long_data = load_processed_dataset(f"{dataset_folder_name}/long_train_set.jsonl")
    long_data = Dataset.from_list(long_data).shuffle(get_config_param("seed", 42)).select(range(16384))

    train_data = concatenate_datasets([short_data, long_data])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    model.train()
    batch_size = get_config_param("train_batch_size", 4)
    epochs = get_config_param("epochs", 5)
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

    train_start = time.time()
    epoch_metrics = []
    epoch_train_times = []
    epoch_eval_times = []

    for epoch in range(epochs):
        # --- Track training time ---
        epoch_train_start = time.time()
        for batch in tqdm(dataloader, desc=f"Centralized Training Epoch {epoch+1}", unit="batch"):
            prompts = [s["prompt"] for s in batch]
            answers = [s["answer"] for s in batch]
            sequences = [p + a for p, a in zip(prompts, answers)]
            encodings = tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)
            labels = input_ids.clone()
            for i, (prompt, _) in enumerate(zip(prompts, answers)):
                prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True)["input_ids"][0]
                prompt_len = prompt_ids.size(0)
                labels[i, :prompt_len] = -100
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            empty_gpu_cache()
        gc.collect()
        epoch_train_end = time.time()
        epoch_train_time = epoch_train_end - epoch_train_start
        epoch_train_times.append(epoch_train_time)

        # --- Evaluation at end of each epoch ---
        eval_start = time.time()
        model.eval()
        val_data = load_processed_dataset(f"{dataset_folder_name}/balanced_test_set.jsonl")
        num_eval_examples = get_config_param("eval_examples", len(val_data))
        print(f"Evaluating on {num_eval_examples} examples.")
        val_data = Dataset.from_list(val_data).shuffle(get_config_param("seed", 42)).select(range(num_eval_examples))
        val_dataloader = DataLoader(val_data, batch_size=get_config_param("eval_batch_size", 2), collate_fn=lambda x: x)

        references = []
        candidates = []
        output_file = f"centralized{sys.argv[1]}_epoch{epoch+1}_eval_outputs.jsonl"
        with open(output_file, "w") as jsonfile:
            for batch in tqdm(val_dataloader, desc=f"Centralized Evaluation (Epoch {epoch+1})", unit="batch"):
                prompts = [sample["prompt"] for sample in batch]
                answers = [sample["answer"] for sample in batch]
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
                input_ids = inputs["input_ids"]
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=inputs.get("attention_mask"),
                        max_new_tokens=64,
                        do_sample=False,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                for i in range(len(batch)):
                    generated_ids = outputs[i][input_ids.shape[-1]:]
                    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    references.append(answers[i])
                    candidates.append(answer)
                    json.dump({
                        "prompt": prompts[i],
                        "prediction": answer,
                        "reference": answers[i]
                    }, jsonfile, indent=2)
                    jsonfile.write("\n")
                empty_gpu_cache()

        eval_end = time.time()
        eval_time = eval_end - eval_start
        epoch_eval_times.append(eval_time)

        # Ensure candidates and references are valid
        if len(candidates) == 0 or len(references) == 0:
            print(f"Warning: No predictions or references generated for epoch {epoch+1}. Skipping metric computation.")
            epoch_metrics.append({
                "epoch": epoch + 1,
                "rougeL": None,
                "bert": None
            })
        else:
            metric = evaluate.load("rouge")
            results = metric.compute(predictions=candidates, references=references)
            avg_rouge = results["rougeL"]

            P, R, F1 = bert_score_fn(candidates, references, lang="en", model_type="bert-base-uncased")
            avg_bert = F1.mean().item()

            epoch_metrics.append({
                "epoch": epoch + 1,
                "rougeL": avg_rouge,
                "bert": avg_bert
            })

        model.train()

    # --- Save all epoch metrics and times at the end ---
    metrics_save_path = f"centralized{sys.argv[1]}_epoch_metrics.jsonl"
    with open(metrics_save_path, "w") as metrics_file:
        json.dump(epoch_metrics, metrics_file, indent=2)

    train_times_save_path = f"centralized{sys.argv[1]}_epoch_train_times.json"
    with open(train_times_save_path, "w") as f:
        json.dump(epoch_train_times, f, indent=2)

    eval_times_save_path = f"centralized{sys.argv[1]}_epoch_eval_times.json"
    with open(eval_times_save_path, "w") as f:
        json.dump(epoch_eval_times, f, indent=2)

    train_end = time.time()
    train_time = train_end - train_start
    train_time_file = f"centralized{sys.argv[1]}_training_time.txt"
    with open(train_time_file, "w") as f:
        f.write(f"{train_time:.4f}\n")

    # Evaluation
    eval_start = time.time()

    model.eval()
    val_data = load_processed_dataset(f"{dataset_folder_name}/balanced_test_set.jsonl")
    num_eval_examples = get_config_param("eval_examples", len(val_data))
    val_data = Dataset.from_list(val_data).shuffle(get_config_param("seed", 42)).select(range(num_eval_examples))
    val_dataloader = DataLoader(val_data, batch_size=get_config_param("eval_batch_size", 2), collate_fn=lambda x: x)

    references = []
    candidates = []
    output_file = f"centralized{sys.argv[1]}_eval_outputs.jsonl"
    with open(output_file, "w") as jsonfile:
        for batch in tqdm(val_dataloader, desc="Centralized Evaluation", unit="batch"):
            prompts = [sample["prompt"] for sample in batch]
            answers = [sample["answer"] for sample in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            input_ids = inputs["input_ids"]
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=64,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
            for i in range(len(batch)):
                generated_ids = outputs[i][input_ids.shape[-1]:]
                answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                references.append(answers[i])
                candidates.append(answer)
                json.dump({
                    "prompt": prompts[i],
                    "prediction": answer,
                    "reference": answers[i]
                }, jsonfile, indent=2)
                jsonfile.write("\n")
            empty_gpu_cache()

    eval_end = time.time()
    eval_time = eval_end - eval_start

    # Save evaluation time
    eval_time_file = f"centralized{sys.argv[1]}_evaluation_time.txt"
    with open(eval_time_file, "w") as f:
        f.write(f"{eval_time:.4f}\n")

    # Add this check before metric computation
    if len(candidates) == 0 or len(references) == 0:
        print("Warning: No predictions or references generated in final evaluation. Skipping metric computation.")
        avg_rouge = None
        avg_bert = None
    else:
        metric = evaluate.load("rouge")
        results = metric.compute(predictions=candidates, references=references)
        avg_rouge = results["rougeL"]

        P, R, F1 = bert_score_fn(candidates, references, lang="en", model_type="bert-base-uncased")
        avg_bert = F1.mean().item()

    model_save_path = f"centralized{sys.argv[1]}_model.pt"
    torch.save(model.state_dict(), model_save_path)

    metrics_save_path = f"centralized{sys.argv[1]}_metrics.jsonl"
    with open(metrics_save_path, "w") as metrics_file:
        json.dump({
            "rougeL": avg_rouge,
            "bert": avg_bert
        }, metrics_file, indent=2)

    print(f"Centralized training and evaluation complete in {time.time() - start_time:.2f} seconds.")
    print(f"ROUGE-L: {avg_rouge:.4f} | BERTScore F1: {avg_bert:.4f}")

    del model
    del tokenizer
    del train_data
    del val_data
    gc.collect()
    empty_gpu_cache()

if __name__ == "__main__":
    main()