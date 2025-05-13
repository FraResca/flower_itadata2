import flwr as fl
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
import json
from newflutils import empty_gpu_cache
from flwr.server import ServerConfig
from datasets import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from dataset_manager import load_processed_dataset, preprocess_all, save_all_train_test, create_balanced_test_set
from bert_score import score as bert_score_fn

def get_sever_config_param(param_name, default_value):
    with open("server_config.json", "r") as f:
        data = json.load(f)
        param_value = data.get(param_name, default_value)
    return param_value

class TokenWeightedFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        weights_results = []
        total_tokens = 0
        for _, fit_res in results:
            num_tokens = fit_res.metrics.get("num_tokens", 0)
            total_tokens += num_tokens
            weights_results.append((fit_res.parameters, num_tokens))

        if total_tokens == 0:
            return super().aggregate_fit(rnd, results, failures)

        aggregated = None
        for params, num_tokens in weights_results:
            weight = num_tokens / total_tokens
            params_ndarrays = fl.common.parameters_to_ndarrays(params)
            if aggregated is None:
                aggregated = [weight * p for p in params_ndarrays]
            else:
                for i, p in enumerate(params_ndarrays):
                    aggregated[i] += weight * p

        aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated)
        return aggregated_parameters, {}

def evaluate_fn(server_round, parameters, config):
    empty_gpu_cache()
    if get_sever_config_param("round_0", False) == False and server_round == 0:
        return 0.0, {}

    seed = get_sever_config_param("seed", 42)

    dataset_folder_name = "datasets"

    model_name = get_sever_config_param("model_name", "HuggingFaceTB/SmolLM2-135M")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
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

    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v).to(device) for k, v in params_dict}

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    val_dataset = load_processed_dataset(f"{dataset_folder_name}/balanced_test_set.jsonl")
    val_dataset = Dataset.from_list(val_dataset).shuffle(seed=seed).select(range(get_sever_config_param("eval_examples", len(val_dataset))))
    references = []
    candidates = []

    output_file = f"eval_outputs_round{server_round}.jsonl"
    batch_size = get_sever_config_param("eval_batch_size", 2)
    dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=lambda x: x)
    
    with open(output_file, "a") as jsonfile:
        for batch in tqdm(dataloader, desc="Server Evaluation", unit="batch"):
            prompts = [sample["prompt"] for sample in batch]
            answers = [sample["answer"] for sample in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            input_ids = inputs["input_ids"]
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
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

    metric = evaluate.load("rouge")
    results = metric.compute(predictions=candidates, references=references)
    avg_rouge = results["rougeL"]

    P, R, F1 = bert_score_fn(candidates, references, lang="en", model_type="bert-base-uncased")
    avg_bert = F1.mean().item()

    model_save_path = f"server_model_round{server_round}.pt"
    torch.save(model.state_dict(), model_save_path)

    # Save metrics
    metrics_save_path = f"server_metrics.json"
    with open(metrics_save_path, "w") as metrics_file:
        json.dump({
            "round": server_round,
            "rougeL": avg_rouge,
            "bert": avg_bert
        }, metrics_file, indent=2)

    del model
    del tokenizer
    del val_dataset

    empty_gpu_cache()

    return avg_rouge, {"rougeL": avg_rouge, "bert": avg_bert}

def main():
    dataset_folder_name = "datasets"

    if not os.path.exists(dataset_folder_name):
        os.makedirs(dataset_folder_name)
        preprocess_all()
        save_all_train_test()
        create_balanced_test_set()

    min_clients = 1
    server_config = ServerConfig(
        num_rounds=25,
    )

    strategy = TokenWeightedFedAvg(
        min_available_clients=min_clients,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        evaluate_fn=evaluate_fn
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=server_config
    )

if __name__ == "__main__":
    main()