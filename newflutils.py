from datasets import load_dataset
from tqdm import tqdm
import json
import os
import random
import torch
import gc

def download_hcm_dataset():
    # Download the HCM dataset
    dataset = load_dataset("wangrongsheng/HealthCareMagic-100k-en")
    if "test" in dataset:
        train_data = dataset["train"].shuffle(seed=42)
        test_data = dataset["test"].shuffle(seed=42)
    else:
        # split the dataset into train and test sets
        train_data = dataset["train"].shuffle(seed=42).train_test_split(test_size=0.1)["train"]
        test_data = dataset["train"].shuffle(seed=42).train_test_split(test_size=0.1)["test"]

    return train_data, test_data

def create_hcm_dataset():
    # Check if the dataset is already downloaded
    if os.path.exists("hcm_dataset"):
        print("Dataset already exists. Skipping download.")
        return
    else:
        train_data, test_data = download_hcm_dataset()

        dataset_folder_name = "hcm_dataset"
        # Create a folder for the dataset if it doesn't exist
        if not os.path.exists(dataset_folder_name):
            os.makedirs(dataset_folder_name)

        # separate each line of both in prompt (instruction + input) and answer (output)
        # train_data = train_data.map(lambda x: {"prompt": x["instruction"] + " " + x["input"], "answer": x["output"]})
        # test_data = test_data.map(lambda x: {"prompt": x["instruction"] + " " + x["input"], "answer": x["output"]})

        train_data = train_data.map(lambda x: {"prompt": x["input"], "answer": x["output"]})
        test_data = test_data.map(lambda x: {"prompt": x["input"], "answer": x["output"]})

        with open(f"{dataset_folder_name}/test_data.jsonl", "w") as jsonfile:
            for sample in test_data:
                json.dump({
                    "prompt": sample["prompt"],
                    "answer": sample["answer"]
                }, jsonfile)
                jsonfile.write("\n")
        print("Test data saved to test_data.jsonl")

        partition_size = 4096
        num_partitions = (len(train_data) + partition_size - 1) // partition_size
        for i in tqdm(range(0, len(train_data), partition_size), desc="Saving train partitions", unit="partition", total=num_partitions):
            indices = list(range(i, min(i + partition_size, len(train_data))))
            partition = train_data.select(indices)
            partition_file = f"{dataset_folder_name}/train_data_{i // partition_size}.jsonl"
            with open(partition_file, "w") as jsonfile:
                for sample in partition:
                    json.dump({
                        "prompt": sample["prompt"],
                        "answer": sample["answer"]
                    }, jsonfile)
                    jsonfile.write("\n")
        print(f"Created {len(train_data) // partition_size} partitions of the training data.")

def load_train_partition():
    dataset_folder_name = "hcm_dataset"

    partition_files = [f for f in os.listdir(dataset_folder_name) if f.startswith("train_data_")]
    partition_file = os.path.join(dataset_folder_name, random.choice(partition_files))

    with open(partition_file, "r") as jsonfile:
        data = [json.loads(line) for line in jsonfile]
    return data

def load_test_data():
    dataset_folder_name = "hcm_dataset"

    with open(f"{dataset_folder_name}/test_data.jsonl", "r") as jsonfile:
        data = [json.loads(line) for line in jsonfile]
    return data

def empty_gpu_cache():
    """Empty the GPU cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    gc.collect()