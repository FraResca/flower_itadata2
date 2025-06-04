import os
import json
import random
from datasets import load_dataset
from tqdm import tqdm

###############################
########## VARIABLES ##########
###############################
DATASETS_PATH = "./datasets"

# Validation dataset bilanciato (seeddato)

# Validation dataset completamente random

# Lista di dataset disponibili, così si può creare un sottoinsieme

# Picker di soli elementi il cui output è in una lunghezza target (min e max ?)

###########################
########## UTILS ##########
###########################
def remove_newlines(string):
    return string.replace("\n", " ")

def clean_string(string):
    '''
        Turns all newlines \n to spaces
        Turns all double spaces to single spaces
        Converts all unicode with corresponding characters
        Converts all leftovers \\ in '
    '''
    string = string.replace("\\n", " ")
    string = string.replace("  ", " ")
    string = string.encode().decode("unicode_escape")
    string = string.replace("\\", "'")
    return string

def save_processed_dataset(dataset, save_path): 
    '''
        Saves preprocessed dataset as a JSONL file. Each line is a JSON object with keys "prompt" and "answer".
        dataset: dictionary with one key "train"
        dataset["train"]: dictionary with two keys "input" and "output"
        dataset["train"]["input"] and dataset["train"]["output"]: lists of strings
    '''
    target = len(dataset["train"]["input"])
    next_update = 1
    print(f"Saving: {save_path}")
    with open(save_path, "w", encoding="utf-8") as outfile, tqdm(total=100) as pbar:
        for i in range(target):
            example = {
                "prompt": dataset["train"]["input"][i],
                "answer": dataset["train"]["output"][i]
            }
            outfile.write(json.dumps(example, ensure_ascii=False))
            outfile.write("\n")

            # Print indicative progress
            progress = (i + 1) * 100 / target
            if progress >= next_update:
                pbar.update(1)
                next_update += 1
        


def load_processed_dataset(load_path):
    '''
        Loads preprocessed dataset saved as JSONL.
        Each line is a JSON object with keys "prompt" and "answer".
    '''
    with open(load_path, "r") as jsonfile:
        dataset = [json.loads(line) for line in jsonfile]

    return dataset

def print_stats():
    total_examples = 0
    for filename in os.listdir(DATASETS_PATH):
        with open(f"{DATASETS_PATH}/{filename}", "r", encoding="utf-8") as dataset_file:
            example_count = sum(1 for _ in dataset_file)
        print(f"{filename}: {example_count} examples")
        total_examples += example_count
    print(f"Total: {total_examples} examples")


################################
########## PREPROCESS ##########
################################
def preprocess_all():
    if not os.path.exists(DATASETS_PATH):
        os.makedirs(DATASETS_PATH)
    
    preprocess_chatdoctor_icliniq_7k()
    preprocess_medical_meadow_medical_flashcards_34k()
    preprocess_medical_meadow_wikidoc_10k()
    preprocess_medical_meadow_wikidoc_patient_information_6k()
    preprocess_pubmed_qa_211k()
    print_stats()

def preprocess_chatdoctor_icliniq_7k():

    # Load dataset
    raw_dataset = load_dataset("lavita/medical-qa-datasets", "chatdoctor-icliniq")

    # Initialize processed dataset structure
    chatdoctor_icliniq_7k = {}
    chatdoctor_icliniq_7k["train"] = {}
    chatdoctor_icliniq_7k["train"]["input"] = []
    chatdoctor_icliniq_7k["train"]["output"] = []

    # Custom processing of raw data
    for raw_example in raw_dataset["test"]:
        input = clean_string(raw_example["input"])
        chatdoctor_icliniq_7k["train"]["input"].append(input)
        
        output = clean_string(raw_example["answer_chatgpt"])
        chatdoctor_icliniq_7k["train"]["output"].append(output)

    # Save
    save_processed_dataset(chatdoctor_icliniq_7k, f"{DATASETS_PATH}/chatdoctor_icliniq_7k.jsonl")

# def preprocess_chatdoctor_healthcaremagic_112k():
#     # Load dataset
#     raw_dataset = load_dataset("lavita/medical-qa-datasets", "chatdoctor_healthcaremagic")
#     # Veramente di scarsa qualità, vediamo se si riesce a fare qualcosa con ciò che rimane

def preprocess_medical_meadow_medical_flashcards_34k():
    # Load dataset
    raw_dataset = load_dataset("lavita/medical-qa-datasets", "medical_meadow_medical_flashcards")

    # Initialize processed dataset structure
    medical_meadow_medical_flashcards_34k = {}
    medical_meadow_medical_flashcards_34k["train"] = {}
    medical_meadow_medical_flashcards_34k["train"]["input"] = []
    medical_meadow_medical_flashcards_34k["train"]["output"] = []

    # Custom processing of raw data
    # Dataset is of pretty good quality out of the box, there are some empty inputs and outputs that have to be removed
    for raw_example in raw_dataset["train"]:
        input = raw_example["input"]
        output = raw_example["output"]
        
        # Skip if either input or output are empty
        if len(input) == 0 or len(output) == 0:
            continue

        # If both input and output are valid, turn newlines to spaces so the string is all on one line
        input = remove_newlines(input)
        output = remove_newlines(output)
        
        medical_meadow_medical_flashcards_34k["train"]["input"].append(input)
        medical_meadow_medical_flashcards_34k["train"]["output"].append(output)

    # Save
    save_processed_dataset(medical_meadow_medical_flashcards_34k, f"{DATASETS_PATH}/medical_meadow_medical_flashcards_34k.jsonl")


def preprocess_medical_meadow_wikidoc_10k():
    # Load dataset
    raw_dataset = load_dataset("lavita/medical-qa-datasets", "medical_meadow_wikidoc")

    # Initialize processed dataset structure
    medical_meadow_wikidoc_10k = {}
    medical_meadow_wikidoc_10k["train"] = {}
    medical_meadow_wikidoc_10k["train"]["input"] = []
    medical_meadow_wikidoc_10k["train"]["output"] = []

    # Custom processing of raw data
    # Dataset is of pretty good quality out of the box, there are some empty inputs and outputs that have to be removed
    for raw_example in raw_dataset["train"]:
        input = raw_example["input"]
        output = raw_example["output"]
        
        # Skip if either input or output are empty
        if len(input) == 0 or len(output) == 0:
            continue

        # Skip also for output too short (from a qualitative human analysis, these are not significative, such as single words, names of medications, incomplete answers or URLs to other sources)
        # Some relevant examples might be lost too, but it's a negligible amount (less than 1%)
        if len(output) <= 66:
            continue

        # If both input and output are valid, turn newlines to spaces so the string is all on one line
        input = remove_newlines(input)
        output = remove_newlines(output)

        medical_meadow_wikidoc_10k["train"]["input"].append(input)
        medical_meadow_wikidoc_10k["train"]["output"].append(output)

    # Save
    save_processed_dataset(medical_meadow_wikidoc_10k, f"{DATASETS_PATH}/medical_meadow_wikidoc_10k.jsonl")


def preprocess_medical_meadow_wikidoc_patient_information_6k():
    # Load dataset
    raw_dataset = load_dataset("lavita/medical-qa-datasets", "medical_meadow_wikidoc_patient_information")

    # Initialize processed dataset structure
    medical_meadow_wikidoc_patient_information_6k = {}
    medical_meadow_wikidoc_patient_information_6k["train"] = {}
    medical_meadow_wikidoc_patient_information_6k["train"]["input"] = []
    medical_meadow_wikidoc_patient_information_6k["train"]["output"] = []

    # Custom processing of raw data
    # Dataset is of pretty good quality out of the box, just turn newlines to spaces so the string is all on one line
    for raw_example in raw_dataset["train"]:
        input = remove_newlines(raw_example["input"])
        output = remove_newlines(raw_example["output"])

        medical_meadow_wikidoc_patient_information_6k["train"]["input"].append(input)
        medical_meadow_wikidoc_patient_information_6k["train"]["output"].append(output)

    # Save
    save_processed_dataset(medical_meadow_wikidoc_patient_information_6k, f"{DATASETS_PATH}/medical_meadow_wikidoc_patient_information_6k.jsonl")

def preprocess_pubmed_qa_211k():
    # Load dataset
    raw_dataset = load_dataset("lavita/medical-qa-datasets", "pubmed-qa")

    # Initialize processed dataset structure
    pubmed_qa_211k = {}
    pubmed_qa_211k["train"] = {}
    pubmed_qa_211k["train"]["input"] = []
    pubmed_qa_211k["train"]["output"] = []

    # Custom processing of raw data
    # Dataset is of pretty good quality out of the box, there are some short inputs and outputs that have to be removed
    for raw_example in raw_dataset["train"]:
        input = raw_example["QUESTION"]
        output = raw_example["LONG_ANSWER"]
        
        # Skip also for output too short (from a qualitative human analysis, these are not significative, such as single words, or just comments instead of real answers)
        # Some relevant examples might be lost too, but it's a negligible amount (less than 1%)
        if len(output) <= 50:
            continue

        # If both input and output are valid, turn newlines to spaces so the string is all on one line
        input = remove_newlines(input)
        output = remove_newlines(output)

        pubmed_qa_211k["train"]["input"].append(input)
        pubmed_qa_211k["train"]["output"].append(output)
    
    # This dataset has also a premade validation set, so preprocess that too and merge everything together
    for raw_example in raw_dataset["validation"]:
        input = raw_example["QUESTION"]
        output = raw_example["LONG_ANSWER"]

        if len(output) <= 60:
            continue

        input = remove_newlines(input)
        output = remove_newlines(output)

        pubmed_qa_211k["train"]["input"].append(input)
        pubmed_qa_211k["train"]["output"].append(output)

    # Save
    save_processed_dataset(pubmed_qa_211k, f"{DATASETS_PATH}/pubmed_qa_211k.jsonl")

def save_train_test_split(dataset, dataset_name, train_size=0.9, seed=42):
    '''
        Splits the dataset into train and test sets.
        dataset: list of dictionaries with keys "prompt" and "answer"
        train_size: proportion of the dataset to include in the train split
        seed: random seed for reproducibility
    '''
    # Shuffle the dataset
    random.seed(seed)
    random.shuffle(dataset)

    # Split the dataset
    split_index = int(len(dataset) * train_size)
    train_set = dataset[:split_index]
    test_set = dataset[split_index:]

    # Save the train and test sets
    with open(f"{DATASETS_PATH}/{dataset_name}_train_set.jsonl", "w") as train_file:
        for example in train_set:
            json.dump(example, train_file)
            train_file.write("\n")
        
    with open(f"{DATASETS_PATH}/{dataset_name}_test_set.jsonl", "w") as test_file:
        for example in test_set:
            json.dump(example, test_file)
            test_file.write("\n")

def save_all_train_test(seed=42):
    '''
        Saves all datasets in the DATASETS_PATH directory
        ignore the ones that are already train/test sets
    '''
    for filename in os.listdir(DATASETS_PATH):
        if filename.endswith(".jsonl") and not filename.endswith("_train_set.jsonl") and not filename.endswith("_test_set.jsonl"):
            print(f"Loading {filename}")  # Add this line
            dataset = load_processed_dataset(f"{DATASETS_PATH}/{filename}")
            save_train_test_split(dataset, filename.split(".")[0], seed=seed)
            
    # create ALL_train_set.jsonl
    all_train_data = load_all_train()
    with open(f"{DATASETS_PATH}/ALL_train_set.jsonl", "w") as train_file:
        for example in all_train_data:
            json.dump(example, train_file)
            train_file.write("\n")
    
    # for every dataset (files in the folder that don't end with _train_set.jsonl or _test_set.jsonl) that
    # has less than 10000 examples, unite their train sets as small_sets_united_train_set.jsonl
    # and their test sets as small_sets_united_test_set.jsonl
    small_sets_train = []
    small_sets_test = []

    for filename in os.listdir(DATASETS_PATH):
        if filename.endswith("_train_set.jsonl") or filename.endswith("_test_set.jsonl") or not filename.endswith(".jsonl"):
            continue
        print(f"Loading {filename} for small sets")  # Add this line
        dataset = load_processed_dataset(f"{DATASETS_PATH}/{filename}")
        if len(dataset) < 10000:
            small_sets_train.extend(dataset)
            small_sets_test.extend(dataset)
    
    
    with open(f"{DATASETS_PATH}/small_sets_united_train_set.jsonl", "w") as train_file:
        for example in small_sets_train:
            json.dump(example, train_file)
            train_file.write("\n")
    
    with open(f"{DATASETS_PATH}/small_sets_united_test_set.jsonl", "w") as test_file:
        for example in small_sets_test:
            json.dump(example, test_file)
            test_file.write("\n")
            

def create_balanced_test_set(num_samples=1024):
    '''
        Creates a balanced test set from the test datasets in the DATASETS_PATH directory.
        Samples from each test set proportionally to the number of examples in the test set.
        num_samples: number of samples to include in the validation set
    '''

    # Load all test sets
    datasets = {}
    for filename in os.listdir(DATASETS_PATH):
        if filename.endswith("_test_set.jsonl") and not filename.startswith("small_sets_united"):
            dataset_name = filename.split("_test_set.jsonl")[0]
            datasets[dataset_name] = load_processed_dataset(f"{DATASETS_PATH}/{filename}")

    # Number of examples in each test set
    dataset_sizes = {dataset_name: len(dataset) for dataset_name, dataset in datasets.items()}

    total_size = sum(dataset_sizes.values())
    if total_size == 0:
        print("No test sets found in DATASETS_PATH. No balanced test set created.")
        return

    # Scale them to sum to num_samples
    scale_factor = num_samples / total_size
    # Guarantee at least 1 sample per dataset if possible
    scaled_sizes = {dataset_name: max(1, int(size * scale_factor)) for dataset_name, size in dataset_sizes.items()}

    # Ensure that the sum of the scaled sizes is equal to num_samples
    scaled_sum = sum(scaled_sizes.values())
    if scaled_sum != num_samples:
        last_dataset_name = list(scaled_sizes.keys())[-1]
        scaled_sizes[last_dataset_name] += (num_samples - scaled_sum)

    print(f"Scaled sizes: {scaled_sizes}")

    balanced_test_set = []
    random.seed(42)  # Set seed once for reproducibility
    for dataset_name, dataset in datasets.items():
        sample_size = scaled_sizes[dataset_name]
        if sample_size == 0:
            print(f"Warning: sample_size for {dataset_name} is 0, skipping.")
            continue
        random.shuffle(dataset)
        if sample_size > len(dataset):
            print(f"Warning: Requested {sample_size} samples from {dataset_name}, but only {len(dataset)} available. Using all available samples.")
            sample_size = len(dataset)
        samples = dataset[:sample_size]
        balanced_test_set.extend(samples)
        print(f"Added {sample_size} samples from {dataset_name} to the balanced test set")

    # Save the balanced test set
    with open(f"{DATASETS_PATH}/balanced_test_set.jsonl", "w") as test_file:
        for example in balanced_test_set:
            json.dump(example, test_file)
            test_file.write("\n")

    # Save a json file with the number of examples in each dataset, their ids and the number of examples in the balanced test set
    with open(f"{DATASETS_PATH}/dataset_info.json", "w") as info_file:
        dataset_info = {}
        for dataset_name, dataset in datasets.items():
            dataset_info[dataset_name] = {
                "num_examples": len(dataset),
                "scaled_size": scaled_sizes[dataset_name],
                "balanced_test_set_size": len(balanced_test_set)
            }
        json.dump(dataset_info, info_file, indent=4)


def load_all_train():
    '''
        Loads all train datasets in the DATASETS_PATH directory as one
    '''
    # Load all datasets that end with _train_set.jsonl
    datasets = {}
    for filename in os.listdir(DATASETS_PATH):
        if filename == "ALL_train_set.jsonl":
            continue
        if filename.endswith("_train_set.jsonl"):
            dataset_name = filename.split("_train_set.jsonl")[0]
            datasets[dataset_name] = load_processed_dataset(f"{DATASETS_PATH}/{filename}")
    # Concatenate all datasets
    all_train_data = []
    for dataset_name, dataset in datasets.items():
        all_train_data.extend(dataset)
    
    print(f"Loaded {len(all_train_data)} training examples from {len(datasets)} datasets")

    return all_train_data


##########################
########## MAIN ##########
##########################

# For preprocessing purposes only
if __name__ == "__main__":
    preprocess_all()
    save_all_train_test()
    create_balanced_test_set()
    print("All datasets preprocessed and saved.")
