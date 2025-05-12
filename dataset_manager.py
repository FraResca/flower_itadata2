import os
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
        Saves preprocessed dataset. Requires the following dataset structure:
        dataset: dictionary with one key "train"
        dataset["train"]: dictionary with two keys "input" and "output"
        dataset["train"]["input"] and dataset["train"]["output"]: lists of strings
    '''
    target = len(dataset["train"]["input"])
    next_update = 1
    print(f"Saving: {save_path}")
    with open(save_path, "w", encoding="utf-8") as outfile, tqdm(total=100) as pbar:
        for i in range(target):
            outfile.write(dataset["train"]["input"][i])
            outfile.write("\n")
            outfile.write(dataset["train"]["output"][i])
            outfile.write("\n")

            # Print indicative progress
            progress = (i + 1) * 100 / target
            if progress >= next_update:
                pbar.update(1)
                next_update += 1

def load_processed_dataset(load_path):
    '''
        Loads preprocessed dataset.
        Assumes that the file alternates lines: one input followed by the corresponding output
    '''
    dataset = {}
    dataset["train"] = {}
    dataset["train"]["input"] = []
    dataset["train"]["output"] = []

    with open(load_path, "r", encoding="utf-8") as infile:
        for i, line in enumerate(infile):
            line = line.strip()
            if i % 2 == 0:
                dataset["train"]["input"].append(line)
            else:
                dataset["train"]["output"].append(line)

    return dataset

def print_stats():
    '''
        Print the number of examples for each dataset and the total combined number of examples
    '''
    total_examples = 0
    for filename in os.listdir(DATASETS_PATH):
        with open(f"{DATASETS_PATH}/{filename}", "r", encoding="utf-8") as dataset_file:
            example_count = sum(0.5 for _ in dataset_file) # One example is on two lines, so add 1 every two lines, or 0.5 every line
        print(f"{filename}: {example_count} examples")
        total_examples += example_count
    print(f"Total: {total_examples} examples")


################################
########## PREPROCESS ##########
################################
def preprocess_all():
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
    save_processed_dataset(chatdoctor_icliniq_7k, f"{DATASETS_PATH}/chatdoctor_icliniq_7k.txt")

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
    save_processed_dataset(medical_meadow_medical_flashcards_34k, f"{DATASETS_PATH}/medical_meadow_medical_flashcards_34k.txt")


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
    save_processed_dataset(medical_meadow_wikidoc_10k, f"{DATASETS_PATH}/medical_meadow_wikidoc_10k.txt")


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
    save_processed_dataset(medical_meadow_wikidoc_patient_information_6k, f"{DATASETS_PATH}/medical_meadow_wikidoc_patient_information_6k.txt")

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
    save_processed_dataset(pubmed_qa_211k, f"{DATASETS_PATH}/pubmed_qa_211k.txt")

##########################
########## MAIN ##########
##########################

# For preprocessing purposes only
if __name__ == "__main__":
    preprocess_all()