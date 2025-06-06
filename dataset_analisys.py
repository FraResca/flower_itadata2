import numpy as np
import matplotlib.pyplot as plt
import json
import os


def create_histogram(word_counts: list, dataset_name: str = "Dataset Name"):
    plt.figure(figsize=(10, 6))
    plt.hist(word_counts, bins=30, color='blue', alpha=0.7)
    plt.title(f'Word Count Distribution {dataset_name}')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show(block = False)  # Use block=False to allow multiple plots in a row
    plt.savefig(f'plots/{dataset_name}_word_count_distribution.png')  # Save the histogram as an image file


def plot_dataset_statistics(dataset: str):
    #dataset are jsonl files with the following keys: prompt, answer
    dataset_dict = {}


    dataset_dict["prompt"] = []
    dataset_dict["answer"] = []
    dataset_dict["word_count"] = []
    with open(dataset, 'r') as f:
        for line in f:
            data = json.loads(line)
            prompt = data.get('prompt', '')
            answer = data.get('answer', '')
            word_count = len(prompt.split()) + len(answer.split())
            #print(f"Prompt: {prompt}")
            #print(f"Answer: {answer}")
            #print(f"Word Count: {word_count}")
            dataset_dict["prompt"].append(prompt)
            dataset_dict["answer"].append(answer)
            dataset_dict["word_count"].append(word_count)
    
    return dataset_dict

def merge_datasets(datasets: list, path: str):
    #dataset are jsonl files with the following keys: prompt, answer
    dataset_dict = {}


    dataset_dict["prompt"] = []
    dataset_dict["answer"] = []
    dataset_dict["word_count"] = []

    for dataset in datasets:
        with open(path + dataset, 'r') as f:
            for line in f:
                data = json.loads(line)
                prompt = data.get('prompt', '')
                answer = data.get('answer', '')
                word_count = len(prompt.split()) + len(answer.split())
                #print(f"Prompt: {prompt}")
                #print(f"Answer: {answer}")
                #print(f"Word Count: {word_count}")
                dataset_dict["prompt"].append(prompt)
                dataset_dict["answer"].append(answer)
                dataset_dict["word_count"].append(word_count)
    
    print(f"Total number of samples: {len(dataset_dict['word_count'])}")
        
    return dataset_dict


def dataset_split(dataset_dict: dict, len_threshold: int = 60, path: str = './datasets/'):
    # Split the dataset into two parts based on the word count threshold
   
    # Create two lists for short and long samples
    # Short samples are those with word count less than len_threshold
    # Long samples are those with word count greater than or equal to len_threshold

    #iterate through the dataset and split into short and long samples
    # Return two lists of dictionaries, one for short samples and one for long samples

    short_samples = []
    long_samples = []
    
    for i, word_count in enumerate(dataset_dict["word_count"]):
        if word_count < len_threshold:
            short_samples.append({
                "prompt": dataset_dict["prompt"][i],
                "answer": dataset_dict["answer"][i],
                "word_count": word_count
            })
        else:
            long_samples.append({
                "prompt": dataset_dict["prompt"][i],
                "answer": dataset_dict["answer"][i],
                "word_count": word_count
            })
    
    print(f"Short samples: {len(short_samples)}")
    print(f"Long samples: {len(long_samples)}")

    # save the split datasets to jsonl files
    with open(path + 'short_samples.jsonl', 'w') as f:
        for sample in short_samples:
            f.write(json.dumps(sample) + '\n')
    with open(path + 'long_samples.jsonl', 'w') as f:
        for sample in long_samples:
            f.write(json.dumps(sample) + '\n')

    return short_samples, long_samples

def print_dataset_statistics(word_count: list):
    # Print the statistics of the word count
    print(f"Total number of samples: {len(word_count)}")
    print(f"Average word count: {np.mean(word_count):.2f}")
    print(f"Median word count: {np.median(word_count):.2f}")
    print(f"Standard deviation of word count: {np.std(word_count):.2f}")
    print(f"Minimum word count: {np.min(word_count)}")
    print(f"Maximum word count: {np.max(word_count)}")

    #print quantiles
    print(f"25th percentile word count: {np.percentile(word_count, 25)}")
    print(f"50th percentile word count: {np.percentile(word_count, 50)}")
    print(f"75th percentile word count: {np.percentile(word_count, 75)}")
    print(f"90th percentile word count: {np.percentile(word_count, 90)}")

    #$create_histogram(dataset_dict["word_count"], dataset.split('/')[-1].replace('.jsonl', ''))

def merge_trainsets_split_save():
    # Find the train sets (their name ends with '_train_set') in the dataset_list and merge them

    dataset_path = './datasets/'
    dataset_list = ["medical_meadow_medical_flashcards_34k.jsonl", "medical_meadow_wikidoc_10k.jsonl", "medical_meadow_wikidoc_patient_information_6k.jsonl", "pubmed_qa_211k.jsonl"]
    # add _train_set to the dataset names
    dataset_list = [dataset.replace('.jsonl', '_train_set.jsonl') for dataset in dataset_list]

    merged_dataset = merge_datasets(dataset_list, dataset_path)
    short_samples, long_samples = dataset_split(merged_dataset, len_threshold=90)

    # print the number of samples in each split and save the files with _train_set

    print(f"Number of short samples: {len(short_samples)}")
    print(f"Number of long samples: {len(long_samples)}")

    with open(dataset_path + 'short_train_set.jsonl', 'w') as f:
        for sample in short_samples:
            f.write(json.dumps(sample) + '\n')
    with open(dataset_path + 'long_train_set.jsonl', 'w') as f:
        for sample in long_samples:
            f.write(json.dumps(sample) + '\n')

if __name__ == "__main__":
    dataset_path = './datasets/'  # Replace with your dataset path
    dataset_list = ["medical_meadow_medical_flashcards_34k.jsonl", "medical_meadow_wikidoc_10k.jsonl", "medical_meadow_wikidoc_patient_information_6k.jsonl", "pubmed_qa_211k.jsonl"]

    global_word_count = []

    '''

    for dataset in dataset_list:
        print(f"Processing dataset: {dataset}")
        dataset_dict = plot_dataset_statistics(dataset_path + dataset)
        global_word_count.extend(dataset_dict["word_count"])    
        print(f"Finished processing dataset: {dataset}")
    create_histogram(global_word_count, "All Datasets")
    print_dataset_statistics(global_word_count)

    '''

    merged_dataset = merge_datasets(dataset_list, dataset_path)
    short_samples, long_samples = dataset_split(merged_dataset, len_threshold=90)
    print("SHORT:")
    print_dataset_statistics([sample["word_count"] for sample in short_samples])
    create_histogram([sample["word_count"] for sample in short_samples], "Short Samples")

    print("LONG:")
    print_dataset_statistics([sample["word_count"] for sample in long_samples])
    create_histogram([sample["word_count"] for sample in long_samples], "Long Samples")

    print(f"Long samples: {len(long_samples)}")

    print("All datasets processed.")