import pandas as pd
from datasets import Dataset




####################################
########## PATH VARIABLES ##########
####################################

DESCRIPTION_PATH = "medpix2/Descriptions.json"
CASE_TOPIC_PATH = "medpix2/Case_topic.json"





#####################################
########## UTILS FUNCTIONS ##########
#####################################

### Helper Functions for Merging Duplicate U_id Entries ###

# Merge lists and remove duplicates
def merge_lists(series):
    merged = []
    for item in series:
        if isinstance(item, list):
            merged.extend(item)
        else:
            merged.append(item)
    return list(set(merged))

# Merge dictionaries safely
def merge_dicts_safe(series):
    merged = {}
    for d in series:
        if isinstance(d, dict):
            for key, value in d.items():
                if key not in merged or merged[key] is None:
                    merged[key] = value
                elif isinstance(value, str) and isinstance(merged[key], str) and merged[key] != value:
                    merged[key] += f" || {value}"
    return merged

# Create input and output strings:
def in_out_strings(entry):
    instring = ''
    instring += entry.Description['Sex'] + ', ' + entry.Description['Age'] + '. '
    instring += entry.Description['Caption'].replace('||', '') + ' ' if (entry.Description['Caption'].replace('||', '')).endswith('.') else entry.Description['Caption'].replace('||', '') + '. '
    try:
        instring += entry.Case['History'] + '. ' if entry.Case['History'].endswith('.') else entry.Case['History'] + '. '
    except: 
        pass
    try:
        if entry.Case['Exam'] != 'N/A': 
            instring += entry.Case['Exam'] + ' ' if entry.Case['Exam'].endswith('.') else entry.Case['Exam'] + '. '
    except:
        pass
    try:
        instring += entry.Case['Findings'] + ' ' if entry.Case['Findings'].endswith('.') else entry.Case['Findings'] + '. '
    except:
        pass
    
    outstring = ''
    outstring += entry.Topic['Title'] + ' ' if entry.Topic['Title'].endswith('.') else entry.Topic['Title'] + '. '
    outstring += entry.Topic['Disease Discussion'] if entry.Topic['Disease Discussion'].endswith('.') else entry.Topic['Disease Discussion'] + '. '

    return instring, outstring

################################
########## PREPROCESS ##########
################################

# Returns two dataframes:
#   > merged_df merges information from Descrptions.json and Case_topic.json to have both in a single entry
#   > merged_combined_df also combines rows that refer to the same clinical case. It greatly reduces the number of available examples
def preprocess():

    # Read the JSON files into DataFrames
    description_df = pd.read_json(DESCRIPTION_PATH)
    case_topic_df = pd.read_json(CASE_TOPIC_PATH)

    # Initial Merge on 'U_id' (2050 total entries)
    merged_df = pd.merge(description_df, case_topic_df, on='U_id', how='inner')

    # Consolidate all entries by 'U_id' (671 total entries)
    merged_combined_df = merged_df.groupby('U_id').agg({
        'Type': 'first',
        'image': lambda x: list(set(x)),
        'Description': merge_dicts_safe,
        'Location': lambda x: ', '.join(sorted(set(x))),
        'Location Category': lambda x: ', '.join(sorted(set(x))),
        'TAC': merge_lists,
        'MRI': merge_lists,
        'Case': merge_dicts_safe,
        'Topic': merge_dicts_safe
    }).reset_index()

    return merged_df, merged_combined_df





##############################
########## DATASETS ##########
##############################

# Create MedPix2 dataset with the following structure:
#   > A dictionary with only one key: "train"
#   > medipix2["train"] is also a dictionary, with two keys: "input" and " output"
#   > Both contain a list of string. Strings in the same position (index) across the two lists refer to the same clinical case

# Creates the non-combined dataset (more examples)
def medpix2_2050():
    # Dataset
    medpix2 = {}
    medpix2['train'] = {}
    medpix2['train']['input'] = []
    medpix2['train']['output'] = []

    # Get df
    merged_df, _ = preprocess()

    # Create entries
    for entry in merged_df.itertuples():
        instring, outstring = in_out_strings(entry)

        medpix2['train']['input'].append(instring)
        medpix2['train']['output'].append(outstring)
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict({
        "instruction": ["Interpret this medical case and provide diagnosis information"] * len(medpix2["train"]["input"]),
        "input": medpix2['train']['input'],
        "output": medpix2['train']['output']
    })
    
    # Return in the structure expected by load_dataset_fine
    result = {"train": dataset}
    
    return result

# Creates the combined dataset (less examples)
def medpix2_671():
    # Dataset
    medpix2 = {}
    medpix2['train'] = {}
    medpix2['train']['input'] = []
    medpix2['train']['output'] = []

    # Get df
    _, merged_combined_df = preprocess()

    # Create entries
    for entry in merged_combined_df.itertuples():
        instring, outstring = in_out_strings(entry)

        medpix2['train']['input'].append(instring)
        medpix2['train']['output'].append(outstring)

    return medpix2





##########################
########## MAIN ##########
##########################

# For debugging purposes only
if __name__ == "__main__":
    medpix2_large = medpix2_2050()
    medpix2_small = medpix2_671()

    print("Large MedPix2: ", len(medpix2_large['train']['input']), len(medpix2_large['train']['output']))
    print("Small MedPix2: ", len(medpix2_small['train']['input']), len(medpix2_small['train']['output']))