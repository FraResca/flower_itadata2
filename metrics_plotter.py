import glob
import json
import matplotlib.pyplot as plt
import os
import re
import sys
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})

SEARCH_PATH = f"fabric_script/collected_files/**/server{sys.argv[1]}alfa*_metrics.jsonl"

json_obj_pattern = re.compile(r'\{[^}]*\}', re.DOTALL)

# Store all experiments' data
experiments = []

for filepath in glob.glob(SEARCH_PATH, recursive=True):
    print(f"Processing file: {filepath}")
    rounds = []
    rougeL = []
    bert = []

    with open(filepath, 'r') as file:
        content = file.read()
        for match in json_obj_pattern.finditer(content):
            try:
                data = json.loads(match.group())
                rounds.append(data['round'])
                rougeL.append(data['rougeL'])
                bert.append(data['bert'])
            except Exception as e:
                print(f"Error parsing object in {filepath}: {e}\nObject: {match.group()}")

    if rounds:
        experiments.append({
            "label": os.path.basename(filepath),
            "rounds": rounds,
            "rougeL": rougeL,
            "bert": bert
        })
    else:
        print(f"No valid data found in {filepath}.")

if len(sys.argv) != 2:
    print("Usage: python metrics_plotter.py <experiment_label>")
    sys.exit(1)

num = sys.argv[1]

# Plot RougeL
plt.figure(figsize=(10, 5))
for exp in experiments:
    plt.plot(exp["rounds"], exp["rougeL"], marker='o', label=exp["label"])
plt.title('RougeL across Experiments', fontsize=20)
plt.xlabel('Round', fontsize=18)
plt.ylabel('RougeL Score', fontsize=18)
plt.legend(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"rougeL_superposed_{num}.pdf")
print("RougeL plot saved as rougeL_superposed.pdf")
# plt.show()

# Plot BERTScore
plt.figure(figsize=(10, 5))
for exp in experiments:
    plt.plot(exp["rounds"], exp["bert"], marker='x', label=exp["label"])
plt.title('BERTScore across Experiments', fontsize=20)
plt.xlabel('Round', fontsize=18)
plt.ylabel('BERTScore', fontsize=18)
plt.legend(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"bert_superposed_{num}.pdf")
print("BERTScore plot saved as bert_superposed.pdf")
# plt.show()