import json
import os
import random
import yaml
import argparse

# load yaml config
def load_yaml_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
config = load_yaml_config(args.config)

random.seed(args.seed)

# the output files
baseline_dir = config["evaluation"]["base_out"]
sft_dir = config["evaluation"]["sft_out"]
dpo_dir = config["evaluation"]["dpo_out"]
out_path = config["evaluation"]['combine']

# collect baseline files as the anchor list 
baseline_files = [
    f for f in os.listdir(baseline_dir)
    if f.startswith("baseline_model_outputs_") and f.endswith(".json")
]

battle_inputs = []

for base_file in baseline_files:
    label = base_file.replace("baseline_model_outputs_", "").replace(".json", "")

    base_path = os.path.join(baseline_dir, base_file)
    sft_path  = os.path.join(sft_dir, f"sft_model_outputs_{label}.json")
    dpo_path  = os.path.join(dpo_dir, f"dpo_model_outputs_{label}.json")

    missing = []
    if not os.path.exists(sft_path): missing.append("SFT")
    if not os.path.exists(dpo_path): missing.append("DPO")
    if missing:
        print(f"Skipping category `{label}`: missing {', '.join(missing)} file(s).")
        continue

    with open(base_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    with open(sft_path, "r", encoding="utf-8") as f:
        sft = json.load(f)
    with open(dpo_path, "r", encoding="utf-8") as f:
        dpo = json.load(f)

    n = min(len(baseline), len(sft), len(dpo))
    print(f"Processing category: {label} (pairs: {n})")

    for i in range(n):
        base_item = baseline[i]
        sft_item  = sft[i]
        dpo_item  = dpo[i]

        q = base_item["question"]
    
        assert q == sft_item["question"] == dpo_item["question"], f"Question mismatch in {label} at index {i}"

        trio = [
            ("baseline_model", base_item["answer"]),
            ("sft_model",      sft_item["answer"]),
            ("dpo_model",      dpo_item["answer"]),
        ]
        random.shuffle(trio)  

        entry = {
            "category": label,
            "question": q,
            "answer_a": trio[0][1],
            "answer_b": trio[1][1],
            "answer_c": trio[2][1],
            "a_model":  trio[0][0],
            "b_model":  trio[1][0],
            "c_model":  trio[2][0],
        }
        battle_inputs.append(entry)

# save to single file
os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(battle_inputs, f, indent=2, ensure_ascii=False)

print(f"\nCombined 3-way file saved: {out_path} (total: {len(battle_inputs)})")