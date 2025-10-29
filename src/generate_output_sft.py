import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft.peft_model import PeftModel
import yaml
import argparse
import json
from tqdm import tqdm
import os

# load config
def load_yaml_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
args=parser.parse_args()
config = load_yaml_config(args.config)

# load sft model and tokenizer
model_name = config['model_name']
base = AutoModelForCausalLM.from_pretrained(model_name)
tok = AutoTokenizer.from_pretrained(model_name)
tok.pad_token = tok.eos_token
model = PeftModel.from_pretrained(base, config['sft_training']['save_dir'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(torch.bfloat16).to(device)
model.eval()
print("Model ready for inference......")

# generate response
def generate_response(prompt, max_generate_tokens=config['evaluation']['max_generate_length']):
    inputs = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_generate_tokens,
            temperature=config['evaluation']['temperature'],
            top_p=config['evaluation']['top_p'],
            repetition_penalty=config['evaluation']['penalty'],
            do_sample=config['evaluation']['do_sample'],
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id
        )
    gen_ids = output[0, inputs["input_ids"].shape[-1]:]
    response = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return response

# load eval file
eval_ds = config['evaluation']['eval_dataset']
output_dir = config['evaluation']['sft_out']
os.makedirs(output_dir, exist_ok=True)

print(f"Loading {eval_ds} ...")
categories = {}
with open(eval_ds, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line)
        label = data["label"]
        question = data["question"]
        if label not in categories:
            categories[label] = []
        categories[label].append(question)

# generate by category
for label, questions in categories.items():
    print(f"\n Generating answers for category: {label} ({len(questions)} questions)")
    outputs = []

    for q in tqdm(questions, desc=f"Generating ({label})"):
        prompt = f"question: {q.strip()}\nresponse:"
        a = generate_response(prompt)
        outputs.append({"label": label, "question": q, "answer": a})

    # save by category
    output_file = os.path.join(output_dir, f"sft_model_outputs_{label.replace('/', '_')}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
        
    print(f"Saved: {output_file}")

print("\nAll categories finished successfully!")