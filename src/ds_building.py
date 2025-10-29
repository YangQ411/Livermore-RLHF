import torch
from datasets import load_dataset


# load dataset
def load_ds(ds_path):
    dataset = load_dataset("json", data_files=ds_path)["train"]
    splits = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds, val_ds = splits["train"], splits["test"]
    return train_ds, val_ds


# define training module
# combine prompt and response for each input to form one text 
# for each input instance, we get
# <s>[USER]
# prompt
# [ASSISTANT]
# response
def make_formatter(tokenizer, eos, max_len=512):
    def fmt(ex):
        text = f"<s>[USER]\n{ex['prompt']}\n[ASSISTANT]\n{ex['response']}{eos}"
        return tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding=False,                 
            return_attention_mask=True,
        )
    return fmt

# process ds for my DPO training
def load_dpo_dataset(jsonl_path):
    ds = load_dataset("json", data_files=jsonl_path)["train"]
    def _ok(ex):
        return bool(ex.get("prompt")) and bool(ex.get("chosen")) and bool(ex.get("rejected"))
    ds = ds.filter(_ok)
    splits = ds.train_test_split(test_size=0.1, seed=42)
    return splits["train"], splits["test"]

