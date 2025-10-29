import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import yaml
import argparse
from ds_building import load_ds, make_formatter


# load the config yaml file
def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# define the lora config
def get_lora_config(lora_config):
    return LoraConfig(
        r=lora_config['r'],
        lora_alpha=lora_config['alpha'],
        lora_dropout=lora_config['dropout'],
        bias='none',
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_config['target_module']
    )

# training steps:
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args=parser.parse_args()

    config = load_yaml_config(args.config)

    # load model and tokenizer
    model_name=config['model_name']
    base = AutoModelForCausalLM.from_pretrained(model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token=tok.eos_token
    tok.padding_side = "right"
    tok.model_max_length = config['sft_training']['max_len']

    # Apply LoRA
    lora_config = get_lora_config(config['lora_config'])
    model = get_peft_model(base, lora_config)
    model.print_trainable_parameters()

    # dynamic padding + labels at batch time
    collator = DataCollatorForLanguageModeling(
        tokenizer=tok,
        mlm=False,
        pad_to_multiple_of=8,  
    )

    # load dataset
    ds_path = config['dataset']['dataset_path']
    train_ds, val_ds = load_ds(ds_path=ds_path)
    
    train_ds_f = train_ds.map(make_formatter(tok, eos=tok.eos_token, max_len=512), remove_columns=train_ds.column_names)
    val_ds_f  = val_ds.map(make_formatter(tok, eos=tok.eos_token, max_len=512), remove_columns=train_ds.column_names)

    # set precision
    use_bf16 = config['precision'] == 'bf16'

    # training setting
    training_args = TrainingArguments(
        output_dir=config['sft_training']['save_dir'],
        num_train_epochs=config['sft_training']['epochs'],
        per_device_train_batch_size=config['sft_training']['batch_size'],
        learning_rate=float(config['sft_training']['learning_rate']),
        gradient_accumulation_steps=config['sft_training']['gradient_accumulation'],
        warmup_steps=config['sft_training']['warmup_steps'],
        weight_decay=config['sft_training']['weight_decay'],
        bf16=use_bf16,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        logging_steps=10,
        report_to=['tensorboard'],
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds_f,
        eval_dataset=val_ds_f,
        data_collator=collator,
    )

    torch.cuda.reset_peak_memory_stats()
    trainer.train()

    model.save_pretrained(training_args.output_dir)
    tok.save_pretrained(training_args.output_dir)

    print(f"SFT (LoRA) saved to {training_args.output_dir}")

if __name__ == "__main__":
    train()