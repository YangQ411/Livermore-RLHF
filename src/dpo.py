from transformers import AutoTokenizer, AutoModelForCausalLM
from peft.peft_model import PeftModel
from trl import DPOConfig, DPOTrainer
from ds_building import load_dpo_dataset
import argparse
import yaml

# load yaml config
def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

# DPO training
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args=parser.parse_args()

    config = load_yaml_config(args.config)

    # load model and tokenizer
    model_name = config['model_name']
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token

    # load the sft as our policy
    policy = PeftModel.from_pretrained(model, config['sft_training']['save_dir'])

    # check freeze params
    trainable = [n for n, p in policy.named_parameters() if p.requires_grad]
    print(f"[Check] trainable params in policy: {len(trainable)}")
    if len(trainable) == 0:
        if hasattr(policy, "peft_config") and "default" in policy.peft_config:
            policy.peft_config["default"].inference_mode = False
        for n, p in policy.named_parameters():
            if "lora_" in n.lower():    
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
        trainable = [n for n, p in policy.named_parameters() if p.requires_grad]
        print(f"[Fix] re-enabled LoRA, trainable params: {len(trainable)}")
    print("[Sample] first 10 trainable:", trainable[:10])

    # define sft as the ref model
    ref_base = AutoModelForCausalLM.from_pretrained(model_name)
    ref = PeftModel.from_pretrained(ref_base, config['sft_training']['save_dir'])
    # freeze the ref
    for param in ref.parameters():
        param.requires_grad_(False)
    ref.eval()

    # load dataset
    dpo_dataset = config['dataset']['dpo_dataset']
    train_ds, val_ds = load_dpo_dataset(dpo_dataset)

    # define the precision
    use_bf16 = config['precision'] == 'bf16'

    # DPO config
    dpo_config = DPOConfig(
        learning_rate=float(config['dpo_training']['learning_rate']),
        num_train_epochs=config['dpo_training']['epochs'],
        per_device_train_batch_size=config['dpo_training']['batch_size'],
        save_strategy='epoch',
        eval_strategy='epoch',
        report_to=['tensorboard'],
        output_dir=config['dpo_training']['save_dir'],
        load_best_model_at_end=True,
        beta=config['dpo_training']['beta'],
        logging_steps=config['dpo_training']['logging_steps'],
        gradient_accumulation_steps=config['dpo_training']['gradient_accumulation'],
        loss_type="sigmoid",
        warmup_steps=config['dpo_training']['warmup_steps'],
        bf16=use_bf16
    )

    # dpo_training
    dpo_trainer = DPOTrainer(
        model=policy,
        ref_model=ref,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    dpo_trainer.train()
    dpo_trainer.save_model(dpo_config.output_dir)
    tok.save_pretrained(dpo_config.output_dir)

    print(f'Dpo save to {dpo_config.output_dir}')

if __name__ == "__main__":
    train()

