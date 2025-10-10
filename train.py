import argparse
import json
import verifiers as vf
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', required=True)
args = parser.parse_args()

with open(args.config) as f:
    cfg = json.load(f)

wandb.init(project=cfg['wandb_project'], name=cfg['wandb_name'])

model, tokenizer = vf.get_model_and_tokenizer(cfg['model_name'])

env = vf.load_environment(
    cfg['env_name'],
    difficulties=cfg['difficulties'],
    max_moves_per_turn=cfg['max_moves_per_turn'],
    max_episode_steps=cfg['max_episode_steps']
)

training_args = vf.grpo_defaults(run_name=cfg['run_name'])

for k, v in cfg['training_args'].items():
    setattr(training_args, k, v)

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=env,
    args=training_args,
    peft_config=vf.lora_defaults()
)

trainer.train()
trainer.save_model(cfg['output_model_path'])