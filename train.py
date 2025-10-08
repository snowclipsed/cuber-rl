import os
import torch
import verifiers as vf
import wandb
from transformers import TrainerCallback

wandb.init(project="cuber-rl", name="qwen3-4b-grpo-better-instuctions")

# Simple callback to log rewards
class RewardLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "rewards" in logs:
            rewards = logs["rewards"]
            if isinstance(rewards, (list, torch.Tensor)):
                rewards_t = torch.tensor(rewards) if isinstance(rewards, list) else rewards
                wandb.log({
                    "reward/mean": rewards_t.mean().item(),
                    "reward/max": rewards_t.max().item(),
                    "reward/min": rewards_t.min().item(),
                    "reward/std": rewards_t.std().item(),
                }, step=state.global_step)

model_name = "willcb/Qwen3-4B"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

env = vf.load_environment(
    "cuber-rl",
    difficulties=['very_easy'],
    max_moves_per_turn=1,
    max_episode_steps=8
)

run_name = "rubiks-cube-grpo-qwen3b"
training_args = vf.grpo_defaults(run_name=run_name)

training_args.per_device_train_batch_size = 4
training_args.num_generations = 8
training_args.gradient_accumulation_steps = 4
training_args.max_tokens = 2048
training_args.max_seq_len = 4096

training_args.max_steps = 100
training_args.eval_strategy = "steps"
training_args.eval_steps = 2
training_args.save_steps = 10
training_args.logging_steps = 1

training_args.mask_env_responses = True
training_args.max_grad_norm = 0.1
training_args.beta = 0.00
training_args.async_generation_timeout = 600
training_args.learning_rate = 2e-6
training_args.warmup_ratio = 0.1
training_args.fp16 = True
training_args.gradient_checkpointing = True
training_args.output_dir = "./rubiks_cube_checkpoints"
training_args.report_to = "wandb"

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=env,
    args=training_args,
    callbacks=[RewardLoggingCallback()],  # Add callback here
)

trainer.train()
trainer.save_model("./final_rubiks_model")