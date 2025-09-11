#!/usr/bin/env python
import os
import torch
import verifiers as vf
import wandb

wandb.init(project="rubiks-cube-rl", name="qwen3-instruct-grpo")

model_name = "Qwen/Qwen3-4B-Instruct-2507"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

env = vf.load_environment(
    "cuber-rl",
    difficulties=['easy', 'medium'],
    max_moves_per_turn=3,
    max_episode_steps=20
)

run_name = "rubiks-cube-grpo-qwen3b"
training_args = vf.grpo_defaults(run_name=run_name)

training_args.per_device_train_batch_size = 4
training_args.num_generations = 8
training_args.gradient_accumulation_steps = 4
training_args.max_tokens = 2048
training_args.max_seq_len = 2048
training_args.max_steps = 500
training_args.eval_strategy = "steps"
training_args.eval_steps = 50
training_args.save_steps = 100
training_args.logging_steps = 10
training_args.mask_env_responses = True
training_args.max_grad_norm = 0.1
training_args.beta = 0.0
training_args.async_generation_timeout = 600
training_args.learning_rate = 1e-6
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
)

trainer.train()
trainer.save_model("./final_rubiks_model")