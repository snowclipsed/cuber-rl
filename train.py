#!/usr/bin/env python
"""Train Qwen model on Rubik's Cube environment with GRPO"""

import os
import torch
import verifiers as vf
import wandb

# Initialize wandb
wandb.init(project="rubiks-cube-rl", name="qwen3-4b-grpo")

# Load model and tokenizer
model_name = "Qwen/Qwen3-4B-Thinking-2507"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

# Load environment
env = vf.load_environment(
    "cuber-rl",  # Your environment name
    difficulties=['easy', 'medium'],
    max_moves_per_turn=3,
    max_episode_steps=20
)

# Get default GRPO training arguments
run_name = "rubiks-cube-grpo-qwen3-4b"
training_args = vf.grpo_defaults(run_name=run_name)

# Customize training arguments
training_args.per_device_train_batch_size = 4  # Good for 4090s
training_args.num_generations = 8  # Number of rollouts per prompt
training_args.gradient_accumulation_steps = 4
training_args.max_tokens = 256  # Max tokens per turn
training_args.max_seq_len = 2048  # Max sequence length
training_args.max_steps = 500  # Total training steps
training_args.eval_strategy = "steps"
training_args.eval_steps = 50
training_args.save_steps = 100
training_args.logging_steps = 10
training_args.mask_env_responses = True
training_args.max_grad_norm = 0.1
training_args.beta = 0.0  # GRPO beta parameter
training_args.async_generation_timeout = 600
training_args.learning_rate = 1e-6
training_args.warmup_ratio = 0.1
training_args.fp16 = True  # Use mixed precision
training_args.gradient_checkpointing = True
training_args.output_dir = "./rubiks_cube_checkpoints"
training_args.report_to = "wandb"

# vLLM server settings
training_args.vllm_server_host = "localhost"
training_args.vllm_server_port = 8000
training_args.max_concurrent = 100

# Initialize trainer
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=env,
    args=training_args,
)

# Train
trainer.train()

# Save final model
trainer.save_model("./final_rubiks_model")
print("Training complete!")