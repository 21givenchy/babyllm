"""Train MoE-enhanced ExplanationEngine using nanoBeard's training loop.

Combines:
- nanoBeard's training architecture
- Tutel MoE for expert routing
- 4-stage learning (Exploration → Autonomous)
- Self-improvement loop
"""

import argparse
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from dotenv import load_dotenv

from moe_explanation_engine import MoEExplanationEngine
from dataset import IntentDataset, create_training_data, SimpleTokenizer

load_dotenv()


class MoEConfig:
    """Config for MoE ExplanationEngine training."""
    
    # Model
    vocab_size: int = 8192
    embed_dim: int = 384
    num_layers: int = 6
    num_experts: int = 8
    experts_per_token: int = 2
    hidden_dim: int = 1536
    
    # Training
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    warmup_iters: int = 200
    max_iters: int = 20000
    lr_decay_iters: int = 20000
    
    # Batch
    batch_size: int = 32
    block_size: int = 256
    gradient_accumulation_steps: int = 1
    
    # Evaluation
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 10
    
    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16" if torch.cuda.is_available() else "float32"
    seed: int = 1337
    compile: bool = False
    
    # Checkpointing
    run_dir: str = "runs/moe_explanation"
    
    def __post_init__(self):
        os.makedirs(self.run_dir, exist_ok=True)


def get_lr(it: int, config: MoEConfig) -> float:
    """Cosine decay with warmup."""
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / (config.warmup_iters + 1)
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def train_moe_explanation(config: MoEConfig):
    """Train MoE ExplanationEngine using 4-stage learning."""
    
    print(f"\n=== Training MoE ExplanationEngine ===")
    print(f"Device: {config.device}, dtype: {config.dtype}")
    print(f"Model: {config.num_experts} experts, top-{config.experts_per_token}")
    
    # Setup
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config.dtype]
    
    ctx = autocast(device_type="cuda", dtype=ptdtype) if config.device == "cuda" and config.dtype != "float32" else nullcontext()
    scaler = GradScaler(enabled=(config.dtype == "float16"))
    
    # Model
    model = MoEExplanationEngine(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        num_layers=config.num_layers,
        num_experts=config.num_experts,
        experts_per_token=config.experts_per_token,
        hidden_dim=config.hidden_dim,
    ).to(config.device)
    
    if config.compile:
        model = torch.compile(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {total_params:,} params")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Data
    texts, actions = create_training_data()
    tokenizer = SimpleTokenizer()
    dataset = IntentDataset(texts, actions, tokenizer)
    
    print(f"\n=== Stage 1: Exploration ===")
    print(f"Training on {len(dataset)} observations")
    
    # Training loop
    iter_num = 0
    best_val_loss = float("inf")
    t0 = time.time()
    
    while iter_num < config.max_iters:
        lr = get_lr(iter_num, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        
        # Random batch
        idx = torch.randint(0, len(dataset), (config.batch_size,))
        batch_data = [dataset[i.item()] for i in idx]
        
        # Create batch
        batch_texts = torch.stack([d["text"] for d in batch_data]).to(config.device)
        batch_actions = torch.stack([d["action"] for d in batch_data]).unsqueeze(1).to(config.device)
        
        # Forward pass
        with ctx:
            logits, loss = model(batch_texts, batch_actions)
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Logging
        if iter_num % config.log_interval == 0 and iter_num > 0:
            print(f"  iter {iter_num} | loss {loss.item():.4f} | lr {lr:.2e}")
        
        if iter_num % config.eval_interval == 0:
            elapsed = time.time() - t0
            print(f"step {iter_num:>6d} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.1f}s")
        
        iter_num += 1
    
    print(f"\n=== Training Complete ===")
    print(f"Final loss: {loss.item():.4f}")
    
    # Save checkpoint
    checkpoint = {
        "model": model.state_dict() if not isinstance(model, torch.nn.parallel.DataParallel) else model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "iter_num": iter_num,
    }
    
    ckpt_path = os.path.join(config.run_dir, "moe_explanation_ckpt.pth")
    torch.save(checkpoint, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    config = MoEConfig()
    train_moe_explanation(config)
