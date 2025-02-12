"""
Training script for GPT-MoE model with LoRA integration.
This script supports both single GPU and distributed data parallel (DDP) training.

To run on a single GPU:
$ python train.py

To run with DDP on 4 GPUs on 1 node:
$ torchrun --standalone --nproc_per_node=1 train_torch.py

Dependencies:
- torch
- omegaconf
- numpy
- peft (for LoRA)
"""

import os
import time
import math
import pickle
import re
from contextlib import nullcontext
from typing import Dict, Any, Optional

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model_torch import GPTConfig, GPT
from omegaconf import OmegaConf

# Load configuration
try:
    config = OmegaConf.load('config_torch.yaml')
except Exception as e:
    raise RuntimeError(f"Failed to load config: {str(e)}")

# -----------------------------------------------------------------------------
# Setup for DDP and device configuration
# -----------------------------------------------------------------------------

def setup_distributed() -> tuple:
    """
    Setup distributed training environment.
    
    Returns:
        tuple: (ddp, device, master_process, seed_offset, ddp_world_size, ddp_local_rank)
        
    Raises:
        RuntimeError: If required environment variables are missing or invalid
    """
    try:
        # Check for distributed mode
        rank = os.environ.get('RANK')
        if rank is None:
            ddp = False
        else:
            ddp = int(rank) != -1
            
        if ddp:
            # Validate required environment variables
            required_vars = ['LOCAL_RANK', 'WORLD_SIZE']
            missing_vars = [var for var in required_vars if os.environ.get(var) is None]
            if missing_vars:
                raise RuntimeError(f"Missing required environment variables: {missing_vars}")
            
            # Parse environment variables
            ddp_rank = int(rank)
            ddp_local_rank = int(os.environ['LOCAL_RANK'])
            ddp_world_size = int(os.environ['WORLD_SIZE'])
            
            # Setup device and process group
            device = f'cuda:{ddp_local_rank}'
            torch.cuda.set_device(device)
            init_process_group(backend=config.backend)
            
            # Set process roles
            master_process = ddp_rank == 0
            seed_offset = ddp_rank
            
            # Adjust gradient accumulation steps
            original_gas = config.gradient_accumulation_steps
            if original_gas % ddp_world_size != 0:
                raise ValueError(
                    f"gradient_accumulation_steps ({original_gas}) must be divisible "
                    f"by world_size ({ddp_world_size})"
                )
            # Store adjusted value in a new field
            config.adjusted_gradient_accumulation_steps = original_gas // ddp_world_size
        else:
            # Single GPU/CPU setup
            master_process = True
            seed_offset = 0
            ddp_world_size = 1
            ddp_local_rank = 0
            device = config.device
            config.adjusted_gradient_accumulation_steps = config.gradient_accumulation_steps
            
        return ddp, device, master_process, seed_offset, ddp_world_size, ddp_local_rank
        
    except Exception as e:
        raise RuntimeError(f"Failed to setup distributed training: {str(e)}")

ddp, device, master_process, seed_offset, ddp_world_size, ddp_local_rank = setup_distributed()

# Setup device and dtype
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Initialize system
if master_process:
    os.makedirs(config.out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

tokens_per_iter = (config.gradient_accumulation_steps * ddp_world_size * 
                  config.batch_size * config.block_size)
print(f"Tokens per iteration will be: {tokens_per_iter:,}")

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

class DataLoader:
    """Handles data loading and preprocessing."""
    
    def __init__(self, data_dir: str, config: Any, device: str, device_type: str):
        self.data_dir = data_dir
        self.config = config
        self.device = device
        self.device_type = device_type
        
        # Cache memory maps
        self.data_maps = {}
        for split in ['train', 'val']:
            file_path = os.path.join(self.data_dir, f'{split}.bin')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found: {file_path}")
            self.data_maps[split] = np.memmap(file_path, dtype=np.uint16, mode='r')
        
    def get_batch(self, split: str) -> tuple:
        """
        Fetches a batch of data with improved error handling.
        
        Args:
            split (str): 'train' or 'val'
            
        Returns:
            tuple: (input_tensor, target_tensor)
            
        Raises:
            FileNotFoundError: If data file is not found
            ValueError: If data is corrupted or invalid
            RuntimeError: If CUDA errors occur
        """
        try:
            data = self.data_maps[split]
            if len(data) <= self.config.block_size:
                raise ValueError(f"Data file {split}.bin is too small for block_size {self.config.block_size}")
            
            # Generate random indices with bounds checking
            max_index = len(data) - self.config.block_size
            ix = torch.randint(max_index, (self.config.batch_size,))
            
            # Load data with error handling and retry logic
            x = []
            y = []
            valid_samples = 0
            max_retries = self.config.batch_size * 2  # Allow up to 2x batch_size attempts
            retry_count = 0
            
            while valid_samples < self.config.batch_size and retry_count < max_retries:
                try:
                    # Generate a new index if we need more samples
                    if len(ix) <= valid_samples:
                        new_ix = torch.randint(max_index, (self.config.batch_size,))
                        ix = torch.cat([ix[valid_samples:], new_ix])
                    
                    # Convert tensor index to Python integer
                    idx = ix[valid_samples].item()
                    x_slice = data[idx:idx+self.config.block_size]
                    y_slice = data[idx+1:idx+1+self.config.block_size]
                    
                    if len(x_slice) != self.config.block_size or len(y_slice) != self.config.block_size:
                        print(f"Warning: Skipping invalid slice at index {idx}")
                        retry_count += 1
                        continue
                        
                    x.append(torch.from_numpy(x_slice.astype(np.int64)))
                    y.append(torch.from_numpy(y_slice.astype(np.int64)))
                    valid_samples += 1
                    
                except Exception as e:
                    print(f"Warning: Error loading data at index {idx}: {str(e)}")
                    retry_count += 1
                    continue
            
            if valid_samples < self.config.batch_size:
                raise ValueError(
                    f"Failed to gather enough valid samples. "
                    f"Got {valid_samples}/{self.config.batch_size} after {retry_count} attempts"
                )
            
            x = torch.stack(x)
            y = torch.stack(y)
            
            try:
                if self.device_type == 'cuda':
                    # Pin arrays x, y, which allows us to move them to GPU asynchronously
                    x = x.pin_memory().to(self.device, non_blocking=True)
                    y = y.pin_memory().to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device)
                    y = y.to(self.device)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    raise RuntimeError(f"GPU out of memory while loading batch: {str(e)}")
                raise e
                
            return x, y
            
        except Exception as e:
            print(f"Error in get_batch for split {split}: {str(e)}")
            raise

class Trainer:
    """Handles model training and evaluation."""
    
    def __init__(self, model, optimizer, config, data_loader, wandb_enabled: bool = False):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.data_loader = data_loader
        self.scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16'))
        self.raw_model = model.module if isinstance(model, DDP) else model
        self.running_mfu = -1.0
        self.iter_num = 0
        self.best_val_loss = float('inf')
        self.wandb_enabled = wandb_enabled
        self.model_args = self.raw_model.config.__dict__
        
        # Initialize gradient tracking
        self.last_grad_norm = 0.0
        
    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, Dict[str, float]]:
        """
        Estimate loss metrics over train and validation sets.
        
        Returns:
            Dict containing metrics for each split
        """
        out = {}
        self.model.eval()
        
        try:
            for split in ['train', 'val']:
                losses = torch.zeros(self.config.eval_iters)
                main_losses = torch.zeros(self.config.eval_iters)
                aux_losses = torch.zeros(self.config.eval_iters)
                
                for k in range(self.config.eval_iters):
                    try:
                        X, Y = self.data_loader.get_batch(split)
                        with ctx:
                            logits, loss, main_loss, aux_loss = self.model(X, Y)
                            
                        # Check for invalid loss values
                        if not torch.isfinite(loss):
                            raise ValueError(f"Non-finite loss detected in {split} evaluation")
                        
                        losses[k] = loss.item()
                        main_losses[k] = main_loss.item()
                        aux_losses[k] = aux_loss.item()
                        
                    except Exception as e:
                        print(f"Error in evaluation iteration {k} for {split}: {str(e)}")
                        continue
                
                # Calculate metrics
                out[split] = {
                    'loss': losses.mean().item(),
                    'main_loss': main_losses.mean().item(),
                    'aux_loss': aux_losses.mean().item(),
                    'perplexity': torch.exp(losses.mean()).item(),
                }
                    
        except Exception as e:
            print(f"Error in estimate_loss: {str(e)}")
            raise
        finally:
            self.model.train()
            
        return out

    def train_step(self, X: torch.Tensor, Y: torch.Tensor) -> tuple:
        """
        Perform a single training step.
        
        Args:
            X: Input tensor
            Y: Target tensor
            
        Returns:
            tuple: (loss, main_loss, aux_loss, grad_norm)
            
        Raises:
            RuntimeError: If OOM or other runtime errors occur
        """
        try:
            # Forward pass
            with ctx:
                logits, loss, main_loss, aux_loss = self.model(X, Y)
                loss = loss / self.config.adjusted_gradient_accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Compute gradient norm (unscaled)
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.stack([
                p.grad.norm() for p in self.model.parameters() if p.grad is not None
            ]).norm()
            self.last_grad_norm = grad_norm.item()
            
            return loss, main_loss, aux_loss, grad_norm
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"GPU OOM in training step: {str(e)}. "
                    f"Current batch size: {self.config.batch_size}, "
                    f"Try reducing batch size or enabling gradient checkpointing."
                )
            raise e

    def train(self):
        """Main training loop with improved error handling and monitoring."""
        print("Starting training...")
        t0 = time.time()
        local_iter_num = 0
        
        try:
            X, Y = self.data_loader.get_batch('train')
            
            while True:
                # Learning rate scheduling
                lr = get_lr(self.iter_num) if self.config.decay_lr else self.config.learning_rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

                # Evaluation
                if self.iter_num % self.config.eval_interval == 0 and master_process:
                    losses = self.estimate_loss()
                    print(f"Step {self.iter_num}:")
                    for split, metrics in losses.items():
                        print(f"  {split}: loss {metrics['loss']:.4f}, ppl {metrics['perplexity']:.2f}")
                    
                    if self.wandb_enabled:
                        log_metrics(self.iter_num, losses, lr, self.running_mfu)

                # Training step with gradient accumulation over different batches
                # Note: Each micro-step uses a different batch to effectively increase the batch size
                accumulated_loss = 0
                accumulated_main_loss = 0
                accumulated_aux_loss = 0
                for micro_step in range(self.config.adjusted_gradient_accumulation_steps):
                    if ddp:
                        # Only synchronize gradients on last micro-step
                        self.model.require_backward_grad_sync = (
                            micro_step == self.config.adjusted_gradient_accumulation_steps - 1
                        )
                    
                    try:
                        # Process current batch
                        loss, main_loss, aux_loss, grad_norm = self.train_step(X, Y)
                        accumulated_loss += loss.item()
                        accumulated_main_loss += main_loss.item()
                        accumulated_aux_loss += aux_loss.item()
                        
                        # Get next batch for the next micro-step
                        # This means each micro-step processes a different batch,
                        # effectively increasing diversity in the accumulated gradients
                        X, Y = self.data_loader.get_batch('train')
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"OOM in micro-step {micro_step}. Skipping remaining micro-steps.")
                            break
                        raise e

                # Compute average losses over micro-steps
                num_steps = micro_step + 1  # Account for possible early break
                avg_loss = accumulated_loss / num_steps
                avg_main_loss = accumulated_main_loss / num_steps
                avg_aux_loss = accumulated_aux_loss / num_steps

                # Gradient clipping and optimizer step
                if self.config.grad_clip != 0.0:
                    self.scaler.unscale_(self.optimizer)
                    try:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip
                        )
                    except RuntimeError as e:
                        if "found no gradients" in str(e):
                            print("Warning: No gradients found during clipping")
                            grad_norm = torch.tensor(0.0, device=self.device)
                        else:
                            raise e
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                # Timing and logging
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                
                if self.iter_num % self.config.log_interval == 0 and master_process:
                    # Log average losses over all micro-steps
                    self.log_training_step(dt, avg_loss, avg_main_loss, avg_aux_loss, lr, local_iter_num, grad_norm)

                # Checkpointing
                if master_process:
                    self.save_checkpoints()

                # Increment counters
                self.iter_num += 1
                local_iter_num += 1

                # Check termination
                if self.iter_num > self.config.max_iters:
                    break

        except KeyboardInterrupt:
            print("Training interrupted by user")
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
        finally:
            # Cleanup
            if ddp:
                destroy_process_group()
            
            # Save final model
            if master_process and self.config.init_from == 'finetune':
                self.save_final_model()

    def log_training_step(self, dt, loss, main_loss, aux_loss, lr, local_iter_num, grad_norm):
        """Log training step metrics."""
        try:
            lossf = loss.item() * self.config.adjusted_gradient_accumulation_steps
            
            if local_iter_num >= 5:  # Let training loop settle
                mfu = self.raw_model.estimate_mfu(
                    self.config.batch_size * self.config.adjusted_gradient_accumulation_steps, dt
                )
                self.running_mfu = mfu if self.running_mfu == -1.0 else 0.9 * self.running_mfu + 0.1 * mfu
            
            # Log MoE statistics and adjust gate bias if using MoE
            if self.config.use_moe:
                self.log_moe_statistics()
                # Adjust gate bias every 1000 iterations for balanced routing
                if self.iter_num % 1000 == 0:
                    for block in self.raw_model.transformer.h:
                        if hasattr(block.mlp, 'adjust_gate_bias'):
                            block.mlp.adjust_gate_bias()
            
            metrics = {
                "iter": self.iter_num,
                "loss": lossf,
                "main_loss": main_loss.item(),
                "aux_loss": aux_loss.item(),
                "mfu": self.running_mfu * 100,
                "lr": lr,
                "grad_norm": grad_norm,
                "time_ms": dt * 1000
            }
            
            # Print metrics
            print(
                f"iter {self.iter_num}: "
                f"loss {lossf:.4f}, "
                f"grad_norm {grad_norm:.2f}, "
                f"time {dt*1000:.2f}ms, "
                f"mfu {self.running_mfu*100:.2f}%"
            )
            
            # Log to wandb if enabled
            if self.wandb_enabled:
                import wandb
                wandb.log(metrics)
                
        except Exception as e:
            print(f"Error in logging step {self.iter_num}: {str(e)}")

    def log_moe_statistics(self):
        """Log Mixture of Experts usage statistics."""
        try:
            for block in self.raw_model.transformer.h:
                if hasattr(block.mlp, 'get_usage_percentages'):
                    usage_percentages = block.mlp.get_usage_percentages()
                    usage_str = ", ".join([
                        f"Expert {i}: {usage_percentages[i]:.2f}%"
                        for i in range(len(usage_percentages))
                    ])
                    print(f"Block MoE usage: {usage_str}")
                    block.mlp.reset_usage_counts()
        except Exception as e:
            print(f"Error logging MoE statistics: {str(e)}")

    def should_save_checkpoint(self) -> bool:
        """Determine if checkpoint should be saved at current iteration."""
        if not hasattr(self.config, 'save_interval'):
            return True  # Default to saving every time if interval not specified
            
        return (
            self.iter_num % self.config.save_interval == 0 or
            self.iter_num == self.config.max_iters
        )
    
    def save_checkpoints(self):
        """Save model checkpoints."""
        if not self.should_save_checkpoint():
            return
            
        try:
            if self.config.save_lora_only and self.config.init_from == 'finetune':
                self.model.save_pretrained(self.config.lora_ckpt_path)
                print(f"Saved LoRA parameters to {self.config.lora_ckpt_path}")
            else:
                checkpoint = {
                    'model': self.raw_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'model_args': self.model_args,
                    'iter_num': self.iter_num,
                    'best_val_loss': self.best_val_loss,
                    'config': OmegaConf.to_container(self.config),
                    'grad_norm': self.last_grad_norm
                }
                
                # Save checkpoint
                ckpt_path = os.path.join(
                    self.config.out_dir,
                    f'checkpoint_{self.iter_num}.pt'
                )
                torch.save(checkpoint, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")
                
        except Exception as e:
            print(f"Error saving checkpoints: {str(e)}")

    def save_final_model(self):
        """Save the final model after training."""
        try:
            if not self.config.save_lora_only:
                print("Merging LoRA parameters into base model...")
                model = self.model.merge_and_unload()
                torch.save(
                    model.state_dict(),
                    self.config.merged_model_ckpt_path
                )
                print(f"Saved merged model to {self.config.merged_model_ckpt_path}")
        except Exception as e:
            print(f"Error saving final model: {str(e)}")

def get_lr(it: int) -> float:
    """
    Compute learning rate with warmup and cosine decay.
    
    Args:
        it: Current iteration number
        
    Returns:
        float: Learning rate value
    """
    # Linear warmup
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    # After decay
    if it > config.lr_decay_iters:
        return config.min_lr
    # Cosine decay
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

def log_metrics(iter_num: int, losses: dict, lr: float, mfu: float):
    """Log metrics to wandb."""
    try:
        import wandb
        wandb.log({
            "iter": iter_num,
            "train/loss": losses['train']['loss'],
            "val/loss": losses['val']['loss'],
            "train/perplexity": losses['train']['perplexity'],
            "val/perplexity": losses['val']['perplexity'],
            "lr": lr,
            "mfu": mfu * 100,
        })
    except Exception as e:
        print(f"Error logging to wandb: {str(e)}")

def initialize_model():
    """Initialize the model with proper error handling."""
    try:
        # Attempt to derive vocab_size from the dataset
        meta_path = os.path.join('data', config.dataset, 'meta.pkl')
        meta_vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            meta_vocab_size = meta['vocab_size']
            print(f"Found vocab_size = {meta_vocab_size} (inside {meta_path})")

        # Model arguments
        model_args = dict(
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            block_size=config.block_size,
            bias=config.bias,
            vocab_size=None,
            dropout=config.dropout,
            use_moe=config.use_moe,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            static_experts=config.static_experts,
            moe_loss=config.moe_loss,
            moe_loss_type=config.moe_loss_type,
            moe_loss_coef=config.moe_loss_coef,
        )

        # Initialize the model based on config.init_from
        if config.init_from == 'scratch':
            print("Initializing a new model from scratch")
            if meta_vocab_size is None:
                print("Defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
            model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
            
        elif config.init_from == 'resume':
            print(f"Resuming training from {config.out_dir}")
            ckpt_path = os.path.join(config.out_dir, 'current_ckpt.pt')
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")
                
            checkpoint = torch.load(ckpt_path, map_location=device)
            checkpoint_model_args = checkpoint['model_args']
            
            # Ensure model configurations match
            for k in model_args:
                model_args[k] = checkpoint_model_args.get(k, model_args.get(k))
            
            # Create the model
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
            
            # Load state dict
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k in list(state_dict.keys()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            
        elif config.init_from == 'finetune':
            print("Loading pre-trained model for fine-tuning with LoRA")
            if not os.path.exists(config.finetune_ckpt_path):
                raise FileNotFoundError(f"Checkpoint file not found at {config.finetune_ckpt_path}")
                
            checkpoint = torch.load(config.finetune_ckpt_path, map_location=device)
            checkpoint_model_args = checkpoint['model_args']
            
            # Update model_args with checkpoint values
            for k in model_args:
                model_args[k] = checkpoint_model_args.get(k, model_args.get(k))
            
            # Create and load the model
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
            state_dict = checkpoint['model']
            
            # Fix state dict keys if needed
            unwanted_prefix = '_orig_mod.'
            for k in list(state_dict.keys()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            
            print("Applying LoRA to the model")
            try:
                from peft import get_peft_model, LoraConfig, TaskType
            except ImportError:
                raise ImportError("To use LoRA, please install the 'peft' library")
            
            # Define and apply LoRA configuration
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=list(config.lora_target_modules),
                lora_dropout=config.lora_dropout
            )
            model = get_peft_model(model, lora_config)
            print("LoRA has been applied to the model")
            
        else:
            raise ValueError(f"Unknown init_from option: {config.init_from}")

        # Adjust block size if necessary
        if config.block_size < model.config.block_size:
            model.crop_block_size(config.block_size)
            model_args['block_size'] = config.block_size

        # Move model to device
        model.to(device)

        # Compile model if requested
        if config.compile:
            print("Compiling the model... (this may take some time)")
            model = torch.compile(model)

        # Wrap model in DDP if needed
        if ddp:
            model = DDP(model, device_ids=[ddp_local_rank])

        return model

    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        raise

def initialize_optimizer(model):
    """Initialize the optimizer with proper error handling."""
    try:
        if config.init_from == 'finetune':
            # Only optimize LoRA parameters
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                weight_decay=config.weight_decay
            )
            print("Optimizer is set to only update LoRA parameters")
        else:
            # Get the raw model if using DDP
            raw_model = model.module if isinstance(model, DDP) else model
            optimizer = raw_model.configure_optimizers(
                config.weight_decay,
                config.learning_rate,
                (config.beta1, config.beta2),
                device_type
            )
            
            # Load optimizer state if resuming
            if config.init_from == 'resume':
                ckpt_path = os.path.join(config.out_dir, 'current_ckpt.pt')
                checkpoint = torch.load(ckpt_path, map_location=device)
                optimizer.load_state_dict(checkpoint['optimizer'])
                
        return optimizer
        
    except Exception as e:
        print(f"Error initializing optimizer: {str(e)}")
        raise

def main():
    """Main training script with improved error handling."""
    try:
        # Initialize wandb if enabled
        wandb_enabled = config.wandb_log and master_process
        if wandb_enabled:
            import wandb
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=OmegaConf.to_container(config)
            )
        
        # Initialize data loader
        data_loader = DataLoader(
            os.path.join('data', config.dataset),
            config,
            device,
            device_type
        )
        
        # Initialize model and optimizer
        model = initialize_model()
        optimizer = initialize_optimizer(model)
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            config=config,
            data_loader=data_loader,
            wandb_enabled=wandb_enabled
        )
        
        # Start training
        trainer.train()
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise
    finally:
        if wandb_enabled:
            wandb.finish()

if __name__ == '__main__':
    main()
