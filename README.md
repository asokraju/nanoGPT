# GPT Language Model with Mixture of Experts (MoE)

This repository contains an implementation of a GPT (Generative Pre-trained Transformer) language model with Mixture of Experts (MoE) integration. The implementation includes advanced features like distributed training, gradient checkpointing, and LoRA fine-tuning support.

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Key Components](#key-components)
- [Model Configuration](#model-configuration)
- [Training](#training)
- [Advanced Features](#advanced-features)
- [Usage Examples](#usage-examples)

## Architecture Overview

The model implements a GPT architecture with the following key features:

- Transformer-based architecture with multi-head self-attention
- Optional Mixture of Experts (MoE) integration in the feed-forward network
- Support for both regular MLP and MoE layers
- Flash Attention support for improved performance
- LayerNorm with optional bias
- Configurable model size and hyperparameters

## Key Components

### Model Components (`model_torch.py`)

1. **LayerNorm**: Custom implementation with optional bias parameter
```python
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        # Normalizes input with optional bias parameter
```

2. **CausalSelfAttention**: Multi-head self-attention module
```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        # Implements causal self-attention with optional Flash Attention
```

3. **MoE (Mixture of Experts)**: Advanced MLP implementation with expert routing
```python
class MoE(nn.Module):
    def __init__(self, config):
        # Implements Mixture of Experts with dynamic and static experts
```

4. **Block**: Transformer block combining attention and feed-forward layers
```python
class Block(nn.Module):
    def __init__(self, config):
        # Combines LayerNorm, Self-Attention, and MLP/MoE
```

### Training Components (`train_torch.py`)

1. **DataLoader**: Handles efficient data loading and preprocessing
2. **Trainer**: Manages the training loop and evaluation
3. **Distributed Training Support**: Implements DDP for multi-GPU training
4. **Learning Rate Scheduling**: Implements warmup and cosine decay

## Model Configuration

The model can be configured using the following key parameters:

```python
@dataclass
class GPTConfig:
    block_size: int = 1024          # Maximum sequence length
    vocab_size: int = 50304         # Vocabulary size
    n_layer: int = 12              # Number of transformer blocks
    n_head: int = 12               # Number of attention heads
    n_embd: int = 768             # Embedding dimension
    dropout: float = 0.0           # Dropout rate
    bias: bool = True              # Use bias in layers
    use_moe: bool = True           # Enable Mixture of Experts
    num_experts: int = 4           # Total number of experts
    num_experts_per_tok: int = 2   # Experts per token
    static_experts: int = 0        # Number of static experts
```

## Training

### Basic Training

1. Prepare your configuration file (config_torch.yaml)
2. Run training:
```bash
python train_torch.py
```

### Distributed Training

For multi-GPU training:
```bash
torchrun --standalone --nproc_per_node=N train_torch.py
```
where N is the number of GPUs.

## Advanced Features

### 1. Mixture of Experts (MoE)

The MoE implementation includes:
- Dynamic expert routing
- Static experts option
- Load balancing with different loss types:
  - Variance penalty
  - Entropy regularization
  - Diversity regularization

### 2. Training Optimizations

- Gradient checkpointing for memory efficiency
- Flash Attention support for faster attention computation
- Mixed precision training
- Learning rate scheduling with warmup and decay

### 3. LoRA Fine-tuning

Supports efficient fine-tuning using LoRA (Low-Rank Adaptation):
```python
# Configure LoRA parameters
lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    target_modules=config.lora_target_modules,
    lora_dropout=config.lora_dropout
)
```

## Usage Examples

### 1. Training from Scratch

```python
# Initialize configuration
config = GPTConfig(
    n_layer=12,
    n_head=12,
    n_embd=768,
    block_size=1024,
    use_moe=True,
    num_experts=4
)

# Create model
model = GPT(config)

# Initialize trainer
trainer = Trainer(model, optimizer, config, data_loader)
trainer.train()
```

### 2. Fine-tuning with LoRA

```python
# Load pre-trained model
model = GPT.from_pretrained('gpt2')

# Apply LoRA
lora_config = LoraConfig(...)
model = get_peft_model(model, lora_config)

# Fine-tune
trainer = Trainer(model, optimizer, config, data_loader)
trainer.train()
```

### 3. Generating Text

```python
model.eval()
with torch.no_grad():
    output = model.generate(
        idx=context,
        max_new_tokens=100,
        temperature=0.8,
        top_k=40
    )
```

## Requirements

- PyTorch >= 2.0 (for Flash Attention support)
- transformers (for pretrained model support)
- peft (for LoRA fine-tuning)
- omegaconf (for configuration management)
- wandb (optional, for experiment tracking)

## Notes

- Flash Attention requires PyTorch >= 2.0
- For best performance with MoE, adjust num_experts and num_experts_per_tok based on your task
- Monitor expert utilization during training to ensure balanced routing
- Use gradient checkpointing for large models to reduce memory usage
