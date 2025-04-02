import math
import os
import time
import pickle
import inspect
import re
from contextlib import nullcontext
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import numpy as np
from omegaconf import OmegaConf

# Try importing PEFT for LoRA. If not installed, we'll skip it.
try:
    from peft import get_peft_model, LoraConfig, merge_lora_weights
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# ---------------------------------------------------------------------
# Rotary Embedding Utilities
# ---------------------------------------------------------------------

def build_rope_cache(seq_len, dim, base=10000):
    """
    Builds cos/sin rotation buffers for Rotary Position Embeddings.
    seq_len: maximum sequence length (e.g. config.block_size)
    dim: head_dim (embedding_dim / n_head)
    base: base frequency, e.g. 10000
    returns: cos, sin of shape (seq_len, dim)
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq_len, dim/2)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    # Interleave cos/sin for even/odd channels
    cos = cos.unsqueeze(-1).repeat(1, 1, 2).view(seq_len, dim)
    sin = sin.unsqueeze(-1).repeat(1, 1, 2).view(seq_len, dim)
    return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin, seq_len):
    """
    Applies RoPE to q, k in place:
      q, k: (B, n_head, T, head_dim)
      cos, sin: (seq_len, head_dim)
      seq_len: actual sequence length T
    returns: (q_out, k_out)
    """
    cos_t = cos[:seq_len, :].to(q.device, dtype=q.dtype)
    sin_t = sin[:seq_len, :].to(q.device, dtype=q.dtype)

    # Expand to (1, 1, T, head_dim)
    cos_t = cos_t[None, None, :, :]
    sin_t = sin_t[None, None, :, :]

    # q, k shape: (B, n_head, T, head_dim)
    # We'll do a pairwise rotation
    # chunk the last dimension into even/odd
    qcos = q * cos_t
    qsin = q * sin_t
    q_0 = qcos[..., 0::2] - qsin[..., 1::2]
    q_1 = qcos[..., 1::2] + qsin[..., 0::2]

    q_out = torch.zeros_like(q)
    q_out[..., 0::2] = q_0
    q_out[..., 1::2] = q_1

    kcos = k * cos_t
    ksin = k * sin_t
    k_0 = kcos[..., 0::2] - ksin[..., 1::2]
    k_1 = kcos[..., 1::2] + ksin[..., 0::2]

    k_out = torch.zeros_like(k)
    k_out[..., 0::2] = k_0
    k_out[..., 1::2] = k_1

    return q_out, k_out

# ----------------------------
# 1) Model Components
# ----------------------------

class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class MLP(nn.Module):
    """Standard 2-layer MLP: Linear -> GELU -> Linear -> Dropout."""
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class SegmentedMLP(nn.Module):
    """MLP with reduced intermediate dimension for each expert."""
    def __init__(self, config, segment_reduction):
        super().__init__()
        reduced_dim = (4 * config.n_embd) // segment_reduction
        self.c_fc = nn.Linear(config.n_embd, reduced_dim, bias=config.bias)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(reduced_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class MoE(nn.Module):
    """
    Mixture of Experts layer with:
      - shared_experts (always used)
      - dynamic_experts (token-level gating)
      - load balancing losses (expert-level & device-level)
      - optional bias adjustment for gating
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.total_experts = config.num_experts
        self.shared_experts = config.shared_experts
        self.num_dynamic_experts = self.total_experts - self.shared_experts
        self.expert_segmentation = config.expert_segmentation
        self.num_experts_per_tok = config.num_experts_per_tok * self.expert_segmentation

        if self.num_dynamic_experts <= 0:
            raise ValueError("MoE config error: must have at least 1 dynamic expert")
        if self.num_experts_per_tok > self.num_dynamic_experts:
            raise ValueError(
                f"num_experts_per_tok * expert_segmentation = {self.num_experts_per_tok} "
                f" > num_dynamic_experts = {self.num_dynamic_experts}"
            )

        # Shared experts
        if self.shared_experts > 0:
            self.shared_expert_modules = nn.ModuleList([
                SegmentedMLP(config, self.expert_segmentation)
                for _ in range(self.shared_experts)
            ])
        else:
            self.shared_expert_modules = None

        # Dynamic experts
        self.dynamic_experts = nn.ModuleList([
            SegmentedMLP(config, self.expert_segmentation)
            for _ in range(self.num_dynamic_experts)
        ])

        # Gating
        self.gate = nn.Linear(config.n_embd, self.num_dynamic_experts, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(self.num_dynamic_experts))

        # Usage counters (for debugging / stats)
        self.register_buffer("expert_usage_counts", torch.zeros(self.num_dynamic_experts, dtype=torch.long))
        self.total_assignments = 0

    def forward(self, x):
        """
        x: (B, T, C)
        returns: (output, expert_balance_loss, device_balance_loss)
        """
        B, T, C = x.size()

        # Shared experts (process all tokens)
        if self.shared_experts > 0:
            y_shared_list = [expert(x) for expert in self.shared_expert_modules]
            y_shared = torch.stack(y_shared_list, dim=0).sum(dim=0)  # (B, T, C)
        else:
            y_shared = torch.zeros_like(x)

        # Flatten for gating
        x_flat = x.view(-1, C)  # (B*T, C)
        scores = self.gate(x_flat) + self.gate_bias  # shape (B*T, num_dynamic_experts)
        topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        topk_weights = F.softmax(topk_scores, dim=-1).unsqueeze(-1)  # (B*T, k, 1)

        # Update usage counts
        flat_expert_indices = topk_indices.view(-1)
        with torch.no_grad():
            counts = flat_expert_indices.bincount(minlength=self.num_dynamic_experts)
            self.expert_usage_counts += counts
            self.total_assignments += flat_expert_indices.size(0)

        # Dispatch to each expert
        x_repeated = x_flat.unsqueeze(1).expand(-1, self.num_experts_per_tok, -1)
        x_repeated = x_repeated.reshape(-1, C)  # (B*T*k, C)

        expert_inputs = {}
        expert_masks = {}
        for i in range(self.num_dynamic_experts):
            mask = (flat_expert_indices == i)
            if mask.any():
                expert_inputs[i] = x_repeated[mask]
                expert_masks[i] = mask

        y_dynamic = torch.zeros_like(x_repeated, device=x.device, dtype=x.dtype)
        for i, expert in enumerate(self.dynamic_experts):
            if i in expert_inputs:
                out_i = expert(expert_inputs[i])
                y_dynamic[expert_masks[i]] = out_i

        # Combine top-k outputs
        y_dynamic = y_dynamic.view(-1, self.num_experts_per_tok, C)  # (B*T, k, C)
        y_dynamic = (y_dynamic * topk_weights).sum(dim=1)  # (B*T, C)
        y_dynamic = y_dynamic.view(B, T, C)

        # Expert-level balance loss
        expert_probs = torch.zeros(B*T, self.num_dynamic_experts, device=x.device, dtype=x.dtype)
        topk_weights_2d = topk_weights.squeeze(-1)  # (B*T, k)
        expert_probs.scatter_(1, topk_indices, topk_weights_2d)
        mean_prob = expert_probs.mean(dim=0)  # (num_dynamic_experts,)
        target_prob = 1.0 / self.num_dynamic_experts
        expert_balance_loss = ((mean_prob - target_prob)**2).mean()

        # Device-level balance loss
        device_balance_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if hasattr(self.config, 'device_groups') and self.config.device_groups:
            for group in self.config.device_groups:
                group_prob = mean_prob[group].sum()
                group_size = len(group)
                # fraction that group should have
                target_group_prob = group_size / self.num_dynamic_experts
                device_balance_loss += (group_prob - target_group_prob)**2
            device_balance_loss = device_balance_loss / len(self.config.device_groups)

        y_out = y_dynamic + y_shared
        return y_out, expert_balance_loss, device_balance_loss

    def get_usage_percentages(self):
        if self.total_assignments > 0:
            usage_percent = (self.expert_usage_counts.float() / self.total_assignments) * 100.0
            return usage_percent.tolist()
        else:
            return [0.0] * self.num_dynamic_experts

    def reset_usage_counts(self):
        self.expert_usage_counts.zero_()
        self.total_assignments = 0

    def adjust_gate_bias(self, target_usage=None):
        """
        Nudges gate_bias so that overused experts are penalized, underused get a boost.
        """
        if target_usage is None:
            target_usage = 1.0 / self.num_dynamic_experts
        if self.total_assignments == 0:
            return

        current_usage = self.expert_usage_counts.float() / self.total_assignments
        usage_diff = current_usage - target_usage
        adjustment_factor = 0.1
        with torch.no_grad():
            self.gate_bias -= adjustment_factor * usage_diff
        self.reset_usage_counts()

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional RoPE."""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.use_rope = config.use_rope

        # Q, K, V projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Build RoPE cache if needed
        if self.use_rope:
            head_dim = config.n_embd // config.n_head
            cos, sin = build_rope_cache(config.block_size, head_dim, base=getattr(config, "rope_base", 10000))
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        # Flash attention or not
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                    .view(1, 1, config.block_size, config.block_size),
                persistent=False
            )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # If using rotary pos emb, rotate q, k
        if self.use_rope:
            q, k = apply_rotary_pos_emb(q, k, self.rope_cos, self.rope_sin, seq_len=T)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """
    A single Transformer block: LN -> Self-Attn -> LN -> (MoE or MLP).
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)

        if config.use_moe:
            self.mlp = MoE(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x):
        # Self-attention
        x = x + self.attn(self.ln_1(x))

        # MLP or MoE
        if isinstance(self.mlp, MoE):
            mlp_output, exp_loss, dev_loss = self.mlp(self.ln_2(x))
            x = x + mlp_output
            return x, exp_loss, dev_loss
        else:
            mlp_output = self.mlp(self.ln_2(x))
            x = x + mlp_output
            return x, torch.tensor(0.0, device=x.device), torch.tensor(0.0, device=x.device)

# ----------------------------
# 2) Configuration Class
# ----------------------------
from transformers import PretrainedConfig

class GPTConfig(PretrainedConfig):
    """
    GPT-style config with optional use_rope for Rotary Positional Embeddings.
    """
    def __init__(
        self,
        vocab_size=50304,
        n_embd=768,
        n_layer=12,
        n_head=12,
        block_size=1024,
        dropout=0.0,
        bias=True,
        use_moe=False,
        num_experts=4,
        num_experts_per_tok=2,
        expert_segmentation=1,
        shared_experts=0,
        alpha1=0.01,
        alpha2=0.1,
        device_groups=None,
        gradient_checkpointing=False,
        use_rope=False,
        rope_base=10000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.block_size = block_size
        self.dropout = dropout
        self.bias = bias

        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.expert_segmentation = expert_segmentation
        self.shared_experts = shared_experts
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.device_groups = device_groups if device_groups else []
        self.gradient_checkpointing = gradient_checkpointing

        self.use_rope = use_rope
        self.rope_base = rope_base
        self.tie_word_embeddings = True


# ----------------------------
# 3) GPT Model
# ----------------------------
class GPT(nn.Module):
    """
    A GPT Language Model that can optionally:
      - Use MoE for MLP layers
      - Use RoPE for positional embeddings
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.vocab_size is None:
            raise ValueError("Must specify vocab_size in config")
        if config.block_size is None:
            raise ValueError("Must specify block_size in config")

        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing

        # Embeddings: token (wte) + positional (wpe)
        # We'll optionally skip adding wpe if use_rope==True
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': LayerNorm(config.n_embd, bias=config.bias)
        })

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # Weight initialization
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2*config.n_layer))

        total_params = self.get_num_params()
        print(f"GPT model initialized with {total_params/1e6:.2f}M parameters.")

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        # exclude positional embedding if non_embedding is True
        if non_embedding:
            n_params -= self.transformer['wpe'].weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx: (B, T) token indices
        targets: (B, T), optional for training
        returns: (logits, total_loss, main_loss, balance_loss)
        """
        device = idx.device
        b, t = idx.shape
        if t > self.config.block_size:
            raise ValueError(f"Sequence length {t} > block_size {self.config.block_size}")

        # token embedding
        tok_emb = self.transformer['wte'](idx)  # (B, T, n_embd)

        # if using rope, we skip absolute wpe
        if self.config.use_rope:
            x = tok_emb
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.transformer['wpe'](pos)  # (T, n_embd)
            x = tok_emb + pos_emb

        x = self.transformer['drop'](x)

        # forward blocks
        total_expert_balance_loss = torch.tensor(0.0, device=device)
        total_device_balance_loss = torch.tensor(0.0, device=device)

        def block_forward(block, x_inner):
            return block(x_inner)

        for block in self.transformer['h']:
            if self.gradient_checkpointing and self.training:
                x, exp_loss, dev_loss = torch.utils.checkpoint.checkpoint(block_forward, block, x)
            else:
                x, exp_loss, dev_loss = block(x)
            total_expert_balance_loss += exp_loss
            total_device_balance_loss += dev_loss

        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x)

        if targets is not None:
            main_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
            total_loss = main_loss
            if self.config.use_moe:
                total_loss += self.config.alpha1 * total_expert_balance_loss
                total_loss += self.config.alpha2 * total_device_balance_loss
            return logits, total_loss, main_loss, (total_expert_balance_loss + total_device_balance_loss)
        else:
            return logits, None, None, torch.tensor(0.0, device=device)

    def crop_block_size(self, block_size):
        """
        Optionally reduce the block size, if needed.
        """
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer['wpe'].weight = nn.Parameter(
            self.transformer['wpe'].weight[:block_size]
        )
        for block in self.transformer['h']:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        Autoregressive generation (greedy or top-k).
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _, _, _ = self(idx_cond, targets=None)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        Estimate model FLOPS utilization (MFU) in units of A100 bfloat16 peak FLOPS.
        """
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter / dt
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops ~312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

# ----------------------------
# 4) Training Script
# ----------------------------

def setup_distributed(config):
    """
    Setup DDP if environment variables indicate distributed run.
    Returns: (ddp, device, master_process, seed_offset, world_size, local_rank)
    """
    rank = os.environ.get('RANK')
    if rank is None:
        ddp = False
        master_process = True
        seed_offset = 0
        world_size = 1
        local_rank = 0
        device = config.device
        config.adjusted_gradient_accumulation_steps = config.gradient_accumulation_steps
    else:
        ddp = True
        ddp_rank = int(rank)
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])

        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        init_process_group(backend=config.backend)
        master_process = (ddp_rank == 0)
        seed_offset = ddp_rank
        if config.gradient_accumulation_steps % ddp_world_size != 0:
            raise ValueError("gradient_accumulation_steps must be divisible by WORLD_SIZE")
        config.adjusted_gradient_accumulation_steps = config.gradient_accumulation_steps // ddp_world_size

        world_size = ddp_world_size
        local_rank = ddp_local_rank
    return ddp, device, master_process, seed_offset, world_size, local_rank

class DataLoader:
    """
    Basic memmapped data loader. 
    Expects 'train.bin' and 'val.bin' in data_dir.
    """
    def __init__(self, data_dir, config, device, device_type):
        self.config = config
        self.device = device
        self.device_type = device_type

        self.data_maps = {}
        for split in ['train', 'val']:
            path = os.path.join(data_dir, f"{split}.bin")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file not found: {path}")
            self.data_maps[split] = np.memmap(path, dtype=np.uint16, mode='r')

    def get_batch(self, split):
        data = self.data_maps[split]
        block_size = self.config.block_size
        batch_size = self.config.batch_size
        if len(data) <= block_size:
            raise ValueError(f"Data too small for block_size={block_size}")

        max_idx = len(data) - block_size
        ix = torch.randint(max_idx, (batch_size,))
        x_list, y_list = [], []
        for i in ix:
            i = i.item()
            x_list.append(torch.from_numpy(data[i:i+block_size].astype(np.int64)))
            y_list.append(torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)))

        x = torch.stack(x_list)
        y = torch.stack(y_list)
        if self.device_type == 'cuda':
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y

class Trainer:
    def __init__(self, model, optimizer, config, data_loader, master_process, ddp):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.data_loader = data_loader
        self.master_process = master_process
        self.ddp = ddp
        self.scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16'))

        # If DDP, raw_model is model.module
        self.raw_model = model.module if isinstance(model, DDP) else model

        self.running_mfu = -1.0
        self.iter_num = 0
        self.best_val_loss = float('inf')
        self.last_grad_norm = 0.0

        self.wandb_enabled = config.get('wandb_log', False)

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        eval_iters = self.config.eval_iters
        for split in ['train', 'val']:
            losses = []
            main_losses = []
            aux_losses = []
            for _ in range(eval_iters):
                X, Y = self.data_loader.get_batch(split)
                with torch.no_grad():
                    logits, total_loss, main_loss, aux_loss = self.model(X, Y)
                if total_loss is None:
                    continue
                losses.append(total_loss.item())
                main_losses.append(main_loss.item())
                aux_losses.append(aux_loss.item())
            if len(losses) == 0:
                continue
            mean_loss = float(np.mean(losses))
            mean_main = float(np.mean(main_losses))
            mean_aux = float(np.mean(aux_losses))
            out[split] = {
                'loss': mean_loss,
                'main_loss': mean_main,
                'aux_loss': mean_aux,
                'perplexity': math.exp(mean_loss) if mean_loss < 20 else float('inf')
            }
        self.model.train()
        return out

    def train_step(self, X, Y):
        with torch.autocast(device_type=self.config.device_type, dtype=self.config.ptdtype):
            logits, total_loss, main_loss, aux_loss = self.model(X, Y)
            # gradient accumulation
            total_loss = total_loss / self.config.adjusted_gradient_accumulation_steps

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)

        # measure grad norm
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item()**2
        grad_norm = math.sqrt(total_norm)
        self.last_grad_norm = grad_norm

        return total_loss, main_loss, aux_loss, grad_norm

    def log_moe_statistics(self):
        for block in self.raw_model.transformer['h']:
            if hasattr(block.mlp, "get_usage_percentages"):
                usage = block.mlp.get_usage_percentages()
                usage_str = ", ".join([f"E{i}: {pct:.1f}%" for i, pct in enumerate(usage)])
                print(f" MoE usage -> {usage_str}")
                block.mlp.reset_usage_counts()

    def train(self):
        print("Begin training")
        t0 = time.time()
        local_iter_num = 0
        X, Y = self.data_loader.get_batch('train')

        try:
            while True:
                # LR schedule
                lr = get_lr(self.iter_num, self.config)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

                # Evaluate
                if self.iter_num % self.config.eval_interval == 0 and self.master_process:
                    metrics = self.estimate_loss()
                    print(f"[Step {self.iter_num}]")
                    for split, mm in metrics.items():
                        print(f"  {split}: loss {mm['loss']:.4f}, ppl {mm['perplexity']:.2f}")
                    # If using wandb, log here

                # Grad Accum
                accumulated_loss = 0.0
                accumulated_main = 0.0
                accumulated_aux = 0.0
                micro_steps_used = 0
                for micro_step in range(self.config.adjusted_gradient_accumulation_steps):
                    if self.ddp:
                        self.model.require_backward_grad_sync = (
                            micro_step == (self.config.adjusted_gradient_accumulation_steps - 1)
                        )
                    loss, main_loss, aux_loss, grad_norm = self.train_step(X, Y)
                    accumulated_loss += loss.item()
                    accumulated_main += main_loss.item()
                    accumulated_aux += aux_loss.item()
                    micro_steps_used += 1

                    # Next batch
                    X, Y = self.data_loader.get_batch('train')

                # Clip grad
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                dt = time.time() - t0
                t0 = time.time()

                # MoE gate bias adjust every 1000 steps
                if self.config.use_moe and (self.iter_num % 1000 == 0) and self.iter_num > 0:
                    for block in self.raw_model.transformer['h']:
                        if hasattr(block.mlp, 'adjust_gate_bias'):
                            block.mlp.adjust_gate_bias()

                # log step
                if self.iter_num % self.config.log_interval == 0 and self.master_process:
                    if local_iter_num >= 5:
                        # MFU estimate
                        tokens_per_iter = (self.config.batch_size *
                                           self.config.adjusted_gradient_accumulation_steps *
                                           self.config.block_size)
                        mfu_est = self.raw_model.estimate_mfu(tokens_per_iter, dt)
                        self.running_mfu = (mfu_est if self.running_mfu < 0 else
                                            0.9*self.running_mfu + 0.1*mfu_est)
                    avg_loss = accumulated_loss / micro_steps_used
                    avg_main = accumulated_main / micro_steps_used
                    avg_aux = accumulated_aux / micro_steps_used
                    print(f"iter {self.iter_num}: loss {avg_loss:.4f}, main {avg_main:.4f}, aux {avg_aux:.4f}, "
                          f"grad_norm {grad_norm:.4f}, time {dt*1000:.2f} ms, mfu {self.running_mfu*100:.2f}%")

                    if self.config.use_moe:
                        self.log_moe_statistics()

                # Checkpoint
                if self.master_process and (self.iter_num % self.config.save_interval == 0 or
                                            self.iter_num == self.config.max_iters):
                    self.save_checkpoint()

                self.iter_num += 1
                local_iter_num += 1
                if self.iter_num > self.config.max_iters:
                    break

        except KeyboardInterrupt:
            print("Training interrupted.")
        finally:
            if self.ddp:
                destroy_process_group()
            if self.master_process and self.config.init_from == "finetune":
                if PEFT_AVAILABLE and not self.config.save_lora_only:
                    print("Merging LoRA weights into base model ...")
                    raw = self.model.module if isinstance(self.model, DDP) else self.model
                    merge_lora_weights(raw)
                    ckpt_path = os.path.join(self.config.out_dir, 'merged_model.pt')
                    torch.save(raw.state_dict(), ckpt_path)
                    print(f"Saved merged model to {ckpt_path}")

    def save_checkpoint(self):
        # Save a standard checkpoint
        ckpt_path = os.path.join(self.config.out_dir, f"checkpoint_{self.iter_num}.pt")
        checkpoint = {
            'model': self.raw_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_num': self.iter_num,
            'best_val_loss': self.best_val_loss,
            'config': OmegaConf.to_container(self.config),
        }
        torch.save(checkpoint, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

def get_lr(it, config):
    """
    Cosine LR schedule with warmup
    """
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

def main():
    # Load config
    try:
        config = OmegaConf.load('config_torch.yaml')
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {e}")

    ddp, device, master_process, seed_offset, world_size, local_rank = setup_distributed(config)
    torch.manual_seed(1337 + seed_offset)

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    config.device_type = device_type
    if not hasattr(config, "dtype"):
        config.dtype = "float32"
    ptdtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16
    }
    config.ptdtype = ptdtype_map[config.dtype]

    if master_process:
        os.makedirs(config.out_dir, exist_ok=True)

    # Initialize data loader
    data_dir = os.path.join("data", config.dataset)
    data_loader = DataLoader(data_dir, config, device, device_type)

    # Initialize or resume model
    model = initialize_model(config, device)

    # Wrap in DDP if needed
    if ddp:
        model = DDP(model, device_ids=[local_rank])

    # Initialize optimizer
    optimizer = initialize_optimizer(model, config, device)

    # Create trainer
    trainer = Trainer(model, optimizer, config, data_loader, master_process, ddp)
    trainer.train()

def initialize_model(config, device):
    """
    Creates GPT model from scratch or loads from checkpoint, or loads for finetune + LoRA.
    """
    meta_path = os.path.join("data", config.dataset, "meta.pkl")
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta.get("vocab_size", None)
        print(f"Found vocab_size={meta_vocab_size} in {meta_path}")

    model_args = dict(
        vocab_size= meta_vocab_size if meta_vocab_size else 50304,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        block_size=config.block_size,
        dropout=config.dropout,
        bias=config.bias,
        use_moe=config.use_moe,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        expert_segmentation=config.expert_segmentation,
        shared_experts=config.shared_experts,
        alpha1=config.alpha1,
        alpha2=config.alpha2,
        device_groups=config.device_groups,
        gradient_checkpointing=config.gradient_checkpointing,
        use_rope=config.get("use_rope", False),
        rope_base=config.get("rope_base", 10000)
    )

    init_from = config.init_from
    if init_from == "scratch":
        print("Initializing a new model from scratch...")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        # Possibly crop block_size
        if config.block_size < gptconf.block_size:
            model.crop_block_size(config.block_size)
        model.to(device)
        return model

    elif init_from == "resume":
        # We expect a checkpoint in config.out_dir
        ckpts = [ck for ck in os.listdir(config.out_dir) if ck.startswith("checkpoint_") and ck.endswith(".pt")]
        if not ckpts:
            raise FileNotFoundError("No checkpoint_* files found for resume!")
        latest_ckpt = sorted(ckpts, key=lambda x: int(re.findall(r'\d+', x)[0]))[-1]
        ckpt_path = os.path.join(config.out_dir, latest_ckpt)
        print(f"Resuming training from {ckpt_path} ...")
        checkpoint = torch.load(ckpt_path, map_location=device)

        loaded_conf = checkpoint["config"]
        for k,v in model_args.items():
            loaded_conf[k] = v
        gptconf = GPTConfig(**loaded_conf)
        model = GPT(gptconf)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        return model

    elif init_from == "finetune":
        if not hasattr(config, 'finetune_ckpt_path'):
            raise ValueError("No 'finetune_ckpt_path' provided in config for finetune mode.")

        ckpt_path = config.finetune_ckpt_path
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"finetune_ckpt_path not found: {ckpt_path}")

        print(f"Loading pretrained checkpoint for finetuning: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        loaded_conf = checkpoint["config"]
        for k,v in model_args.items():
            loaded_conf[k] = v
        gptconf = GPTConfig(**loaded_conf)
        model = GPT(gptconf)
        model.load_state_dict(checkpoint["model"], strict=False)
        model.to(device)

        if PEFT_AVAILABLE:
            print("Applying LoRA to the model ...")
            lora_conf = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=list(config.lora_target_modules) if hasattr(config, 'lora_target_modules') else None
            )
            model = get_peft_model(model, lora_conf)
            print("LoRA successfully applied.")
        else:
            print("Warning: PEFT not installed, cannot apply LoRA layers.")

        return model
    else:
        raise ValueError(f"Unknown init_from: {init_from}")

def initialize_optimizer(model, config, device):
    if config.init_from == 'finetune' and config.get('save_lora_only', False) and PEFT_AVAILABLE:
        print("Optimizer: Only optimizing LoRA parameters.")
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        return optimizer

    raw_model = model.module if isinstance(model, DDP) else model
    param_dict = {pn: p for pn, p in raw_model.named_parameters() if p.requires_grad}
    decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = (fused_available and config.device_type == 'cuda')
    extra_args = dict(fused=True) if use_fused else {}
    optimizer = torch.optim.AdamW(optim_groups,
                                  lr=config.learning_rate,
                                  betas=(config.beta1, config.beta2),
                                  **extra_args)
    return optimizer

if __name__ == "__main__":
    main()
