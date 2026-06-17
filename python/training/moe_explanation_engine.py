"""MoE-enhanced Explanation Engine using Tutel.

Replaces dense MLP with Mixture-of-Experts for scalable generation.
"""

import torch
import torch.nn as nn
from tutel import moe as tutel_moe


class MoEExplanationEngine(nn.Module):
    """ExplanationEngine with Tutel Mixture-of-Experts.
    
    Architecture:
    - Text encoder
    - MoE layers (replace dense MLPs)
    - Output head
    
    Benefits over dense:
    - Sparse activation: only k experts per token
    - Scalable: add experts without dense parameters
    - Specialized: each expert learns different explanation styles
    """
    
    def __init__(
        self,
        vocab_size: int = 8192,
        embed_dim: int = 384,
        num_layers: int = 6,
        num_experts: int = 8,
        experts_per_token: int = 2,
        hidden_dim: int = 1536,
    ):
        super().__init__()
        
        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(256, embed_dim)  # 256 context length
        
        # Transformer blocks with MoE
        self.layers = nn.ModuleList([
            TransformerBlockWithMoE(
                embed_dim=embed_dim,
                num_heads=6,
                num_experts=num_experts,
                experts_per_token=experts_per_token,
                hidden_dim=hidden_dim,
            )
            for _ in range(num_layers)
        ])
        
        self.ln_final = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Weight tying
        self.token_embed.weight = self.lm_head.weight
    
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """Forward pass.
        
        Args:
            idx: (batch, seq_len) token indices
            targets: (batch, seq_len) target tokens for loss
            
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar loss (if targets provided)
            aux_loss: auxiliary load balancing loss
        """
        B, T = idx.shape
        
        # Embeddings
        token_emb = self.token_embed(idx)  # (B, T, embed_dim)
        pos_ids = torch.arange(T, device=idx.device)
        pos_emb = self.pos_embed(pos_ids)  # (T, embed_dim)
        x = token_emb + pos_emb  # (B, T, embed_dim)
        
        # Transformer layers with MoE
        aux_loss = 0.0
        for layer in self.layers:
            x, layer_aux_loss = layer(x)
            aux_loss = aux_loss + layer_aux_loss
        
        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )
            # Add auxiliary loss for MoE load balancing
            loss = loss + 0.01 * aux_loss
        
        return logits, loss


class TransformerBlockWithMoE(nn.Module):
    """Transformer block with MoE in FFN layer."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_experts: int,
        experts_per_token: int,
        hidden_dim: int,
    ):
        super().__init__()
        
        # Self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        
        # MoE layer (replaces dense FFN)
        self.moe = tutel_moe.moe_layer(
            gate_type={'type': 'top', 'k': experts_per_token},
            model_dim=embed_dim,
            experts={
                'num_experts_per_device': num_experts,
                'type': 'ffn',
                'hidden_size_per_expert': hidden_dim // num_experts,
                'activation_fn': lambda x: torch.nn.functional.gelu(x),
            },
        )
        self.ln2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor):
        """Forward pass with residual connections.
        
        Returns:
            x: (batch, seq_len, embed_dim)
            aux_loss: scalar auxiliary loss for MoE balancing
        """
        # Self-attention block
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.ln1(x)
        
        # MoE block
        moe_out = self.moe(x)
        
        # Extract output and auxiliary loss
        if isinstance(moe_out, tuple):
            moe_out, aux_loss = moe_out
        else:
            moe_out = moe_out
            aux_loss = 0.0
        
        x = x + moe_out
        x = self.ln2(x)
        
        return x, aux_loss


class DistributedMoEExplanationEngine(nn.Module):
    """Distributed version for multi-GPU training.
    
    Uses torch.distributed.run for multi-GPU support.
    Works with both NVIDIA CUDA and AMD ROCm.
    """
    
    def __init__(self, config):
        super().__init__()
        self.model = MoEExplanationEngine(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            num_layers=config.num_layers,
            num_experts=config.num_experts,
            experts_per_token=config.experts_per_token,
            hidden_dim=config.hidden_dim,
        )
        
        # Wrap with DDP for distributed training
        import torch.distributed as dist
        if dist.is_initialized():
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
    
    def forward(self, idx, targets=None):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.model.module(idx, targets)
        return self.model(idx, targets)


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MoEExplanationEngine(
        vocab_size=8192,
        embed_dim=384,
        num_layers=6,
        num_experts=8,
        experts_per_token=2,
        hidden_dim=1536,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 64
    
    idx = torch.randint(0, 8192, (batch_size, seq_len)).to(device)
    targets = torch.randint(0, 8192, (batch_size, seq_len)).to(device)
    
    logits, loss = model(idx, targets)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    print("Backward pass successful!")
