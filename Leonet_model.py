import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpansionContractionBlock(nn.Module):
    """
    Applies expansion–contraction logic to a hidden representation.
    Outputs both motor projection and updated hidden state.
    """
    def __init__(self, hidden_dim, expansion_factor=4, motor_dim=128):
        super().__init__()
        expanded_dim = hidden_dim * expansion_factor
        self.expand = nn.Linear(hidden_dim, expanded_dim)
        self.motor_proj = nn.Linear(expanded_dim, motor_dim)
        self.cognitive_proj = nn.Linear(expanded_dim, hidden_dim)
        self.contract = nn.Linear(hidden_dim + motor_dim, hidden_dim)

    def forward(self, x):
        x_expanded = self.expand(x)
        motor_out = self.motor_proj(x_expanded)
        cognitive_out = self.cognitive_proj(x_expanded)
        merged = torch.cat([cognitive_out, motor_out], dim=-1)
        x_contracted = self.contract(merged)
        return x_contracted, motor_out


class LeoNetBlock(nn.Module):
    """
    Single transformer block with attention and expansion–contraction logic.
    Outputs hidden state and motor vector.
    """
    def __init__(self, hidden_dim, num_heads, expansion_factor=4, motor_dim=128):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ecb = ExpansionContractionBlock(hidden_dim, expansion_factor, motor_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ecb_out, motor_out = self.ecb(x)
        x = self.norm2(x + ecb_out)
        return x, motor_out


class LeoNet(nn.Module):
    """
    LeoNet architecture with dual heads for cognitive and motor output.
    Includes attention, expansion–contraction, and layer-wise motor forking.
    """
    def __init__(self, vocab_size, hidden_dim=512, num_layers=6, num_heads=8, motor_dim=128, max_seq_len=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))

        self.transformer_blocks = nn.ModuleList([
            LeoNetBlock(hidden_dim, num_heads, motor_dim=motor_dim)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.language_head = nn.Linear(hidden_dim, vocab_size)

        # Layer-wise motor heads (optional per-layer extraction)
        self.motor_heads = nn.ModuleList([
            nn.Linear(motor_dim, motor_dim) for _ in range(num_layers)
        ])

        # Final motor output head (compress into final 1024-d flattened vector)
        self.motor_output_head = nn.Sequential(
            nn.Linear(motor_dim * num_layers, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024)  # Final 8x128 flat output
        )

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x + self.pos_embedding[:, :x.size(1), :]

        motor_outputs = []

        for block, motor_head in zip(self.transformer_blocks, self.motor_heads):
            x, motor_out = block(x)
            motor_outputs.append(motor_head(motor_out))  # shape: [B, T, motor_dim]

        x = self.final_norm(x)
        logits = self.language_head(x)

        # Merge all layerwise motor outputs into a single vector
        # Shape: [B, T, L, D] → concatenate over L → [B, T, L*D] → flatten over T
        motor_concat = torch.cat(motor_outputs, dim=-1)  # [B, T, L*D]
        motor_flat = motor_concat.mean(dim=1)  # Global average pooling over sequence
        motor_final = self.motor_output_head(motor_flat)  # [B, 1024]

        return logits, motor_final


# Optional Debug: Gradient Checker
def print_missing_gradients(model):
    print("Checking gradients...")
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            print(f"⚠️ No gradient for: {name}")
