"""
Task 3: Transformer-Based Music Generator.

Autoregressive Transformer decoder for long coherent music sequence generation.

    p(X) = prod_t p(x_t | x_{<t})
    L_TR = - sum_t log p_theta(x_t | x_{<t})
    Perplexity = exp(L_TR / T)

With optional genre conditioning: h_t = Emb(x_t) + Emb(genre)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    VOCAB_SIZE, TF_D_MODEL, TF_N_HEADS, TF_NUM_LAYERS, TF_D_FF,
    TF_DROPOUT, TF_MAX_SEQ_LEN, NUM_GENRES, PAD_TOKEN, DEVICE,
    TF_TEMPERATURE, TF_TOP_K,
)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int = TF_D_MODEL, max_len: int = TF_MAX_SEQ_LEN,
                 dropout: float = TF_DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerDecoderBlock(nn.Module):
    """Single Transformer decoder block with causal self-attention."""

    def __init__(self, d_model: int = TF_D_MODEL, n_heads: int = TF_N_HEADS,
                 d_ff: int = TF_D_FF, dropout: float = TF_DROPOUT):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Causal self-attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask,
                                     key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        # Feed-forward
        x = self.norm2(x + self.ff(x))
        return x


class MusicTransformer(nn.Module):
    """Transformer decoder for autoregressive music generation (Task 3).

    p(X) = prod_t p(x_t | x_{<t})

    Supports optional genre conditioning via additive genre embeddings.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = TF_D_MODEL,
                 n_heads: int = TF_N_HEADS, num_layers: int = TF_NUM_LAYERS,
                 d_ff: int = TF_D_FF, max_seq_len: int = TF_MAX_SEQ_LEN,
                 num_genres: int = NUM_GENRES, dropout: float = TF_DROPOUT,
                 pad_token: int = PAD_TOKEN):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token)
        self.genre_embedding = nn.Embedding(num_genres, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate upper-triangular causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask  # True = positions to mask

    def forward(self, tokens: torch.Tensor, genre: torch.Tensor | None = None):
        """
        Args:
            tokens: (batch, seq_len) token indices
            genre: (batch,) optional genre indices
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device

        # Token + positional embedding
        x = self.token_embedding(tokens) * math.sqrt(self.d_model)  # (B, T, D)
        x = self.pos_encoding(x)

        # Optional genre conditioning: h_t = Emb(x_t) + Emb(genre)
        if genre is not None:
            genre_emb = self.genre_embedding(genre).unsqueeze(1)  # (B, 1, D)
            x = x + genre_emb

        # Causal mask & padding mask
        causal_mask = self._generate_causal_mask(seq_len, device)
        key_padding_mask = (tokens == self.pad_token)

        for layer in self.layers:
            x = layer(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)

        x = self.norm(x)
        logits = self.output_proj(x)  # (B, T, vocab_size)
        return logits

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Autoregressive cross-entropy loss: L_TR = -sum_t log p(x_t | x_{<t})"""
        # logits: (B, T, V), targets: (B, T)
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=self.pad_token,
        )

    @staticmethod
    def perplexity(loss: torch.Tensor) -> torch.Tensor:
        """Perplexity = exp(L_TR / T)"""
        return torch.exp(loss)

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_len: int = 512,
                 genre: torch.Tensor | None = None,
                 temperature: float = TF_TEMPERATURE,
                 top_k: int = TF_TOP_K) -> torch.Tensor:
        """Autoregressive generation: x_t ~ p_theta(x_t | x_{<t}).

        Args:
            prompt: (1, prompt_len) initial token sequence
            max_len: maximum generation length
            genre: (1,) optional genre index
            temperature: sampling temperature
            top_k: number of top tokens to sample from

        Returns:
            generated: (1, total_len) full generated sequence
        """
        self.eval()
        generated = prompt.clone()

        for _ in range(max_len - prompt.size(1)):
            # Truncate to max context window
            context = generated[:, -self.max_seq_len:]
            logits = self.forward(context, genre)
            next_logits = logits[:, -1, :] / temperature  # (1, vocab_size)

            # Top-k filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(next_logits, top_k)
                threshold = top_k_vals[:, -1].unsqueeze(-1)
                next_logits[next_logits < threshold] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
            generated = torch.cat([generated, next_token], dim=1)

            # Stop at EOS
            if next_token.item() == 2:  # EOS_TOKEN
                break

        return generated
