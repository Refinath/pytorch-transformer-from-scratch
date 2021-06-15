"""
Core Transformer modules implemented from scratch using PyTorch.

This file defines the main building blocks of the Transformer architecture: multi‑head
attention, position‑wise feed‑forward networks, positional encoding, encoder and
decoder layers, and the overall Transformer model.  The implementation follows
the design described in Vaswani et al., 2017 and the Annotated Transformer
notebook【751605239765780†L374-L383】.  Residual connections, layer normalisation and
masking are used as described in the paper.

Usage:

    from model import Transformer
    model = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        max_seq_length=50,
        dropout=0.1,
        pad_id=0,
    )
    out = model(src, tgt)

The module is deliberately self‑contained with minimal dependencies.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi‑Head Attention layer.

    Computes scaled dot‑product attention over multiple heads in parallel, then
    concatenates and projects the results back to the original dimension.  The
    implementation closely follows the description in the Annotated Transformer
    【751605239765780†L374-L383】.  Masks are broadcast over all heads.

    Args:
        d_model: Dimensionality of input embeddings.
        num_heads: Number of attention heads.  Must divide d_model.
        dropout: Dropout probability applied to the attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # Linear projections for queries, keys, values and output
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute multi‑head attention.

        Args:
            query: (batch, seq_len, d_model)
            key: (batch, seq_len, d_model)
            value: (batch, seq_len, d_model)
            mask: (batch, 1, seq_len, seq_len) or (batch, seq_len, seq_len) or None

        Returns:
            Attended tensor of shape (batch, seq_len, d_model).
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.q_proj(query)  # (batch, seq, d_model)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Reshape for multi‑head: (batch, num_heads, seq, d_k)
        def _split(x: torch.Tensor) -> torch.Tensor:
            return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        Q = _split(Q)
        K = _split(K)
        V = _split(V)

        # Compute scaled dot‑product attention
        # Q @ K^T -> (batch, num_heads, seq, seq)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # Expand mask if necessary
            # Mask shape should broadcast to (batch, num_heads, seq, seq)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, V)  # (batch, num_heads, seq, d_k)
        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.o_proj(context)
        return out


class PositionWiseFeedForward(nn.Module):
    """Position‑wise feed‑forward network.

    Applies two linear transformations with a ReLU activation in between to each
    position independently【751605239765780†L440-L460】.

    Args:
        d_model: Input and output dimension.
        d_ff: Hidden dimension of the feed‑forward network.
        dropout: Dropout probability applied after the first linear layer.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Adds fixed positional encodings using sine and cosine functions to the input
    embeddings.  This allows the model to utilise the order of tokens【751605239765780†L500-L527】.

    Args:
        d_model: Dimension of the embeddings.
        max_len: Maximum sequence length supported.
        dropout: Dropout probability applied after adding positional encodings.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Create constant positional encoding matrix with shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        # Register as buffer so it's saved with model but not a parameter
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """Single encoder layer consisting of self‑attention and feed‑forward.

    Each encoder layer applies a multi‑head self‑attention sub‑layer followed by
    a position‑wise feed‑forward network with residual connections and layer
    normalisation【751605239765780†L239-L245】.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        d_ff: Feed‑forward hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        # Self‑attention sub‑layer with residual connection
        attn_out = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        # Feed‑forward sub‑layer with residual connection
        ff_out = self.feed_forward(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        return x


class DecoderLayer(nn.Module):
    """Single decoder layer consisting of masked self‑attention, cross‑attention
    and feed‑forward networks.

    The decoder layer uses three sub‑layers as described in the Transformer paper: a
    masked self‑attention sub‑layer to ensure the auto‑regressive property, a
    cross‑attention sub‑layer over the encoder output, and a feed‑forward network
    【751605239765780†L296-L299】【751605239765780†L374-L383】.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        d_ff: Feed‑forward hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor | None,
        tgt_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # Masked self‑attention (decoder attends to its own past)
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        # Cross‑attention (decoder attends over encoder output)
        attn_out = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = x + self.dropout(attn_out)
        x = self.norm2(x)
        # Feed‑forward network
        ff_out = self.feed_forward(x)
        x = x + self.dropout(ff_out)
        x = self.norm3(x)
        return x


class Transformer(nn.Module):
    """Full Transformer model with encoder and decoder stacks.

    This class wraps together embeddings, positional encoding, encoder layers,
    decoder layers and a final projection layer that converts decoder outputs to
    logits over the target vocabulary.  It includes helper methods to create
    source and target masks and to handle padding tokens.

    Args:
        src_vocab_size: Size of the source vocabulary.
        tgt_vocab_size: Size of the target vocabulary.
        d_model: Dimension of embeddings and hidden activations.
        num_heads: Number of attention heads.
        num_layers: Number of encoder and decoder layers.
        d_ff: Hidden dimension of feed‑forward networks.
        max_seq_length: Maximum input and output sequence length.
        dropout: Dropout probability.
        pad_id: Index of the padding token in vocabularies.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 512,
        max_seq_length: int = 100,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.pad_id = pad_id
        # Embedding layers
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        # Encoder and decoder stacks
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        # Final linear to project decoder output to vocab size
        self.out_proj = nn.Linear(d_model, tgt_vocab_size)

    def _generate_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create binary mask for source sequences (1 = keep, 0 = pad)."""
        # src: (batch, src_len)
        mask = (src != self.pad_id).unsqueeze(1).unsqueeze(2)  # (batch,1,1,src_len)
        return mask

    def _generate_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """Create combined padding and subsequent mask for target sequences.

        The target mask prevents positions from attending to subsequent positions to
        preserve the auto‑regressive property【751605239765780†L296-L305】.
        """
        batch_size, tgt_len = tgt.size()
        # Padding mask
        pad_mask = (tgt != self.pad_id).unsqueeze(1).unsqueeze(2)  # (batch,1,1,tgt_len)
        # Subsequent mask: allow attending to current and previous positions
        subsequent_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(0)  # (1,1,tgt_len,tgt_len)
        mask = pad_mask & subsequent_mask  # broadcast
        return mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer.

        Args:
            src: Source sequences of shape (batch, src_len).
            tgt: Target input sequences (typically shifted right) of shape (batch, tgt_len).

        Returns:
            Logits over the target vocabulary with shape (batch, tgt_len, tgt_vocab_size).
        """
        src_mask = self._generate_src_mask(src)
        tgt_mask = self._generate_tgt_mask(tgt)

        # Embedding and positional encoding
        src_emb = self.positional_encoding(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.positional_encoding(self.tgt_embed(tgt) * math.sqrt(self.d_model))

        # Encoder
        enc_out = src_emb
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask)

        # Decoder
        dec_out = tgt_emb
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)

        # Project to vocabulary
        logits = self.out_proj(dec_out)
        return logits
