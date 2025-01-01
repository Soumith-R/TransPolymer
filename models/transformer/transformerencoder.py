# polymer_property_predictor/models/transformer/encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism specialized for chemical structures.
    This allows the model to attend to different chemical substructures simultaneously.
    """
    def __init__(self, 
                 d_model: int = 256,
                 n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Linear projections for Query, Key, and Value
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor for dot product attention
        self.scale = math.sqrt(self.d_head)

    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                chemical_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]
            chemical_bias: Optional bias for chemical substructures
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size = query.size(0)
        
        # Project and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Add chemical bias if provided
        if chemical_bias is not None:
            scores = scores + chemical_bias.unsqueeze(1)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute weighted sum
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project back
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(output)
        
        return output

class PositionWiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network with chemical feature enhancement.
    """
    def __init__(self, 
                 d_model: int = 256,
                 d_ff: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Additional chemical feature processing
        self.chemical_gate = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of position-wise feed-forward network.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Main feed-forward path
        output = F.gelu(self.ff1(x))
        output = self.dropout(output)
        output = self.ff2(output)
        
        # Chemical gating mechanism
        gate = torch.sigmoid(self.chemical_gate(x))
        output = gate * output + (1 - gate) * x
        
        return output

class EncoderLayer(nn.Module):
    """
    Single encoder layer with chemical structure awareness.
    """
    def __init__(self,
                 d_model: int = 256,
                 n_heads: int = 8,
                 d_ff: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                chemical_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of encoder layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]
            chemical_bias: Optional chemical structure bias
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention block
        attn_output = self.self_attn(x, x, x, mask, chemical_bias)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward block
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class PolymerEncoder(nn.Module):
    """
    Complete polymer encoder that processes chemical structures and additional features.
    """
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 768,  # Match decoder
                 n_layers: int = 12,   # Match decoder
                 n_heads: int = 12,    # Match decoder 
                 d_ff: int = 3072,     # Match decoder
                 dropout: float = 0.1,
                 max_length: int = 150):
        super().__init__()
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        
        # Additional polymer feature embeddings
        self.mw_embedding = nn.Linear(1, d_model)
        self.pdi_embedding = nn.Linear(1, d_model)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self,
                token_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                mw: torch.Tensor,
                pdi: torch.Tensor,
                chemical_bias: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of polymer encoder.
        
        Args:
            token_ids: Token indices [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            mw: Molecular weight [batch_size, 1]
            pdi: Polydispersity index [batch_size, 1]
            chemical_bias: Optional chemical structure bias
            
        Returns:
            Dictionary containing encoded representations and attention weights
        """
        # Create position indices
        positions = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0)
        
        # Compute embeddings
        token_emb = self.token_embedding(token_ids)
        pos_emb = self.position_embedding(positions)
        mw_emb = self.mw_embedding(mw.unsqueeze(-1))
        pdi_emb = self.pdi_embedding(pdi.unsqueeze(-1))
        
        # Combine embeddings
        x = token_emb + pos_emb
        x = x + mw_emb.unsqueeze(1) + pdi_emb.unsqueeze(1)
        x = self.dropout(x)
        
        # Create attention mask
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Pass through encoder layers
        layer_outputs = []
        for layer in self.layers:
            x = layer(x, attention_mask, chemical_bias)
            layer_outputs.append(x)
        
        # Final layer normalization
        x = self.norm(x)
        
        return {
            'last_hidden_state': x,
            'all_layer_outputs': layer_outputs,
            'attention_mask': attention_mask
        }
