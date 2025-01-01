# polymer_property_predictor/models/transformer/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for combining encoder outputs with decoder states.
    Specialized for polymer property prediction.
    """
    def __init__(self, d_model: int = 768, n_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Additional projections for chemical features
        self.chemical_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_head]))
        
        # Add relative position bias
        self.rel_pos_bias = nn.Parameter(torch.zeros(n_heads, 150, 150))
        
        # Add gating mechanism
        self.gate = nn.Linear(d_model, d_model)
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                chemical_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of cross-attention mechanism.
        
        Args:
            query: Query tensor from decoder [batch_size, tgt_len, d_model]
            key: Key tensor from encoder [batch_size, src_len, d_model]
            value: Value tensor from encoder [batch_size, src_len, d_model]
            mask: Attention mask
            chemical_features: Optional chemical feature tensor
            
        Returns:
            Tuple of (output tensor, attention weights)
        """
        batch_size = query.size(0)
        
        # Project and reshape
        Q = self.q_proj(query).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        
        # Apply chemical features if provided
        if chemical_features is not None:
            chemical_proj = self.chemical_proj(chemical_features)
            K = K + chemical_proj.unsqueeze(1)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(query.device)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute weighted sum
        output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(output)
        
        return output, attn_weights

class DecoderLayer(nn.Module):
    """
    Single decoder layer with chemical property awareness.
    """
    def __init__(self,
                d_model: int = 256,
                n_heads: int = 8,
                d_ff: int = 1024,
                dropout: float = 0.1):
        super().__init__()
        
        # Self attention
        self.self_attn = CrossAttention(d_model, n_heads, dropout)
        # Cross attention with encoder outputs
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.ff_net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                x: torch.Tensor,
                encoder_output: torch.Tensor,
                self_attn_mask: Optional[torch.Tensor] = None,
                cross_attn_mask: Optional[torch.Tensor] = None,
                chemical_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of decoder layer.
        
        Args:
            x: Input tensor
            encoder_output: Output from encoder
            self_attn_mask: Self-attention mask
            cross_attn_mask: Cross-attention mask
            chemical_features: Optional chemical features
            
        Returns:
            Tuple of (output tensor, attention weights dictionary)
        """
        # Self attention
        self_attn_out, self_attn_weights = self.self_attn(x, x, x, self_attn_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        
        # Cross attention
        cross_attn_out, cross_attn_weights = self.cross_attn(
            x, encoder_output, encoder_output,
            cross_attn_mask, chemical_features
        )
        x = self.norm2(x + self.dropout(cross_attn_out))
        
        # Feed-forward
        ff_out = self.ff_net(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x, {
            'self_attn': self_attn_weights,
            'cross_attn': cross_attn_weights
        }

class PropertyPredictor(nn.Module):
    """
    Head for predicting specific polymer properties.
    """
    def __init__(self, d_model: int, n_properties: int):
        super().__init__()
        
        self.property_net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_properties)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict polymer properties from encoded representation.
        
        Args:
            x: Encoded representation [batch_size, seq_len, d_model]
            
        Returns:
            Property predictions [batch_size, n_properties]
        """
        # Global average pooling
        x = torch.mean(x, dim=1)
        # Predict properties
        return self.property_net(x)

class PolymerDecoder(nn.Module):
    """
    Complete polymer decoder for property prediction and structure generation.
    """
    def __init__(self, vocab_size, d_model=768, n_layers=12, n_heads=12, d_ff=3072, dropout=0.1):
        super().__init__()
        self.d_model = d_model  # Make d_model accessible as instance attribute
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(512, d_model)
        
        # Transformer layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, n_layers)
        
        # Output layers
        self.cloud_point_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        self.phase_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input_ids, encoder_output, attention_mask=None):
        # Generate position IDs
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        
        # Get embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert boolean mask to transformer attention mask
            attention_mask = attention_mask.float()
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
            attention_mask = attention_mask.masked_fill(attention_mask == 1, float(0.0))
        
        # Pass through transformer
        transformer_output = self.transformer_decoder(
            x,
            encoder_output,
            tgt_key_padding_mask=attention_mask if attention_mask is not None else None
        )
        
        # Use [CLS] token output (first token) for classification
        pooled_output = transformer_output[:, 0]
        
        # Get predictions
        cloud_point = self.cloud_point_head(pooled_output)
        phase = self.phase_head(pooled_output)
        
        return {
            'cloud_point': cloud_point,
            'phase': phase
        }
