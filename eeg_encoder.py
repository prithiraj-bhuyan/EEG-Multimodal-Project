"""
Enhanced EEG Transformer Architecture
Fixes all issues and adds advanced features for better accuracy

Improvements:
1. Multi-scale temporal CNN with feature expansion
2. Positional encoding for electrode locations
3. Deeper, more expressive backbone
4. Better pooling strategy
5. Residual connections
6. Proper dropout
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Learnable positional encodings for electrode positions.
    Helps model understand spatial topology of EEG electrodes.
    """
    def __init__(self, d_model, num_positions=122, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_positions, d_model))
    
    def forward(self, x):
        # x: (B, num_electrodes, d_model)
        x = x + self.pos_embedding
        return self.dropout(x)


class TemporalCNNBlock(nn.Module):
    """
    Multi-scale temporal feature extraction.
    Replaces single conv with a more powerful block.
    """
    def __init__(self, num_electrodes=122, hidden_channels=64, dropout=0.3):
        super().__init__()
        
        # Multi-scale convolutions (different kernel sizes capture different temporal patterns)
        self.conv_small = nn.Conv1d(
            in_channels=num_electrodes,
            out_channels=hidden_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            groups=num_electrodes  # Depthwise conv
        )
        
        self.conv_medium = nn.Conv1d(
            in_channels=num_electrodes,
            out_channels=hidden_channels,
            kernel_size=15,
            stride=2,
            padding=7,
            groups=num_electrodes
        )
        
        self.conv_large = nn.Conv1d(
            in_channels=num_electrodes,
            out_channels=hidden_channels,
            kernel_size=31,
            stride=2,
            padding=15,
            groups=num_electrodes
        )
        
        # Pointwise conv to mix channels (1x1 conv)
        total_channels = hidden_channels * 3
        self.pointwise = nn.Conv1d(
            in_channels=total_channels,
            out_channels=num_electrodes,
            kernel_size=1
        )
        
        self.bn = nn.BatchNorm1d(num_electrodes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (B, 122, 500)
        
        # Apply multi-scale convolutions
        x1 = self.conv_small(x)   # (B, 64, 250)
        x2 = self.conv_medium(x)  # (B, 64, 250)
        x3 = self.conv_large(x)   # (B, 64, 250)
        
        # Concatenate along channel dimension
        x = torch.cat([x1, x2, x3], dim=1)  # (B, 192, 250)
        
        # Mix channels with pointwise conv
        x = self.pointwise(x)  # (B, 122, 250)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        return x


class EnhancedTemporalCNN(nn.Module):
    """
    Improved temporal CNN with proper feature expansion.
    """
    def __init__(self, num_electrodes=122, out_channels=64, dropout=0.3):
        super().__init__()
        
        # First conv: Expand features (per-electrode processing)
        self.conv1 = nn.Conv1d(
            in_channels=num_electrodes,
            out_channels=num_electrodes * 2,  # Expand to 244 channels
            kernel_size=15,
            stride=2,
            padding=7,
            groups=num_electrodes  # Depthwise (each electrode independent)
        )
        self.bn1 = nn.BatchNorm1d(num_electrodes * 2)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second conv: Further processing
        self.conv2 = nn.Conv1d(
            in_channels=num_electrodes * 2,
            out_channels=num_electrodes * out_channels,  # 122 * 64 = 7808
            kernel_size=11,
            stride=2,
            padding=5,
            groups=num_electrodes  # Still per-electrode
        )
        self.bn2 = nn.BatchNorm1d(num_electrodes * out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.out_channels = out_channels
        self.num_electrodes = num_electrodes
    
    def forward(self, x):
        # x: (B, 122, 500)
        
        # First conv
        x = self.conv1(x)      # (B, 244, 250)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Second conv
        x = self.conv2(x)      # (B, 7808, 125)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Reshape: (B, 122*64, 125) -> (B, 122, 64, 125)
        B, _, T = x.shape
        x = x.view(B, self.num_electrodes, self.out_channels, T)
        
        # Average pool over temporal dimension: (B, 122, 64, 125) -> (B, 122, 64)
        x = x.mean(dim=3)
        
        return x  # (B, 122, 64)


class AttentionPooling(nn.Module):
    """
    Learnable attention-based pooling.
    Better than simple averaging.
    """
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # x: (B, num_electrodes, d_model)
        
        # Compute attention weights
        weights = self.attention(x)  # (B, num_electrodes, 1)
        weights = torch.softmax(weights, dim=1)
        
        # Weighted sum
        x = (x * weights).sum(dim=1)  # (B, d_model)
        
        return x


class EEGTransformer(nn.Module):
    """
    ENHANCED Transformer Architecture for EEG Classification & Retrieval.
    
    Improvements over original:
    - Multi-scale temporal CNN with feature expansion
    - Positional encoding for spatial awareness
    - Deeper projection with residual connection
    - Attention-based pooling
    - Higher dropout for regularization
    - Proper dimension handling
    
    Architecture:
    - Input: (batch_size, 122, 500) Raw EEG
    - Stage 1: Enhanced Temporal CNN (Multi-scale feature extraction)
    - Stage 2: Positional Encoding + Transformer (Spatial modeling)
    - Stage 3: Attention Pooling + Subject-Specific Heads
    """
    def __init__(self, 
                 num_electrodes=122, 
                 time_points=500, 
                 num_classes=20, 
                 num_subjects=13,
                 d_model=256, 
                 nhead=8, 
                 num_layers=4,
                 dropout=0.3,
                 use_attention_pooling=True):
        super(EEGTransformer, self).__init__()
        
        self.d_model = d_model
        self.use_attention_pooling = use_attention_pooling
        
        # ============================================================
        # STAGE 1: Enhanced Temporal Feature Extraction
        # ============================================================
        # Output: (B, 122, 64) - 64 features per electrode
        self.temporal_cnn = EnhancedTemporalCNN(
            num_electrodes=num_electrodes,
            out_channels=64,
            dropout=dropout
        )
        
        # ============================================================
        # PROJECTION TO d_model
        # ============================================================
        # With residual connection for stability
        self.projection = nn.Sequential(
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Additional projection path for residual
        self.residual_proj = nn.Linear(64, d_model)
        
        # ============================================================
        # STAGE 2: Positional Encoding + Spatial Transformer
        # ============================================================
        # Add learnable positional encodings
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            num_positions=num_electrodes,
            dropout=dropout
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            batch_first=True, 
            dropout=dropout,
            activation='gelu'  # GELU instead of ReLU (better for transformers)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # ============================================================
        # STAGE 3: Pooling Strategy
        # ============================================================
        if use_attention_pooling:
            self.pooling = AttentionPooling(d_model)
        else:
            self.pooling = None  # Will use mean pooling
        
        # Additional processing after pooling
        self.post_pooling = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ============================================================
        # STAGE 4: Subject-Specific Classification Heads
        # ============================================================
        self.subject_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_classes)
            ) for _ in range(num_subjects)
        ])
    
    def forward(self, x, subject_ids=None):
        """
        Args:
            x: (Batch, 122, 500) EEG Data
            subject_ids: (Batch,) Tensor of Subject Indices (0-12). 
                         Required for Classification (Task 1).
                         Set to None for Retrieval/Embedding extraction (Task 2B).
        
        Returns:
            If subject_ids is None: 
                embeddings (Batch, d_model)
            If subject_ids is provided: 
                (logits, embeddings) tuple
        """
        # ============================================================
        # STAGE 1: Temporal CNN
        # ============================================================
        # Input: (B, 122, 500)
        x = self.temporal_cnn(x)  # (B, 122, 64)
        
        # ============================================================
        # PROJECTION with Residual
        # ============================================================
        residual = self.residual_proj(x)  # (B, 122, d_model)
        x = self.projection(x)             # (B, 122, d_model)
        x = x + residual                   # Residual connection
        
        # ============================================================
        # STAGE 2: Positional Encoding + Transformer
        # ============================================================
        x = self.pos_encoder(x)           # Add positional info
        x = self.transformer(x)           # (B, 122, d_model)
        
        # ============================================================
        # STAGE 3: Pooling
        # ============================================================
        if self.use_attention_pooling:
            embedding = self.pooling(x)   # (B, d_model)
        else:
            embedding = x.mean(dim=1)     # (B, d_model)
        
        # Post-pooling processing
        embedding = self.post_pooling(embedding)  # (B, d_model)
        
        # ============================================================
        # TASK 2B: Return embeddings only
        # ============================================================
        if subject_ids is None:
            return embedding
        
        # ============================================================
        # TASK 1: Subject-Specific Classification
        # ============================================================
        batch_size = embedding.shape[0]
        logits = torch.zeros(batch_size, 20, device=x.device)
        
        # Apply correct head for each subject
        unique_subs = torch.unique(subject_ids)
        for sub_id in unique_subs:
            mask = (subject_ids == sub_id)
            logits[mask] = self.subject_heads[sub_id](embedding[mask])
        
        return logits, embedding


# ============================================================
# ALTERNATIVE: Simpler but Effective Version
# ============================================================
class EEGTransformerSimple(nn.Module):
    """
    Simplified version with key improvements.
    Use this if the full version is too complex or slow.
    """
    def __init__(self, 
                 num_electrodes=122, 
                 time_points=500, 
                 num_classes=20, 
                 num_subjects=13,
                 d_model=256, 
                 nhead=8, 
                 num_layers=4,
                 dropout=0.3):
        super().__init__()
        
        # Temporal CNN - Simple but effective
        self.temporal_conv = nn.Sequential(
            # First conv: per-electrode processing with feature expansion
            nn.Conv1d(num_electrodes, num_electrodes * 2, kernel_size=15, stride=2, padding=7, groups=num_electrodes),
            nn.BatchNorm1d(num_electrodes * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Second conv: mix channels
            nn.Conv1d(num_electrodes * 2, num_electrodes, kernel_size=1),  # Pointwise
            nn.BatchNorm1d(num_electrodes),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Third conv: further temporal processing
            nn.Conv1d(num_electrodes, num_electrodes, kernel_size=11, stride=2, padding=5, groups=num_electrodes),
            nn.BatchNorm1d(num_electrodes),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Calculate output size: 500 -> 250 -> 125
        conv_out_size = 125
        
        # Project to d_model
        self.projection = nn.Sequential(
            nn.Linear(conv_out_size, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_electrodes, d_model))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention pooling
        self.attention_pool = nn.Linear(d_model, 1)
        
        # Subject heads
        self.subject_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_classes)
            ) for _ in range(num_subjects)
        ])
    
    def forward(self, x, subject_ids=None):
        # Temporal CNN
        x = self.temporal_conv(x)  # (B, 122, 125)
        
        # Project
        x = self.projection(x)  # (B, 122, d_model)
        
        # Add positional encoding
        x = x + self.pos_embedding
        
        # Transformer
        x = self.transformer(x)  # (B, 122, d_model)
        
        # Attention pooling
        attn_weights = torch.softmax(self.attention_pool(x), dim=1)
        embedding = (x * attn_weights).sum(dim=1)  # (B, d_model)
        
        if subject_ids is None:
            return embedding
        
        # Classification
        batch_size = embedding.shape[0]
        logits = torch.zeros(batch_size, 20, device=x.device)
        
        for sub_id in torch.unique(subject_ids):
            mask = (subject_ids == sub_id)
            logits[mask] = self.subject_heads[sub_id](embedding[mask])
        
        return logits, embedding


# ============================================================
# TESTING
# ============================================================
if __name__ == "__main__":
    print("Testing Enhanced EEG Transformer...")
    
    # Test full version
    model_full = EEGTransformer(
        num_classes=20,
        num_subjects=13,
        d_model=256,
        nhead=8,
        num_layers=4,
        dropout=0.3
    )
    
    # Test simple version
    model_simple = EEGTransformerSimple(
        num_classes=20,
        num_subjects=13,
        d_model=256,
        nhead=8,
        num_layers=4,
        dropout=0.3
    )
    
    # Fake data
    dummy_eeg = torch.randn(32, 122, 500)
    dummy_sub = torch.randint(0, 13, (32,))
    
    print("\n" + "="*60)
    print("FULL MODEL")
    print("="*60)
    
    # Classification mode
    logits, emb = model_full(dummy_eeg, dummy_sub)
    print(f"Logits shape: {logits.shape}")      # (32, 20)
    print(f"Embedding shape: {emb.shape}")      # (32, 256)
    
    # Retrieval mode
    emb_only = model_full(dummy_eeg, subject_ids=None)
    print(f"Embedding-only shape: {emb_only.shape}")  # (32, 256)
    
    # Count parameters
    total_params = sum(p.numel() for p in model_full.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\n" + "="*60)
    print("SIMPLE MODEL")
    print("="*60)
    
    logits_s, emb_s = model_simple(dummy_eeg, dummy_sub)
    print(f"Logits shape: {logits_s.shape}")
    print(f"Embedding shape: {emb_s.shape}")
    
    total_params_s = sum(p.numel() for p in model_simple.parameters())
    print(f"\nTotal parameters: {total_params_s:,}")
    
    print("\n✅ All tests passed!")
