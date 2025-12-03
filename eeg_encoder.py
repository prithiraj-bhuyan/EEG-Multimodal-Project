"""
Advanced EEG Transformer Architecture
Compliant with Write-up Section 3:
1. Temporal Learning (1D CNN)
2. Spatial Learning (Transformer)
3. Subject-Specific Classification Heads
"""

import torch
import torch.nn as nn

class EEGTransformer(nn.Module):
    """
    Unified Transformer Architecture for both Classification and Retrieval.
    
    Architecture:
    - Input: (batch_size, 122, 500) Raw EEG
    - Stage 1: Temporal 1D CNN (Independent per electrode)
    - Stage 2: Spatial Transformer (Attention across electrodes)
    - Stage 3: Subject-Specific Heads (13 separate linear layers)
    """
    def __init__(self, 
                 num_electrodes=122, 
                 time_points=500, 
                 num_classes=20, 
                 num_subjects=13,
                 d_model=256, 
                 nhead=8, 
                 num_layers=4):
        super(EEGTransformer, self).__init__()
        
        # 1. Define Conv Parameters
        k_size = 15
        stride = 4
        pad = 7
        
        # 2. Temporal Feature Extraction
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels=num_electrodes, 
                      out_channels=num_electrodes, 
                      kernel_size=k_size, 
                      stride=stride, 
                      padding=pad, 
                      groups=num_electrodes),
            nn.BatchNorm1d(num_electrodes),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        
        # 3. Calculate Output Dimension Dynamically
        # Formula: L_out = floor((L_in + 2*padding - kernel) / stride) + 1
        conv_out_size = int((time_points + 2*pad - k_size) / stride) + 1
        
        # Now use this calculated size instead of hardcoded '125'
        self.projection = nn.Linear(conv_out_size, d_model)
        
        # 4. Spatial Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=nhead, 
                                                   dim_feedforward=d_model*4, 
                                                   batch_first=True, 
                                                   dropout=0.25)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Subject-Specific Heads
        self.subject_heads = nn.ModuleList([
            nn.Linear(d_model, num_classes) for _ in range(num_subjects)
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
                embeddings (Batch, 256)
            If subject_ids is provided: 
                (logits, embeddings)
        """
        # 1. Temporal Conv: (B, 122, 500) -> (B, 122, 125)
        x = self.temporal_conv(x)
        
        # 2. Project to d_model: (B, 122, 125) -> (B, 122, 256)
        x = self.projection(x)
        
        # 3. Spatial Transformer: (B, 122, 256) -> (B, 122, 256)
        # The transformer treats the 122 electrodes as the sequence length.
        x = self.transformer(x)
        
        # 4. Aggregate: Average over electrodes -> (B, 256)
        embedding = x.mean(dim=1)
        
        # TASK 2B: If we just want embeddings, stop here.
        if subject_ids is None:
            return embedding
        
        # TASK 1: Subject-Specific Classification
        batch_size = x.shape[0]
        logits = torch.zeros(batch_size, 20, device=x.device)
        
        # Efficiently apply the correct head for each sample
        # (We iterate heads because vectorizing dynamic indexing of modules is hard)
        unique_subs = torch.unique(subject_ids)
        for sub_id in unique_subs:
            mask = (subject_ids == sub_id)
            logits[mask] = self.subject_heads[sub_id](embedding[mask])
                
        return logits, embedding

# ============================================================
# HOW TO USE THIS
# ============================================================

if __name__ == "__main__":
    # Test Architecture
    model = EEGTransformer(num_classes=20, num_subjects=13)
    
    # Fake Data
    dummy_eeg = torch.randn(32, 122, 500)
    dummy_sub = torch.randint(0, 13, (32,)) # Random subjects 0-12
    
    # 1. Classification Mode (Task 1)
    logits, emb = model(dummy_eeg, dummy_sub)
    print(f"Logits: {logits.shape}") # (32, 20)
    
    # 2. Retrieval Mode (Task 2B)
    emb_only = model(dummy_eeg, subject_ids=None)
    print(f"Embeddings: {emb_only.shape}") # (32, 256)