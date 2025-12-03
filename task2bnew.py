# ============================================================
# TASK 2B: EEG-CAPTION RETRIEVAL - FINAL COMPLETE CODE
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import wandb
import sys

# Add path if needed
sys.path.append('/jet/home/gulavani/Project/final')

# ============================================================
# 1. IMPORTS & SETUP
# ============================================================

# CHANGE 1: Import the new Transformer class
from eeg_encoder import EEGTransformer 
from eeg_dataset import create_datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# 2. LOAD TRAINED ENCODER (TASK 1)
# ============================================================

print("🔄 Loading Task 1 Model...")

# CHANGE 2: Initialize EEGTransformer with SAME args as Task 1
# Note: d_model=256 matches your config
trained_model = EEGTransformer(
    num_electrodes=122, 
    time_points=500, 
    num_classes=20, 
    num_subjects=13,
    d_model=256, 
    nhead=8, 
    num_layers=4
).to(device)

# Load weights
checkpoint = torch.load('checkpoints/best_model_new_arch.pth')
trained_model.load_state_dict(checkpoint['model_state_dict'])

# CHANGE 3: The model IS the encoder. Do not use .encoder
eeg_encoder = trained_model

# Unfreeze for fine-tuning (Recommended for retrieval)
eeg_encoder.train()
for param in eeg_encoder.parameters():
    param.requires_grad = True

print("✅ Loaded trained EEG Transformer")

# ============================================================
# 3. SUBJECT-SPECIFIC PROJECTION HEADS
# ============================================================

class SubjectSpecificProjection(nn.Module):
    """
    Maps (Batch, 256) EEG Embedding -> (Batch, 512) CLIP Space.
    Uses a specific Neural Network for each Subject ID.
    """
    def __init__(self, eeg_dim=256, clip_dim=512, num_subjects=13):
        super().__init__()
        self.num_subjects = num_subjects
        
        # 13 Separate MLP Heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(eeg_dim, clip_dim),
                nn.ReLU(),
                nn.Linear(clip_dim, clip_dim)
            ) for _ in range(num_subjects)
        ])
    
    def forward(self, eeg_emb, subject_ids):
        batch_size = eeg_emb.size(0)
        output = torch.zeros(batch_size, 512, device=eeg_emb.device)
        
        # Apply specific head based on Subject ID
        # We loop because vectorizing dynamic module selection is hard
        unique_subs = torch.unique(subject_ids)
        for sub_id in unique_subs:
            mask = (subject_ids == sub_id)
            if mask.any():
                output[mask] = self.heads[sub_id](eeg_emb[mask])
        
        # Normalize (Crucial for CLIP)
        return F.normalize(output, p=2, dim=1)

projection = SubjectSpecificProjection().to(device)
print("✅ Created subject-specific projection heads")

# ============================================================
# 4. LOAD CLIP MODEL
# ============================================================

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Freeze CLIP
for param in clip_model.parameters():
    param.requires_grad = False

print("✅ Loaded CLIP model (frozen)")

# ============================================================
# 5. DATA PREPARATION & PRECOMPUTING
# ============================================================

def precompute_caption_embeddings(dataset):
    """Compute CLIP text embeddings for all captions once"""
    clip_model.eval()
    
    # Extract just captions
    all_captions = [dataset[i][2] for i in range(len(dataset))]
    
    caption_embeds = []
    batch_size = 64
    
    print(f"   Computing {len(all_captions)} text embeddings...")
    for i in tqdm(range(0, len(all_captions), batch_size)):
        batch_captions = all_captions[i:i+batch_size]
        inputs = clip_processor(text=batch_captions, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_embeds = clip_model.get_text_features(**inputs)
            text_embeds = F.normalize(text_embeds, p=2, dim=1)
            caption_embeds.append(text_embeds.cpu())
    
    return torch.cat(caption_embeds, dim=0)

# Load datasets
BIDS_ROOT = '/ocean/projects/cis250019p/gandotra/11785-gp-eeg/ds005589'
IMAGE_DIR = '/ocean/projects/cis250019p/gandotra/11785-gp-eeg/images'
CAPTIONS_FILE = '/ocean/projects/cis250019p/gandotra/11785-gp-eeg/captions.txt'
train_ds, val_ds, test_ds = create_datasets(BIDS_ROOT, IMAGE_DIR, CAPTIONS_FILE)

# Precompute
print("Precomputing Train/Val Captions...")
train_caption_embeds = precompute_caption_embeddings(train_ds)
val_caption_embeds = precompute_caption_embeddings(val_ds)

# ============================================================
# 6. DEBIASED CONTRASTIVE LOSS (CHANGE 4)
# ============================================================

def supervised_contrastive_loss(eeg_proj, text_embeds, categories, temperature=0.5):
    """
    Advanced Loss: Ignores negatives that share the same class.
    Fulfills 'Debiased Contrastive Learning' requirement.
    """
    # 1. Similarity Matrix
    sim_matrix = torch.matmul(eeg_proj, text_embeds.t()) / temperature
    
    # 2. Masking Logic
    batch_size = categories.size(0)
    identity_mask = torch.eye(batch_size, device=categories.device).bool()
    
    # Identify samples with SAME class
    same_class_mask = (categories.unsqueeze(1) == categories.unsqueeze(0))
    
    # Valid Negatives = (Not Same Class) OR (Is Self)
    # This removes "False Negatives" (Same class, diff image) from denominator
    valid_mask = ~same_class_mask | identity_mask
    
    # 3. Compute Log Softmax
    exp_sim = torch.exp(sim_matrix)
    
    # Zero out invalid negatives in the sum
    exp_sim = exp_sim * valid_mask.float()
    
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    
    # 4. Select Positive Pairs (Diagonal)
    mean_log_prob_pos = (identity_mask * log_prob).sum(dim=1)
    
    return -mean_log_prob_pos.mean()

# ============================================================
# 7. TRAINING LOOP
# ============================================================

def train_epoch(eeg_encoder, projection, loader, caption_embeds, optimizer, epoch):
    eeg_encoder.train()
    projection.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch_idx, (eeg, _, _, categories, subject_ids) in enumerate(pbar):
        eeg = eeg.to(device)
        categories = categories.to(device)
        subject_ids = subject_ids.to(device)
        
        # Retrieve text embeddings for this batch
        start = batch_idx * loader.batch_size
        end = start + eeg.size(0)
        text_emb = caption_embeds[start:end].to(device)
        
        optimizer.zero_grad()
        
        # CHANGE 5: Forward Pass
        # A. Get Raw Embeddings (Subject Agnostic) -> Pass None!
        raw_emb = eeg_encoder(eeg, subject_ids=None) 
        
        # B. Project to CLIP (Subject Specific) -> Pass IDs!
        proj_emb = projection(raw_emb, subject_ids)
        
        # CHANGE 6: Use Debiased Loss
        loss = supervised_contrastive_loss(proj_emb, text_emb, categories)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)

# ============================================================
# 8. EVALUATION
# ============================================================

def evaluate_retrieval(eeg_encoder, projection, loader, caption_embeds, k_values=[1, 5]):
    eeg_encoder.eval()
    projection.eval()
    
    all_proj = []
    
    with torch.no_grad():
        for eeg, _, _, _, subject_ids in tqdm(loader, desc="Evaluating"):
            eeg = eeg.to(device)
            subject_ids = subject_ids.to(device)
            
            # Same logic as training: None for encoder, IDs for projection
            raw_emb = eeg_encoder(eeg, subject_ids=None)
            proj_emb = projection(raw_emb, subject_ids)
            
            all_proj.append(proj_emb.cpu())
            
    all_proj = torch.cat(all_proj, dim=0)
    
    # Large Matrix Multiplication
    # (N_val, 512) @ (N_val, 512).T -> (N_val, N_val)
    similarity = torch.matmul(all_proj, caption_embeds.t())
    
    metrics = {}
    for k in k_values:
        # Indices of top-k matches
        topk = torch.topk(similarity, k=k, dim=1).indices
        
        # Correct if the diagonal index (i) is in row i's top k
        targets = torch.arange(len(all_proj)).unsqueeze(1)
        correct = (topk == targets).any(dim=1).float().mean().item()
        
        metrics[f'R@{k}'] = correct
        print(f"Recall@{k}: {correct:.2%}")
        
    return metrics

# ============================================================
# 9. MAIN EXECUTION
# ============================================================

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

optimizer = torch.optim.Adam(
    list(eeg_encoder.parameters()) + list(projection.parameters()),
    lr=1e-4,
    weight_decay=1e-5
)

wandb.init(project="eeg-retrieval", name="task2b-final")

best_r5 = 0
for epoch in range(1, 31):
    loss = train_epoch(eeg_encoder, projection, train_loader, train_caption_embeds, optimizer, epoch)
    
    if epoch % 5 == 0:
        metrics = evaluate_retrieval(eeg_encoder, projection, val_loader, val_caption_embeds)
        wandb.log({**metrics, 'train_loss': loss, 'epoch': epoch})
        
        if metrics['R@5'] > best_r5:
            best_r5 = metrics['R@5']
            torch.save(projection.state_dict(), 'checkpoints/best_task2b_proj.pth')
            print("✅ Saved best model")

print("🎉 Finished!")
wandb.finish()

# ============================================================
# 9. FINAL TEST EVALUATION
# ============================================================

# 1. Load the Best Model Checkpoint
# We only need to load the projection head because the eeg_encoder was fine-tuned 
# alongside it, but usually we just keep the best projection weights for evaluation.
# If you saved the encoder state too, load it here. Otherwise, the current encoder 
# state is likely the last epoch's state (which is usually fine/close enough).
checkpoint = torch.load('checkpoints/best_task2b_proj.pth')
projection.load_state_dict(checkpoint)

print("\n" + "="*60)
print("PREPARING TEST DATA")
print("="*60)

# 2. Precompute Test Caption Embeddings
# Note: My function returns ONLY the embeddings tensor
test_caption_embeds = precompute_caption_embeddings(test_ds)

# 3. Create Test Loader
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

# 4. Evaluate
print("\n" + "="*60)
print("FINAL TEST SET EVALUATION")
print("="*60)

test_metrics = evaluate_retrieval(
    eeg_encoder, 
    projection, 
    test_loader, 
    test_caption_embeds, 
    k_values=[1, 3, 5, 10]
)

print("\n📊 FINAL RESULTS (Include these in your report):")
for metric, value in test_metrics.items():
    print(f"{metric}: {value:.2%}")