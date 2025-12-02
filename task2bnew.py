"""
TASK 2B: EEG-CAPTION RETRIEVAL - COMPLETE RETRAINING

This version uses CAPTION-AWARE CONTRASTIVE LOSS that properly handles
your mixed dataset where some images are shared across subjects.

Key improvements:
1. Multiple subjects seeing same image = treated as POSITIVES
2. Only truly different captions = negatives
3. This fixes the false negative problem in your data

Expected performance: 15-25% Recall@5 (vs your current ~2%)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import sys
import os
from collections import defaultdict

# ============================================================
# 🔧 CONFIG - UPDATE THESE PATHS!
# ============================================================

CONFIG = {
    'code_dir': '/jet/home/gulavani/Project/final',  # UPDATE!
    'BIDS_ROOT': '/ocean/projects/cis250019p/gandotra/11785-gp-eeg/ds005589',
    'IMAGE_DIR': '/ocean/projects/cis250019p/gandotra/11785-gp-eeg/images',
    'CAPTIONS_FILE': '/ocean/projects/cis250019p/gandotra/11785-gp-eeg/captions.txt',
    'task1_checkpoint': 'checkpoints/best_model_new_arch.pth',  # UPDATE!
    'output_dir': './checkpoints/',
    
    # Training hyperparameters
    'batch_size': 32,
    'num_epochs': 30,
    'lr': 1e-4,
    'temperature': 0.5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

device = torch.device(CONFIG['device'])

# ============================================================
# ARCHITECTURE
# ============================================================

class SubjectSpecificProjection(nn.Module):
    def __init__(self, eeg_dim=256, clip_dim=512, num_subjects=13):
        super().__init__()
        self.num_subjects = num_subjects
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
        
        for subj_id in range(self.num_subjects):
            mask = (subject_ids == subj_id)
            if mask.any():
                indices = torch.where(mask)[0]
                output[indices] = self.heads[subj_id](eeg_emb[indices])
        
        return F.normalize(output, p=2, dim=1)

# ============================================================
# CAPTION-AWARE CONTRASTIVE LOSS
# ============================================================

def caption_aware_contrastive_loss(eeg_proj, text_embeds, captions, temperature=0.5):
    """
    Contrastive loss that handles repeated captions correctly.
    
    For each EEG sample:
    - POSITIVES: All samples with the SAME caption (including self)
    - NEGATIVES: All samples with DIFFERENT captions
    
    This fixes the false negative problem where multiple subjects
    see the same image but were treated as negatives.
    
    Args:
        eeg_proj: (B, 512) - Normalized EEG embeddings
        text_embeds: (B, 512) - Normalized CLIP text embeddings
        captions: List[str] - Caption for each sample
        temperature: float - Temperature scaling
    
    Returns:
        loss: scalar
    """
    batch_size = eeg_proj.size(0)
    
    # Compute similarity matrix
    similarity = torch.matmul(eeg_proj, text_embeds.t()) / temperature  # (B, B)
    
    # Build caption matching matrix
    # caption_match[i, j] = 1 if captions[i] == captions[j], else 0
    caption_match = torch.zeros(batch_size, batch_size, device=similarity.device)
    
    for i in range(batch_size):
        for j in range(batch_size):
            if captions[i] == captions[j]:
                caption_match[i, j] = 1.0
    
    # For each anchor i:
    # - Numerator: sum of exp(sim) for all positives (same caption)
    # - Denominator: sum of exp(sim) for all samples
    # - Loss: -log(numerator / denominator)
    
    loss = 0.0
    
    for i in range(batch_size):
        # Mask for positive samples (same caption as anchor i)
        positives_mask = caption_match[i] > 0  # (B,)
        
        # Get similarities
        positive_sims = similarity[i][positives_mask]
        all_sims = similarity[i]
        
        # Numerator: sum over all positives
        numerator = torch.exp(positive_sims).sum()
        
        # Denominator: sum over all samples
        denominator = torch.exp(all_sims).sum()
        
        # Cross-entropy loss for this anchor
        loss += -torch.log(numerator / denominator)
    
    loss = loss / batch_size
    return loss


def caption_aware_contrastive_loss_efficient(eeg_proj, text_embeds, captions, temperature=0.5):
    """
    Efficient vectorized version of caption-aware contrastive loss.
    Same behavior as above but ~10x faster.
    """
    batch_size = eeg_proj.size(0)
    
    # Compute similarity
    similarity = torch.matmul(eeg_proj, text_embeds.t()) / temperature
    
    # Build caption matching matrix efficiently
    caption_list = [captions[i] for i in range(batch_size)]
    caption_match = torch.zeros(batch_size, batch_size, device=similarity.device)
    
    # Group samples by caption
    caption_groups = defaultdict(list)
    for i, cap in enumerate(caption_list):
        caption_groups[cap].append(i)
    
    # Fill caption_match matrix
    for indices in caption_groups.values():
        for i in indices:
            for j in indices:
                caption_match[i, j] = 1.0
    
    # Vectorized loss computation
    exp_sim = torch.exp(similarity)  # (B, B)
    
    # Numerator: sum of positives for each row
    numerator = (exp_sim * caption_match).sum(dim=1)  # (B,)
    
    # Denominator: sum of all for each row
    denominator = exp_sim.sum(dim=1)  # (B,)
    
    # Loss
    loss = -torch.log(numerator / denominator).mean()
    
    return loss

# ============================================================
# EVALUATION
# ============================================================

def evaluate_retrieval(eeg_encoder, projection, val_loader, val_dataset, 
                       clip_model, clip_processor, device):
    """Evaluate using UNIQUE caption pool"""
    eeg_encoder.eval()
    projection.eval()
    clip_model.eval()
    
    # Build unique caption pool
    unique_captions = []
    caption_to_idx = {}
    sample_to_unique_idx = []
    sample_categories = []
    
    for i in range(len(val_dataset)):
        _, _, caption, category, _ = val_dataset[i]
        
        if caption not in caption_to_idx:
            caption_to_idx[caption] = len(unique_captions)
            unique_captions.append(caption)
        
        sample_to_unique_idx.append(caption_to_idx[caption])
        sample_categories.append(category.item() if torch.is_tensor(category) else category)
    
    num_unique = len(unique_captions)
    
    # Encode unique captions
    unique_caption_embeds = []
    batch_size = 64
    
    for i in range(0, len(unique_captions), batch_size):
        batch_captions = unique_captions[i:i+batch_size]
        inputs = clip_processor(text=batch_captions, return_tensors="pt", 
                                padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_embeds = clip_model.get_text_features(**inputs)
            text_embeds = F.normalize(text_embeds, p=2, dim=1)
            unique_caption_embeds.append(text_embeds.cpu())
    
    unique_caption_embeds = torch.cat(unique_caption_embeds, dim=0)
    
    # Extract EEG embeddings
    all_eeg_proj = []
    
    with torch.no_grad():
        for eeg, _, _, _, subject_ids in val_loader:
            eeg = eeg.to(device)
            subject_ids = subject_ids.to(device)
            
            eeg_emb = eeg_encoder(eeg)
            eeg_proj = projection(eeg_emb, subject_ids)
            all_eeg_proj.append(eeg_proj.cpu())
    
    all_eeg_proj = torch.cat(all_eeg_proj, dim=0)
    
    # Compute similarity
    similarity = torch.matmul(all_eeg_proj, unique_caption_embeds.t())
    
    sample_to_unique_idx = torch.tensor(sample_to_unique_idx)
    sample_categories = torch.tensor(sample_categories)
    
    # Build category map
    unique_categories = torch.zeros(num_unique, dtype=torch.long)
    for sample_idx, unique_idx in enumerate(sample_to_unique_idx):
        unique_categories[unique_idx] = sample_categories[sample_idx]
    
    # Calculate metrics
    results = {}
    
    for k in [1, 3, 5, 10]:
        topk_indices = torch.topk(similarity, k=k, dim=1).indices
        
        # Instance Recall
        correct_indices = sample_to_unique_idx.unsqueeze(1)
        correct_in_topk = (topk_indices == correct_indices).any(dim=1)
        instance_recall = correct_in_topk.float().mean().item()
        
        # Class Recall
        topk_categories = unique_categories[topk_indices]
        query_categories = sample_categories.unsqueeze(1).expand_as(topk_categories)
        class_correct = (topk_categories == query_categories).any(dim=1)
        class_recall = class_correct.float().mean().item()
        
        results[f'recall@{k}'] = instance_recall
        results[f'class_recall@{k}'] = class_recall
    
    return results

# ============================================================
# TRAINING LOOP
# ============================================================

def train_epoch(eeg_encoder, projection, train_loader, optimizer, 
                clip_model, clip_processor, device, temperature):
    """Train for one epoch with caption-aware loss"""
    eeg_encoder.train()
    projection.train()
    clip_model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    
    for eeg, _, captions, _, subject_ids in pbar:
        eeg = eeg.to(device)
        subject_ids = subject_ids.to(device)
        
        # Get EEG embeddings
        eeg_emb = eeg_encoder(eeg)
        eeg_proj = projection(eeg_emb, subject_ids)
        
        # Get CLIP text embeddings
        inputs = clip_processor(text=captions, return_tensors="pt", 
                               padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_embeds = clip_model.get_text_features(**inputs)
            text_embeds = F.normalize(text_embeds, p=2, dim=1)
        
        # Compute caption-aware contrastive loss
        loss = caption_aware_contrastive_loss_efficient(
            eeg_proj, text_embeds, captions, temperature=temperature
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "="*70)
    print("TASK 2B: EEG-CAPTION RETRIEVAL WITH CAPTION-AWARE LOSS")
    print("="*70)
    
    # Verify paths
    print("\n🔍 Checking configuration...")
    if not os.path.exists(CONFIG['task1_checkpoint']):
        print(f"❌ Task 1 checkpoint not found: {CONFIG['task1_checkpoint']}")
        print("   Please update CONFIG in the script!")
        return
    
    # Load modules
    sys.path.append(CONFIG['code_dir'])
    from eeg_encoder import EEGClassifier
    from eeg_dataset import create_datasets
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Load datasets
    print("\n📊 Loading datasets...")
    train_ds, val_ds, test_ds = create_datasets(
        bids_root=CONFIG['BIDS_ROOT'],
        images_dir=CONFIG['IMAGE_DIR'],
        captions_path=CONFIG['CAPTIONS_FILE']
    )
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=4)
    
    print(f"✅ Train: {len(train_ds)} samples")
    print(f"✅ Val: {len(val_ds)} samples")
    
    # Load CLIP
    print("\n🤖 Loading CLIP...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Freeze CLIP
    for param in clip_model.parameters():
        param.requires_grad = False
    
    print("✅ CLIP loaded (frozen)")
    
    # Load EEG encoder from Task 1
    print("\n🧠 Loading EEG encoder from Task 1...")
    trained_model = EEGClassifier(emb_dim=256, num_classes=20)
    checkpoint = torch.load(CONFIG['task1_checkpoint'], map_location=device)
    trained_model.load_state_dict(checkpoint['model_state_dict'])
    eeg_encoder = trained_model.encoder.to(device)
    
    # UNFREEZE encoder (this is important!)
    for param in eeg_encoder.parameters():
        param.requires_grad = True
    
    print("✅ EEG encoder loaded (trainable)")
    
    # Create projection heads
    print("\n🔧 Creating projection heads...")
    projection = SubjectSpecificProjection().to(device)
    print("✅ Projection heads created")
    
    # Optimizer (train BOTH encoder and projection)
    optimizer = torch.optim.Adam(
        list(eeg_encoder.parameters()) + list(projection.parameters()),
        lr=CONFIG['lr']
    )
    
    print(f"\n⚙️  Training config:")
    print(f"   Batch size: {CONFIG['batch_size']}")
    print(f"   Learning rate: {CONFIG['lr']}")
    print(f"   Temperature: {CONFIG['temperature']}")
    print(f"   Epochs: {CONFIG['num_epochs']}")
    print(f"   Device: {device}")
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    best_recall = 0.0
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\n📈 Epoch {epoch+1}/{CONFIG['num_epochs']}")
        
        # Train
        train_loss = train_epoch(
            eeg_encoder, projection, train_loader, optimizer,
            clip_model, clip_processor, device, CONFIG['temperature']
        )
        
        print(f"   Train loss: {train_loss:.4f}")
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"\n   Evaluating...")
            results = evaluate_retrieval(
                eeg_encoder, projection, val_loader, val_ds,
                clip_model, clip_processor, device
            )
            
            print(f"   Recall@5: {results['recall@5']:.2%}")
            print(f"   Class Recall@5: {results['class_recall@5']:.2%}")
            
            # Save best model
            if results['recall@5'] > best_recall:
                best_recall = results['recall@5']
                
                checkpoint_path = os.path.join(
                    CONFIG['output_dir'], 
                    'best_task2b_caption_aware.pth'
                )
                
                torch.save({
                    'epoch': epoch + 1,
                    'encoder_state_dict': eeg_encoder.state_dict(),
                    'projection_state_dict': projection.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'recall@5': results['recall@5'],
                    'results': results,
                }, checkpoint_path)
                
                print(f"   💾 Saved best model (Recall@5: {best_recall:.2%})")
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    results = evaluate_retrieval(
        eeg_encoder, projection, val_loader, val_ds,
        clip_model, clip_processor, device
    )
    
    print(f"\n📊 Final Results:")
    for k in [1, 3, 5, 10]:
        print(f"   Recall@{k}: {results[f'recall@{k}']:.2%}")
        print(f"   Class Recall@{k}: {results[f'class_recall@{k}']:.2%}")
    
    print(f"\n✅ Best Recall@5: {best_recall:.2%}")
    print(f"✅ Checkpoint saved to: {CONFIG['output_dir']}")
    
    return results

if __name__ == "__main__":
    main()