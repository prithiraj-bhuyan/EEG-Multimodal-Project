"""
TASK 1 TRAINING - Enhanced with New Dataset Features + Enhanced Architecture
Production-ready code with all optimizations

New Features:
- Works with both EEGTransformer (full) and EEGTransformerSimple
- Class-weighted loss for handling imbalance
- Learning rate scheduling
- Gradient clipping
- Early stopping
- Better checkpoint management
- Comprehensive metrics tracking
"""

# ============================================================
# STEP 1: IMPORTS
# ============================================================
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import numpy as np

# Add your project folder to path (adjust if needed)
sys.path.append('/jet/home/gulavani/Project/final')

# Import your fixed modules
from eeg_dataset import create_datasets  # Enhanced dataset with augmentation
# CHANGE 1: Import both model versions
from eeg_encoder import EEGTransformer, EEGTransformerSimple

print("✅ Imports successful!")


# ============================================================
# STEP 2: SETUP PATHS AND CONFIG
# ============================================================
BIDS_ROOT = '/ocean/projects/cis250019p/gandotra/11785-gp-eeg/ds005589'
IMAGE_DIR = '/ocean/projects/cis250019p/gandotra/11785-gp-eeg/images'
CAPTIONS_FILE = '/ocean/projects/cis250019p/gandotra/11785-gp-eeg/captions.txt'

config = {
    # Data
    'batch_size': 64,  # Increased from 32 for better gradient estimates
    'augment_train': True,  # Enable augmentation
    
    # Model
    'model_type': 'EEGTransformerSimple',  # CHANGE 2: Choose 'EEGTransformer' or 'EEGTransformerSimple'
    'd_model': 256,
    'nhead': 8,
    'num_layers': 4,
    'dropout': 0.3,  # Dropout rate
    'use_attention_pooling': True,  # CHANGE 3: NEW - For full model only
    
    # Optimization
    'learning_rate': 1e-3,  # Start higher, will decay
    'weight_decay': 1e-4,  # Increased for better regularization
    'num_epochs': 100,  # Increased for better convergence
    'grad_clip': 1.0,  # Gradient clipping
    'label_smoothing': 0.1,  # Label smoothing
    
    # Scheduler
    'scheduler': 'cosine',  # 'cosine' or 'plateau'
    'warmup_epochs': 5,  # Warmup period
    
    # Early stopping
    'patience': 15,  # Stop if no improvement for 15 epochs
    
    # Loss weighting
    'use_class_weights': True,  # Handle class imbalance
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {device}")


# ============================================================
# STEP 3: CREATE ENHANCED DATASETS (NO CHANGES NEEDED)
# ============================================================
print("\n" + "="*60)
print("CREATING DATASETS WITH AUGMENTATION")
print("="*60)

train_ds, val_ds, test_ds = create_datasets(
    bids_root=BIDS_ROOT,
    images_dir=IMAGE_DIR,
    captions_path=CAPTIONS_FILE,
    augment_train=config['augment_train']
)

# Get class weights from dataset
if config['use_class_weights']:
    class_weights = train_ds.get_class_weights().to(device)
    print(f"\n✅ Class weights loaded: {class_weights.shape}")
    print(f"   Min weight: {class_weights.min():.4f}")
    print(f"   Max weight: {class_weights.max():.4f}")
else:
    class_weights = None

# Create dataloaders
train_loader = DataLoader(
    train_ds, 
    batch_size=config['batch_size'], 
    shuffle=True, 
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_ds, 
    batch_size=config['batch_size'], 
    shuffle=False, 
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_ds, 
    batch_size=config['batch_size'], 
    shuffle=False, 
    num_workers=4,
    pin_memory=True
)

print(f"\n✅ DataLoaders created:")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches:   {len(val_loader)}")
print(f"   Test batches:  {len(test_loader)}")


# ============================================================
# STEP 4: CREATE MODEL - ENHANCED VERSION
# ============================================================
print("\n" + "="*60)
print("CREATING ENHANCED MODEL")
print("="*60)

# CHANGE 4: Conditional model creation based on config
if config['model_type'] == 'EEGTransformer':
    print("Using FULL Enhanced EEG Transformer (Multi-scale CNN + Attention Pooling)")
    model = EEGTransformer(
        num_electrodes=122,
        time_points=500,
        num_classes=20,
        num_subjects=13,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        use_attention_pooling=config.get('use_attention_pooling', True)
    ).to(device)
elif config['model_type'] == 'EEGTransformerSimple':
    print("Using SIMPLE Enhanced EEG Transformer (Simplified but effective)")
    model = EEGTransformerSimple(
        num_electrodes=122,
        time_points=500,
        num_classes=20,
        num_subjects=13,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
else:
    raise ValueError(f"Unknown model_type: {config['model_type']}")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✅ Model created:")
print(f"   Architecture: {config['model_type']}")
print(f"   Total parameters:     {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")


# ============================================================
# STEP 5: SETUP TRAINING WITH ENHANCEMENTS (NO CHANGES NEEDED)
# ============================================================
# Loss with class weights and label smoothing
criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=config.get('label_smoothing', 0.0)
)

# Optimizer
optimizer = optim.AdamW(
    model.parameters(), 
    lr=config['learning_rate'], 
    weight_decay=config['weight_decay'],
    betas=(0.9, 0.999)
)

# Learning rate scheduler
if config['scheduler'] == 'cosine':
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=config['num_epochs'],
        eta_min=1e-6
    )
elif config['scheduler'] == 'plateau':
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
else:
    scheduler = None

# Warmup scheduler
def get_lr_multiplier(epoch, warmup_epochs=5):
    """Linear warmup for first few epochs"""
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0

# Create checkpoint directory
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('results', exist_ok=True)


# ============================================================
# STEP 6: ENHANCED TRAINING FUNCTIONS (NO CHANGES NEEDED)
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device, epoch, config):
    """Train for one epoch with all enhancements"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Apply warmup
    lr_mult = get_lr_multiplier(epoch, config['warmup_epochs'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = config['learning_rate'] * lr_mult
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # Unpack batch
        eeg, _, _, labels, subject_ids = batch
        
        eeg = eeg.to(device)
        labels = labels.to(device)
        subject_ids = subject_ids.to(device)
        
        # Auto-transpose if needed
        if eeg.shape[1] == 500 and eeg.shape[2] == 122:
            eeg = eeg.permute(0, 2, 1)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits, _ = model(eeg, subject_ids) 
        
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            config['grad_clip']
        )
        
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{total_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'lr': f'{current_lr:.6f}'
        })
    
    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Validating'):
            eeg, _, _, labels, subject_ids = batch
            
            eeg = eeg.to(device)
            labels = labels.to(device)
            subject_ids = subject_ids.to(device)
            
            # Auto-transpose if needed
            if eeg.shape[1] == 500 and eeg.shape[2] == 122:
                eeg = eeg.permute(0, 2, 1)
            
            # Forward pass
            logits, _ = model(eeg, subject_ids)
            
            loss = criterion(logits, labels)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


# ============================================================
# STEP 7: INITIALIZE WANDB
# ============================================================
# CHANGE 5: Updated run name to reflect model type
wandb.init(
    project="eeg-classification-enhanced",
    name=f"{config['model_type']}-{config['d_model']}d-{config['num_layers']}L-aug",
    config=config,
    tags=['transformer', 'subject_heads', 'task1', 'augmented', config['model_type']]
)


# ============================================================
# REST OF THE CODE REMAINS EXACTLY THE SAME
# ============================================================
# STEP 8: TRAINING LOOP WITH EARLY STOPPING
# STEP 9: SAVE TRAINING METRICS
# STEP 10: PLOT COMPREHENSIVE RESULTS
# STEP 11: LOAD BEST MODEL AND TEST
# STEP 12: VERIFY MODEL FOR TASK 2B
# ... (Keep all remaining code exactly as is)

print("\n" + "="*60)
print("STARTING TRAINING WITH ENHANCEMENTS")
print("="*60)

best_val_acc = 0
best_epoch = 0
patience_counter = 0
num_epochs = config['num_epochs']

# Lists for plotting
train_losses = []
train_accs = []
val_losses = []
val_accs = []
learning_rates = []

for epoch in range(1, num_epochs + 1):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch}/{num_epochs}")
    print('='*60)
    
    # Train
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device, epoch, config
    )
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # Store metrics
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    
    # Log to wandb
    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'learning_rate': current_lr
    })
    
    # Print results
    print(f"\nResults:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f}   | Val Acc:   {val_acc:.2f}%")
    print(f"  Learning Rate: {current_lr:.6f}")
    
    # Update scheduler
    if scheduler is not None:
        if config['scheduler'] == 'plateau':
            scheduler.step(val_acc)
        else:
            scheduler.step()
    
    # Save checkpoint every 10 epochs
    if epoch % 10 == 0:
        checkpoint_path = f'checkpoints/model_epoch_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_acc': val_acc,
            'train_acc': train_acc,
            'config': config,
        }, checkpoint_path)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        patience_counter = 0
        
        checkpoint_path = 'checkpoints/best_model_enhanced.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_acc': val_acc,
            'train_acc': train_acc,
            'config': config,
        }, checkpoint_path)
        
        wandb.save(checkpoint_path)
        print(f"  ✅ NEW BEST! Val Acc: {val_acc:.2f}% (Epoch {epoch})")
    else:
        patience_counter += 1
        print(f"  No improvement. Patience: {patience_counter}/{config['patience']}")
    
    # Early stopping
    if patience_counter >= config['patience']:
        print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
        print(f"   Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
        break

print("\n" + "="*60)
print("🎉 TRAINING COMPLETE!")
print("="*60)
print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")


# ============================================================
# STEP 9: SAVE TRAINING METRICS
# ============================================================
metrics = {
    'best_val_acc': best_val_acc,
    'best_epoch': best_epoch,
    'final_train_acc': train_accs[-1],
    'final_val_acc': val_accs[-1],
    'train_losses': train_losses,
    'train_accs': train_accs,
    'val_losses': val_losses,
    'val_accs': val_accs,
    'learning_rates': learning_rates,
    'config': config
}

with open('results/training_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("✅ Metrics saved to results/training_metrics.json")


# ============================================================
# STEP 10: PLOT COMPREHENSIVE RESULTS
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Accuracy
axes[0, 0].plot(train_accs, label='Train Acc', linewidth=2)
axes[0, 0].plot(val_accs, label='Val Acc', linewidth=2)
axes[0, 0].axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best ({best_epoch})')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy (%)')
axes[0, 0].set_title('Training & Validation Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Loss
axes[0, 1].plot(train_losses, label='Train Loss', linewidth=2)
axes[0, 1].plot(val_losses, label='Val Loss', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Training & Validation Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Learning Rate
axes[1, 0].plot(learning_rates, linewidth=2, color='green')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Learning Rate')
axes[1, 0].set_title('Learning Rate Schedule')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Train-Val Gap
gap = np.array(train_accs) - np.array(val_accs)
axes[1, 1].plot(gap, linewidth=2, color='orange')
axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy Gap (%)')
axes[1, 1].set_title('Generalization Gap (Train - Val)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_comprehensive.png', dpi=300, bbox_inches='tight')
print("✅ Comprehensive training plots saved to results/training_comprehensive.png")


# ============================================================
# STEP 11: LOAD BEST MODEL AND TEST
# ============================================================
print("\n" + "="*60)
print("LOADING BEST MODEL FOR FINAL EVALUATION")
print("="*60)

checkpoint = torch.load('checkpoints/best_model_enhanced.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Evaluate on test set
test_loss, test_acc = validate(model, test_loader, criterion, device)

print(f"\n✅ FINAL TEST RESULTS:")
print(f"   Test Loss: {test_loss:.4f}")
print(f"   Test Acc:  {test_acc:.2f}%")

wandb.log({
    'test_loss': test_loss,
    'test_acc': test_acc
})


# ============================================================
# STEP 12: VERIFY MODEL FOR TASK 2B
# ============================================================
print("\n" + "="*60)
print("VERIFYING MODEL FOR TASK 2B COMPATIBILITY")
print("="*60)

with torch.no_grad():
    test_eeg, _, _, _, _ = next(iter(val_loader))
    test_eeg = test_eeg.to(device)
    
    # Auto-transpose
    if test_eeg.shape[1] == 500 and test_eeg.shape[2] == 122:
        test_eeg = test_eeg.permute(0, 2, 1)
    
    # Extract embeddings (pass None for subject_ids)
    embeddings = model(test_eeg, subject_ids=None)
    
    print(f"✅ Embeddings extracted successfully!")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Mean:  {embeddings.mean().item():.4f}")
    print(f"   Std:   {embeddings.std().item():.4f}")

print("\n" + "="*60)
print("✅ MODEL IS READY FOR TASK 2B!")
print("="*60)

# Save final summary
summary = {
    'best_val_acc': best_val_acc,
    'best_epoch': best_epoch,
    'test_acc': test_acc,
    'test_loss': test_loss,
    'total_params': total_params,
    'trainable_params': trainable_params,
    'config': config
}

with open('results/final_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)

print("\n✅ Final summary saved to results/final_summary.json")

wandb.finish()
