# ============================================================
# STEP 1: IMPORTS
# ============================================================
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# Add your project folder to path
sys.path.append('/jet/home/gulavani/Project/final')

# Import your fixed modules
from eeg_dataset import create_datasets
from eeg_encoder import EEGClassifier

print("✅ Imports successful!")


# ============================================================
# STEP 2: SETUP PATHS AND CONFIG
# ============================================================
BIDS_ROOT = '/ocean/projects/cis250019p/gandotra/11785-gp-eeg/ds005589'
IMAGE_DIR = '/ocean/projects/cis250019p/gandotra/11785-gp-eeg/images'
CAPTIONS_FILE = '/ocean/projects/cis250019p/gandotra/11785-gp-eeg/captions.txt'

config = {
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'num_epochs': 50,  # Should be fast to train
    'model_type': 'EEGClassifier_v2',
    'emb_dim': 256,  # NEW: embedding dimension
    'hid_dim': 512,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {device}")


# ============================================================
# STEP 3: CREATE DATASETS
# ============================================================
print("\n" + "="*60)
print("CREATING DATASETS")
print("="*60)

train_ds, val_ds, test_ds = create_datasets(
    bids_root=BIDS_ROOT,
    images_dir=IMAGE_DIR,
    captions_path=CAPTIONS_FILE
)

# Create dataloaders
train_loader = DataLoader(
    train_ds, 
    batch_size=config['batch_size'], 
    shuffle=True, 
    num_workers=4,
    pin_memory=True
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
# STEP 4: TEST DATALOADER (VERIFY IT WORKS)
# ============================================================
print("\n" + "="*60)
print("TESTING DATALOADER")
print("="*60)

eeg, img, caption, cat_label, subj_id = next(iter(train_loader))

print(f"✅ Batch loaded successfully:")
print(f"   EEG shape:     {eeg.shape}")        # Should be (32, 122, 500)
print(f"   Image shape:   {img.shape}")        # Should be (32, 3, 224, 224)
print(f"   Caption:       {type(caption)} (len={len(caption)})")
print(f"   Category:      {cat_label.shape}")  # Should be (32,)
print(f"   Subject ID:    {subj_id.shape}")    # Should be (32,)
print(f"\n   Sample category labels: {cat_label[:5]}")
print(f"   Sample subject IDs:     {subj_id[:5]}")


# ============================================================
# STEP 5: CREATE NEW MODEL
# ============================================================
print("\n" + "="*60)
print("CREATING NEW MODEL")
print("="*60)

model = EEGClassifier(
    in_dim=122*500,
    hid_dim=config['hid_dim'],
    emb_dim=config['emb_dim'],
    num_classes=20
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✅ Model created:")
print(f"   Total parameters:     {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")


# ============================================================
# STEP 6: TEST MODEL (VERIFY FORWARD PASS WORKS)
# ============================================================
print("\n" + "="*60)
print("TESTING MODEL")
print("="*60)

model.eval()
with torch.no_grad():
    test_eeg = eeg[:4].to(device)  # Test with 4 samples
    
    # Test classification
    logits = model(test_eeg)
    print(f"✅ Classification works:")
    print(f"   Input:  {test_eeg.shape}")
    print(f"   Output: {logits.shape}")  # Should be (4, 20)
    
    # Test embedding extraction
    embeddings = model.get_embeddings(test_eeg)
    print(f"\n✅ Embedding extraction works:")
    print(f"   Input:  {test_eeg.shape}")
    print(f"   Output: {embeddings.shape}")  # Should be (4, 256)

print("\n✅ Model ready for training!")


# ============================================================
# STEP 7: SETUP TRAINING
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(), 
    lr=config['learning_rate'], 
    weight_decay=config['weight_decay']
)

# Create checkpoint directory
os.makedirs('checkpoints', exist_ok=True)


# ============================================================
# STEP 8: TRAINING FUNCTIONS
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # Unpack batch (new format: 5 items)
        eeg, _, _, labels, _ = batch
        
        eeg = eeg.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(eeg)  # No softmax!
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        if (batch_idx + 1) % 50 == 0:
            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
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
            eeg, _, _, labels, _ = batch
            
            eeg = eeg.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(eeg)
            loss = criterion(logits, labels)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


# ============================================================
# STEP 9: INITIALIZE WANDB (OPTIONAL BUT RECOMMENDED)
# ============================================================
wandb.login(key="825201e63a02e53435b53a136158ab39815c89a4")  # Your key

wandb.init(
    project="eeg-classification",
    name="new-architecture-v2",
    config=config,
    tags=['new_arch', 'task2b_ready']
)

wandb.watch(model, criterion, log="all", log_freq=100)


# ============================================================
# STEP 10: TRAINING LOOP
# ============================================================
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

best_val_acc = 0
num_epochs = config['num_epochs']

for epoch in range(1, num_epochs + 1):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch}/{num_epochs}")
    print('='*60)
    
    # Train
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device, epoch
    )
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # Log to wandb
    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
    })
    
    # Print results
    print(f"\nResults:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f}   | Val Acc:   {val_acc:.2f}%")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        
        checkpoint_path = 'checkpoints/best_model_new_arch.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'train_acc': train_acc,
            'config': config,
        }, checkpoint_path)
        
        wandb.save(checkpoint_path)
        
        print(f"  ✅ NEW BEST! Val Acc: {val_acc:.2f}%")
        
        wandb.run.summary["best_val_acc"] = val_acc
        wandb.run.summary["best_epoch"] = epoch

print("\n" + "="*60)
print("🎉 TRAINING COMPLETE!")
print("="*60)
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")


# ============================================================
# STEP 11: FINAL VERIFICATION
# ============================================================
print("\n" + "="*60)
print("VERIFYING TRAINED MODEL")
print("="*60)

# Load best model
checkpoint = torch.load('checkpoints/best_model_new_arch.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Test embedding extraction
with torch.no_grad():
    test_eeg, _, _, _, _ = next(iter(val_loader))
    test_eeg = test_eeg.to(device)
    
    embeddings = model.get_embeddings(test_eeg)
    print(f"✅ Embeddings extracted successfully!")
    print(f"   Shape: {embeddings.shape}")  # Should be (batch_size, 256)
    print(f"   Mean:  {embeddings.mean().item():.4f}")
    print(f"   Std:   {embeddings.std().item():.4f}")

print("\n✅ MODEL IS READY FOR TASK 2B!")
print("\nNext steps:")
print("1. This encoder is now Task 2B compatible")
print("2. You can extract embeddings: model.get_embeddings(eeg)")
print("3. Ready to implement projection heads")
print("4. Ready to start CLIP integration")

wandb.finish()
# ```

# ---

## 🏃 **HOW TO RUN THIS**

# ### **Option 1: Jupyter Notebook** (Recommended)

# 1. Create new notebook: `train_new_encoder.ipynb`
# 2. Copy the entire code above
# 3. Split into cells at the `# ======` lines
# 4. Run cell by cell
# 5. Watch the training progress!

# ### **Option 2: Python Script**

# 1. Save code as `train_new_encoder.py`
# 2. Run: `python train_new_encoder.py`

# ---

# ## ⏱️ **EXPECTED TIMELINE**

# | Step | Time | What Happens |
# |------|------|--------------|
# | Imports & Setup | 30 sec | Load libraries |
# | Create Datasets | 2-3 min | Scan all EEG files |
# | Model Creation | 10 sec | Initialize model |
# | Epoch 1-10 | ~10 min | Quick convergence |
# | Epoch 11-30 | ~20 min | Stabilization |
# | Epoch 31-50 | ~20 min | Fine-tuning |
# | **TOTAL** | **~50 min** | Complete training |

# ---

# ## 📊 **WHAT TO EXPECT**

# ### **Training Progress:**
# ```
# Epoch 1/50:   Train 15% → Val 6%
# Epoch 5/50:   Train 30% → Val 7%
# Epoch 10/50:  Train 45% → Val 8%
# Epoch 20/50:  Train 55% → Val 8.5%
# Epoch 30/50:  Train 60% → Val 9%     ← BEST
# Epoch 50/50:  Train 65% → Val 8.8%   (slight overfit)