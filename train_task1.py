
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
# CHANGE 1: New Class Import
from eeg_encoder import EEGTransformer 

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
    'num_epochs': 50,
    'model_type': 'EEGTransformer_v1', # Updated Name
    'd_model': 256,  # Updated Config Key
    'nhead': 8,
    'num_layers': 4
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
print(f"   Category:      {cat_label.shape}")  # Should be (32,)
print(f"   Subject ID:    {subj_id.shape}")    # Should be (32,)


# ============================================================
# STEP 5: CREATE NEW MODEL
# ============================================================
print("\n" + "="*60)
print("CREATING NEW MODEL")
print("="*60)

# CHANGE 2: New Model Init
model = EEGTransformer(
    num_electrodes=122,
    time_points=500,
    num_classes=20,
    num_subjects=13,
    d_model=config['d_model'],
    nhead=config['nhead'],
    num_layers=config['num_layers']
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
    test_sub = subj_id[:4].to(device) # Needs subject IDs
    
    # Test classification (Task 1)
    logits, _ = model(test_eeg, test_sub)
    print(f"✅ Classification works:")
    print(f"   Input:  {test_eeg.shape}")
    print(f"   Output: {logits.shape}")  # Should be (4, 20)
    
    # Test embedding extraction (Task 2B)
    embeddings = model(test_eeg, subject_ids=None) # CHANGE 3: New extraction method
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
        # CHANGE 4: Unpack subject_ids
        eeg, _, _, labels, subject_ids = batch
        
        eeg = eeg.to(device)
        labels = labels.to(device)
        subject_ids = subject_ids.to(device) # Move to GPU
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass (Pass subject_ids)
        logits, _ = model(eeg, subject_ids) 
        
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
            # CHANGE 5: Unpack subject_ids
            eeg, _, _, labels, subject_ids = batch
            
            eeg = eeg.to(device)
            labels = labels.to(device)
            subject_ids = subject_ids.to(device)
            
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
# STEP 9: INITIALIZE WANDB
# ============================================================
# wandb.login(key="YOUR_KEY")  # Uncomment if needed

wandb.init(
    project="eeg-classification",
    name="transformer-arch-v1",
    config=config,
    tags=['transformer', 'subject_heads', 'task1']
)

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
    
    # CHANGE 6: Correct usage for verification
    embeddings = model(test_eeg, subject_ids=None)
    
    print(f"✅ Embeddings extracted successfully!")
    print(f"   Shape: {embeddings.shape}")  # Should be (batch_size, 256)
    print(f"   Mean:  {embeddings.mean().item():.4f}")
    print(f"   Std:   {embeddings.std().item():.4f}")

print("\n✅ MODEL IS READY FOR TASK 2B!")
wandb.finish()
