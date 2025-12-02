"""
Fixed EEG Encoder Architecture
Compatible with BOTH Task 1 (Classification) and Task 2B (Retrieval)
"""

import torch
import torch.nn as nn


class EEGEncoder(nn.Module):
    """
    EEG Encoder that outputs embeddings.
    Can be used for both classification and retrieval.
    
    Architecture:
    - Input: (batch_size, 122, 500) raw EEG
    - Flatten: (batch_size, 61000)
    - Hidden: (batch_size, 512) with BatchNorm + ReLU
    - Output: (batch_size, 256) embeddings
    """
    def __init__(self, 
                 in_dim=122*500,  # 61,000
                 hid_dim=512, 
                 emb_dim=256):    # Embedding dimension
        super(EEGEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hid_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, 122, 500) EEG data
        
        Returns:
            embeddings: (batch_size, 256) feature embeddings
        """
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)  # (batch_size, 61000)
        embeddings = self.encoder(x_flat)    # (batch_size, 256)
        return embeddings


class EEGClassifier(nn.Module):
    """
    Complete classification model with encoder + classification head.
    
    For Task 1: Use this for training classification.
    For Task 2B: Load the encoder part only.
    """
    def __init__(self, 
                 in_dim=122*500,
                 hid_dim=512,
                 emb_dim=256,
                 num_classes=20):
        super(EEGClassifier, self).__init__()
        
        # Shared encoder
        self.encoder = EEGEncoder(in_dim, hid_dim, emb_dim)
        
        # Classification head
        self.classifier = nn.Linear(emb_dim, num_classes)
    
    def forward(self, x, return_embeddings=False):
        """
        Args:
            x: (batch_size, 122, 500) EEG data
            return_embeddings: If True, return both logits and embeddings
        
        Returns:
            If return_embeddings=False: logits (batch_size, 20)
            If return_embeddings=True: (logits, embeddings)
        """
        embeddings = self.encoder(x)        # (batch_size, 256)
        logits = self.classifier(embeddings) # (batch_size, 20)
        
        if return_embeddings:
            return logits, embeddings
        return logits
    
    def get_embeddings(self, x):
        """
        Extract embeddings without classification.
        Useful for Task 2B retrieval.
        """
        return self.encoder(x)


# ============================================================
# HOW TO USE THIS FOR YOUR PROJECT
# ============================================================

"""
TASK 1 (Classification) - Current Task:
---------------------------------------
# Create model
model = EEGClassifier(num_classes=20)

# Training loop
for batch in train_loader:
    eeg, _, _, labels, _ = batch
    
    logits = model(eeg)  # No softmax here!
    loss = criterion(logits, labels)  # CrossEntropyLoss handles softmax
    
    loss.backward()
    optimizer.step()


TASK 2B (Retrieval) - Future Task:
----------------------------------
# Load the trained encoder
checkpoint = torch.load('checkpoints/best_model.pth')

# Option 1: Load just the encoder
encoder = EEGEncoder()
encoder.load_state_dict(checkpoint['model_state_dict']['encoder'])

# Option 2: Load full model and extract encoder
full_model = EEGClassifier()
full_model.load_state_dict(checkpoint['model_state_dict'])
encoder = full_model.encoder

# Now use encoder for Task 2B
eeg_embeddings = encoder(eeg_data)  # (batch_size, 256)
# Then project to CLIP space with your projection heads
"""


# ============================================================
# CONVERSION SCRIPT FOR YOUR EXISTING CHECKPOINT
# ============================================================

def convert_old_checkpoint_to_new_format(old_checkpoint_path, new_checkpoint_path):
    """
    Convert your existing ModelFC checkpoint to EEGClassifier format.
    
    Your old model structure:
    - model.0.weight: (512, 61000) - First linear layer
    - model.1.weight: (512,) - BatchNorm
    - model.3.weight: (20, 512) - Classification layer
    
    New model structure:
    - encoder.encoder.0.weight: (512, 61000)
    - encoder.encoder.1.weight: (512,)
    - encoder.encoder.4.weight: (256, 512)  ← NEW layer
    - encoder.encoder.5.weight: (256,)      ← NEW layer
    - classifier.weight: (20, 256)
    
    WARNING: This won't work directly because dimensions changed!
    You'll need to RETRAIN with the new architecture.
    """
    import torch
    
    print("❌ CANNOT DIRECTLY CONVERT!")
    print("Your old model: 61000 → 512 → 20")
    print("New model: 61000 → 512 → 256 → 20")
    print()
    print("You need to:")
    print("1. Create new EEGClassifier model")
    print("2. Retrain from scratch (or fine-tune)")
    print("3. This should be FAST since you know it works!")


# ============================================================
# QUICK RETRAIN SCRIPT
# ============================================================

def quick_retrain_with_new_architecture(train_loader, val_loader, 
                                       num_epochs=50, device='cuda'):
    """
    Quick retraining script with new architecture.
    Should reach similar accuracy quickly since data pipeline works.
    """
    import torch.optim as optim
    
    # Create new model
    model = EEGClassifier(
        in_dim=122*500,
        hid_dim=512,
        emb_dim=256,      # New embedding dimension
        num_classes=20
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        for batch in train_loader:
            eeg, _, _, labels, _ = batch
            eeg = eeg.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(eeg)  # (batch_size, 20)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                eeg, _, _, labels, _ = batch
                eeg = eeg.to(device)
                labels = labels.to(device)
                
                logits = model(eeg)
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'checkpoints/best_model_new_arch.pth')
            
            print(f"Epoch {epoch}: Val Acc = {val_acc:.2f}% (Best!)")
        else:
            print(f"Epoch {epoch}: Val Acc = {val_acc:.2f}%")
    
    return model


if __name__ == "__main__":
    # Test the architecture
    model = EEGClassifier()
    
    # Test input
    dummy_eeg = torch.randn(32, 122, 500)
    
    # Test classification
    logits = model(dummy_eeg)
    print(f"Logits shape: {logits.shape}")  # Should be (32, 20)
    
    # Test embeddings extraction
    embeddings = model.get_embeddings(dummy_eeg)
    print(f"Embeddings shape: {embeddings.shape}")  # Should be (32, 256)
    
    print("\n✅ Architecture looks good!")
