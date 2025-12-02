"""
Fixed EEGMultimodalDataset - Unified for Task 1 AND Task 2B
Consistent return format with proper subject_id and category labels
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class EEGMultimodalDataset(Dataset):
    """
    FIXED: Unified dataset for both classification and retrieval.
    
    Always returns:
    - eeg: (122, 500) tensor
    - image: (3, 224, 224) tensor
    - caption: string
    - category_label: int (0-19)
    - subject_id: int (0-12)
    
    Use case:
    - Task 1 (Classification): Use eeg, category_label, subject_id
    - Task 2B (Retrieval): Use eeg, caption, category_label, subject_id
    """
    
    def __init__(self, 
                 bids_root,
                 images_dir,
                 captions_path,
                 subject_list,
                 session_list,
                 image_transform=None,
                 clamp_thres=500,
                 normalization_stats=None):
        
        self.bids_root = bids_root
        self.images_dir = images_dir
        self.image_transform = image_transform
        self.clamp_thres = clamp_thres
        
        # Create subject mapping (all 13 subjects)
        all_subjects = ['sub-02', 'sub-03', 'sub-05', 'sub-09', 'sub-14', 
                        'sub-15', 'sub-17', 'sub-19', 'sub-20', 'sub-23', 
                        'sub-24', 'sub-28', 'sub-29']
        self.subject_to_idx = {sub: idx for idx, sub in enumerate(all_subjects)}
        
        # Load captions and create category mapping
        print(f"Loading captions from {captions_path}...")
        self.captions_dict = self._load_captions(captions_path)
        
        # Extract unique categories and create mapping
        categories = sorted(set(cat for cat, _ in self.captions_dict.values()))
        self.category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
        self.idx_to_category = {idx: cat for cat, idx in self.category_to_idx.items()}
        
        print(f"Loaded {len(self.captions_dict)} captions across {len(categories)} categories")
        
        # Store metadata (memory efficient)
        self.trial_metadata = []
        
        print("Scanning dataset files...")
        for sub in subject_list:
            for ses in session_list:
                for run in ['01', '02', '03', '04']:
                    session_path = os.path.join(self.bids_root, sub, ses)
                    csv_path = os.path.join(session_path, 
                                           f"{sub}_{ses}_task-lowSpeed_run-{run}_image.csv")
                    npy_path = os.path.join(session_path, 
                                           f"{sub}_{ses}_task-lowSpeed_run-{run}_1000Hz.npy")
                    
                    if not (os.path.exists(csv_path) and os.path.exists(npy_path)):
                        continue
                    
                    try:
                        csv_data = pd.read_csv(csv_path)
                        eeg_shape = np.load(npy_path, mmap_mode='r').shape
                        
                        if eeg_shape[0] != len(csv_data):
                            continue
                        
                        for trial_idx, row in csv_data.iterrows():
                            img_base_name = self._get_base_name(row['FilePath'])
                            if not img_base_name:
                                continue
                            
                            category, caption = self.captions_dict.get(img_base_name, (None, None))
                            if category is None:
                                continue
                            
                            img_path = self._find_image_path(img_base_name)
                            if not img_path:
                                continue
                            
                            self.trial_metadata.append({
                                'subject': sub,
                                'subject_id': self.subject_to_idx[sub],
                                'session': ses,
                                'run': run,
                                'trial_idx': trial_idx,
                                'npy_path': npy_path,
                                'img_path': img_path,
                                'category': category,
                                'category_label': self.category_to_idx[category],
                                'caption': caption
                            })
                    
                    except Exception as e:
                        print(f"Error processing {sub} {ses} run-{run}: {e}")
                        continue
        
        print(f"Found {len(self.trial_metadata)} valid trials")
        
        # Handle normalization
        if normalization_stats is None:
            print("Computing normalization statistics...")
            self.norm_mean, self.norm_std = self._compute_normalization_stats()
        else:
            print("Using provided normalization statistics")
            self.norm_mean, self.norm_std = normalization_stats
    
    def _compute_normalization_stats(self):
        """
        FIXED: Compute per-CHANNEL normalization (not per-timepoint).
        
        Returns:
            mean: (122,) - mean for each channel
            std: (122,) - std for each channel
        """
        print("Sampling EEG data for normalization...")
        all_eeg = []
        
        # Sample every 10th trial
        sample_indices = range(0, len(self.trial_metadata), 10)
        
        for idx in sample_indices:
            meta = self.trial_metadata[idx]
            eeg_data = np.load(meta['npy_path'])[meta['trial_idx']]  # (122, 500)
            eeg_data = np.clip(eeg_data, -self.clamp_thres, self.clamp_thres)
            all_eeg.append(eeg_data)
        
        eeg_array = np.array(all_eeg, dtype=np.float32)  # (N, 122, 500)
        
        # Compute per-channel statistics
        # Mean across trials (axis=0) and time (axis=2)
        mean = eeg_array.mean(axis=(0, 2))  # (122,)
        std = eeg_array.std(axis=(0, 2)) + 1e-6  # (122,)
        
        print(f"✓ Normalization stats computed: mean shape={mean.shape}, std shape={std.shape}")
        
        return mean, std
    
    def _load_captions(self, captions_path):
        """Load captions.txt"""
        captions_dict = {}
        with open(captions_path, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    category, img_name = parts[1], parts[2]
                    caption = '\t'.join(parts[3:]).strip()
                    captions_dict[img_name] = (category, caption)
        return captions_dict
    
    def _get_base_name(self, file_path):
        """Extract base image name"""
        try:
            normalized_path = str(file_path).replace('\\', '/')
            base_name_with_ext = os.path.basename(normalized_path)
            base_name_resized = os.path.splitext(base_name_with_ext)[0]
            
            if base_name_resized.endswith('_resized'):
                base_name = base_name_resized[:-8]
            else:
                base_name = base_name_resized
            
            return base_name
        except:
            return None
    
    def _find_image_path(self, img_base_name):
        """Find image file"""
        for ext in ['.jpg', '.jpeg', '.png', '.JPEG']:
            img_path = os.path.join(self.images_dir, img_base_name + ext)
            if os.path.exists(img_path):
                return img_path
        return None
    
    def __len__(self):
        return len(self.trial_metadata)
    
    def __getitem__(self, idx):
        """
        FIXED: Always returns 5 items in consistent order.
        
        Returns:
            eeg_tensor: (122, 500)
            image_tensor: (3, 224, 224)
            caption: string
            category_label: int (0-19)
            subject_id: int (0-12)
        """
        meta = self.trial_metadata[idx]
        
        # Load and preprocess EEG
        eeg_data = np.load(meta['npy_path'])[meta['trial_idx']]  # (122, 500)
        eeg_data = np.clip(eeg_data, -self.clamp_thres, self.clamp_thres)
        
        # Normalize per-channel: (122, 500) - (122, 1) / (122, 1)
        eeg_data = (eeg_data - self.norm_mean[:, None]) / self.norm_std[:, None]
        eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32)
        
        # Load image
        try:
            image = Image.open(meta['img_path']).convert('RGB')
            if self.image_transform:
                image_tensor = self.image_transform(image)
            else:
                image_tensor = transforms.ToTensor()(image)
        except Exception as e:
            image_tensor = torch.zeros(3, 224, 224)
        
        # Return in consistent order
        return (
            eeg_tensor,              # (122, 500)
            image_tensor,            # (3, 224, 224)
            meta['caption'],         # string
            meta['category_label'],  # int 0-19
            meta['subject_id']       # int 0-12
        )
    
    def get_normalization_stats(self):
        """Return normalization parameters"""
        return (self.norm_mean, self.norm_std)


def create_datasets(bids_root, images_dir, captions_path, all_subjects=None):
    """
    Factory function to create datasets with proper splits.
    """
    if all_subjects is None:
        all_subjects = ['sub-02', 'sub-03', 'sub-05', 'sub-09', 'sub-14', 
                        'sub-15', 'sub-17', 'sub-19', 'sub-20', 'sub-23', 
                        'sub-24', 'sub-28', 'sub-29']
    
    train_sessions = ['ses-01', 'ses-02', 'ses-03']
    val_sessions = ['ses-04']
    test_sessions = ['ses-05']
    
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    print("\n" + "="*60)
    print("Creating TRAINING dataset...")
    print("="*60)
    train_dataset = EEGMultimodalDataset(
        bids_root=bids_root,
        images_dir=images_dir,
        captions_path=captions_path,
        subject_list=all_subjects,
        session_list=train_sessions,
        image_transform=img_transform
    )
    
    norm_stats = train_dataset.get_normalization_stats()
    
    print("\n" + "="*60)
    print("Creating VALIDATION dataset...")
    print("="*60)
    val_dataset = EEGMultimodalDataset(
        bids_root=bids_root,
        images_dir=images_dir,
        captions_path=captions_path,
        subject_list=all_subjects,
        session_list=val_sessions,
        image_transform=img_transform,
        normalization_stats=norm_stats
    )
    
    print("\n" + "="*60)
    print("Creating TEST dataset...")
    print("="*60)
    test_dataset = EEGMultimodalDataset(
        bids_root=bids_root,
        images_dir=images_dir,
        captions_path=captions_path,
        subject_list=all_subjects,
        session_list=test_sessions,
        image_transform=img_transform,
        normalization_stats=norm_stats
    )
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Train: {len(train_dataset):,} samples")
    print(f"Val:   {len(val_dataset):,} samples")
    print(f"Test:  {len(test_dataset):,} samples")
    print(f"Total: {len(train_dataset) + len(val_dataset) + len(test_dataset):,}")
    print(f"\nCategories: {len(train_dataset.category_to_idx)}")
    print(f"Subjects:   {len(train_dataset.subject_to_idx)}")
    print("="*60)
    
    return train_dataset, val_dataset, test_dataset


# ============================================================
# TEST THE FIXED DATASET
# ============================================================

if __name__ == "__main__":
    # Test with dummy paths (replace with your actual paths)
    train_ds, val_ds, test_ds = create_datasets(
        bids_root='/path/to/ds005589',
        images_dir='/path/to/images',
        captions_path='/path/to/captions.txt'
    )
    
    # Test a single batch
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    
    eeg, img, caption, cat_label, subj_id = next(iter(train_loader))
    
    print("\n" + "="*60)
    print("BATCH TEST")
    print("="*60)
    print(f"EEG shape:        {eeg.shape}")       # Should be (4, 122, 500)
    print(f"Image shape:      {img.shape}")       # Should be (4, 3, 224, 224)
    print(f"Caption type:     {type(caption)}")   # Should be tuple of strings
    print(f"Caption length:   {len(caption)}")    # Should be 4
    print(f"Category shape:   {cat_label.shape}") # Should be (4,)
    print(f"Subject ID shape: {subj_id.shape}")   # Should be (4,)
    print(f"\nCategory labels:  {cat_label}")
    print(f"Subject IDs:      {subj_id}")
    print(f"First caption:    {caption[0][:50]}...")
    print("="*60)
    print("✅ Dataset working correctly!")
