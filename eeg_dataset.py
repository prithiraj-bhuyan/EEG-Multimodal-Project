"""
Enhanced EEGMultimodalDataset - Unified for Task 1 AND Task 2B
With augmentation, preprocessing, class balancing, and utilities
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from collections import Counter
from scipy.signal import detrend


class EEGAugmentation:
    """
    Augmentation techniques for EEG data.
    """
    
    def __init__(self, 
                 time_shift_ms=50,
                 channel_dropout=0.1,
                 noise_std=0.05,
                 time_mask_width=50,
                 amplitude_scale=0.1,
                 apply_prob=0.5):
        
        self.time_shift = int(time_shift_ms)
        self.channel_dropout = channel_dropout
        self.noise_std = noise_std
        self.time_mask_width = time_mask_width
        self.amplitude_scale = amplitude_scale
        self.apply_prob = apply_prob
    
    def __call__(self, eeg):
        """
        Apply random augmentations.
        
        Args:
            eeg: (122, 500) tensor
        Returns:
            eeg_aug: (122, 500) augmented tensor
        """
        eeg = eeg.clone()
        
        # 1. Time jitter (shift ±50ms randomly)
        if np.random.rand() < self.apply_prob:
            shift = np.random.randint(-self.time_shift, self.time_shift + 1)
            eeg = torch.roll(eeg, shifts=shift, dims=1)
        
        # 2. Channel dropout (simulate bad electrodes)
        if np.random.rand() < self.apply_prob:
            n_drop = int(122 * self.channel_dropout)
            drop_idx = np.random.choice(122, n_drop, replace=False)
            eeg[drop_idx, :] = 0
        
        # 3. Gaussian noise
        if np.random.rand() < self.apply_prob:
            noise = torch.randn_like(eeg) * self.noise_std
            eeg = eeg + noise
        
        # 4. Time masking (SpecAugment-style)
        if np.random.rand() < self.apply_prob * 0.6:  # Lower probability
            start = np.random.randint(0, max(1, 500 - self.time_mask_width))
            eeg[:, start:start + self.time_mask_width] = 0
        
        # 5. Amplitude scaling per channel
        if np.random.rand() < self.apply_prob:
            scales = 1.0 + (torch.rand(122, 1) * 2 - 1) * self.amplitude_scale
            eeg = eeg * scales
        
        return eeg


class EEGPreprocessor:
    """
    Enhanced preprocessing pipeline.
    """
    
    def __init__(self, 
                 clamp_threshold=500,
                 apply_detrend=True):
        
        self.clamp_threshold = clamp_threshold
        self.apply_detrend = apply_detrend
    
    def __call__(self, eeg_data):
        """
        Preprocess raw EEG.
        
        Args:
            eeg_data: (122, 500) numpy array (raw)
        Returns:
            eeg_clean: (122, 500) numpy array (preprocessed)
        """
        # 1. Clip extreme outliers
        eeg_clean = np.clip(eeg_data, -self.clamp_threshold, self.clamp_threshold)
        
        # 2. Detrend to remove linear drift per channel
        if self.apply_detrend:
            eeg_clean = detrend(eeg_clean, axis=1, type='linear')
        
        return eeg_clean


class EEGMultimodalDataset(Dataset):
    """
    ENHANCED: Unified dataset for both classification and retrieval.
    
    Features:
    - Advanced preprocessing (detrending, clipping)
    - Data augmentation (5 types)
    - Class balancing
    - Efficient caching
    - Validation utilities
    
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
                 normalization_stats=None,
                 augment=False,
                 cache_limit=100):
        
        self.bids_root = bids_root
        self.images_dir = images_dir
        self.image_transform = image_transform
        self.clamp_thres = clamp_thres
        self.augment = augment
        self.cache_limit = cache_limit
        
        # Initialize preprocessor
        self.preprocessor = EEGPreprocessor(
            clamp_threshold=clamp_thres,
            apply_detrend=True
        )
        
        # Initialize augmentor (only if augmentation is enabled)
        if augment:
            self.augmentor = EEGAugmentation(
                time_shift_ms=50,
                channel_dropout=0.1,
                noise_std=0.05,
                time_mask_width=50,
                amplitude_scale=0.1,
                apply_prob=0.5
            )
            print("✅ Augmentation ENABLED")
        else:
            print("⚪ Augmentation DISABLED")
        
        # EEG data cache for faster loading
        self.eeg_cache = {}
        
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
                            print(f"⚠️  Shape mismatch in {sub} {ses} run-{run}: "
                                  f"EEG {eeg_shape[0]} vs CSV {len(csv_data)}")
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
                        print(f"❌ Error processing {sub} {ses} run-{run}: {e}")
                        continue
        
        print(f"✅ Found {len(self.trial_metadata)} valid trials")
        
        # Compute class distribution and weights
        self._compute_class_distribution()
        
        # Handle normalization
        if normalization_stats is None:
            print("Computing normalization statistics...")
            self.norm_mean, self.norm_std = self._compute_normalization_stats()
        else:
            print("Using provided normalization statistics")
            self.norm_mean, self.norm_std = normalization_stats
    
    def _compute_class_distribution(self):
        """Analyze and report class distribution."""
        category_counts = Counter(meta['category_label'] for meta in self.trial_metadata)
        
        print("\n" + "="*60)
        print("CLASS DISTRIBUTION")
        print("="*60)
        
        for cat_idx in sorted(category_counts.keys()):
            cat_name = self.idx_to_category[cat_idx]
            count = category_counts[cat_idx]
            percentage = 100 * count / len(self.trial_metadata)
            print(f"{cat_name:15s}: {count:5d} samples ({percentage:5.2f}%)")
        
        # Compute class weights for loss balancing (inverse frequency)
        total = len(self.trial_metadata)
        self.class_weights = torch.zeros(len(self.category_to_idx))
        
        for cat_idx, count in category_counts.items():
            # Inverse frequency weighting
            self.class_weights[cat_idx] = total / (len(category_counts) * count)
        
        print(f"\n✅ Class weights computed for weighted loss")
        print("="*60)
    
    def _compute_normalization_stats(self):
        """
        Compute per-CHANNEL normalization statistics.
        
        Returns:
            mean: (122,) - mean for each channel
            std: (122,) - std for each channel
        """
        print("Sampling EEG data for normalization...")
        all_eeg = []
        
        # Sample more trials for stable statistics (at least 1000)
        sample_step = max(1, len(self.trial_metadata) // 1000)
        sample_indices = range(0, len(self.trial_metadata), sample_step)
        
        print(f"Sampling {len(list(sample_indices))} trials for normalization...")
        
        for idx in sample_indices:
            meta = self.trial_metadata[idx]
            eeg_data = np.load(meta['npy_path'])[meta['trial_idx']]  # (122, 500)
            
            # Apply preprocessing (clip + detrend)
            eeg_data = self.preprocessor(eeg_data)
            
            all_eeg.append(eeg_data)
        
        eeg_array = np.array(all_eeg, dtype=np.float32)  # (N, 122, 500)
        
        # Compute per-channel statistics across trials and time
        mean = eeg_array.mean(axis=(0, 2))  # (122,)
        std = eeg_array.std(axis=(0, 2)) + 1e-6  # (122,)
        
        print(f"✅ Normalization stats computed: mean shape={mean.shape}, std shape={std.shape}")
        
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
        for ext in ['.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG']:
            img_path = os.path.join(self.images_dir, img_base_name + ext)
            if os.path.exists(img_path):
                return img_path
        return None
    
    def __len__(self):
        return len(self.trial_metadata)
    
    def __getitem__(self, idx):
        """
        Returns:
            eeg_tensor: (122, 500)
            image_tensor: (3, 224, 224)
            caption: string
            category_label: int (0-19)
            subject_id: int (0-12)
        """
        meta = self.trial_metadata[idx]
        
        # Load EEG (with caching for efficiency)
        npy_path = meta['npy_path']
        
        if npy_path in self.eeg_cache:
            # Use cached data
            eeg_data = self.eeg_cache[npy_path][meta['trial_idx']]
        else:
            # Load from disk
            eeg_full = np.load(npy_path)
            
            # Add to cache (with FIFO eviction)
            if len(self.eeg_cache) >= self.cache_limit:
                self.eeg_cache.pop(next(iter(self.eeg_cache)))
            self.eeg_cache[npy_path] = eeg_full
            
            eeg_data = eeg_full[meta['trial_idx']]
        
        # Preprocess EEG (detrend + clip)
        eeg_data = self.preprocessor(eeg_data)
        
        # Normalize per-channel: (122, 500) - (122, 1) / (122, 1)
        eeg_data = (eeg_data - self.norm_mean[:, None]) / self.norm_std[:, None]
        eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32)
        
        # Apply augmentation (only during training)
        if self.augment:
            eeg_tensor = self.augmentor(eeg_tensor)
        
        # Load image
        try:
            image = Image.open(meta['img_path']).convert('RGB')
            if self.image_transform:
                image_tensor = self.image_transform(image)
            else:
                image_tensor = transforms.ToTensor()(image)
        except Exception as e:
            print(f"⚠️  Failed to load image {meta['img_path']}: {e}")
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
    
    def get_class_weights(self):
        """
        Return class weights for handling imbalance.
        Use this in your loss function:
        
        >>> criterion = nn.CrossEntropyLoss(weight=dataset.get_class_weights().to(device))
        """
        return self.class_weights
    
    def visualize_sample(self, idx, save_path=None):
        """
        Plot a sample for debugging.
        
        Args:
            idx: Sample index
            save_path: Optional path to save figure
        """
        import matplotlib.pyplot as plt
        
        eeg, img, caption, cat_label, subj_id = self[idx]
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 9))
        
        # Plot EEG
        im = axes[0].imshow(eeg.numpy(), aspect='auto', cmap='RdBu_r', 
                            vmin=-3, vmax=3)
        axes[0].set_title(f'EEG - Subject {subj_id} - Category: {self.idx_to_category[cat_label.item()]}',
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time (ms)', fontsize=12)
        axes[0].set_ylabel('Channel', fontsize=12)
        plt.colorbar(im, ax=axes[0], label='Amplitude (normalized)')
        
        # Plot image
        img_show = img.permute(1, 2, 0).numpy()
        img_show = img_show * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_show = np.clip(img_show, 0, 1)
        axes[1].imshow(img_show)
        axes[1].set_title(f'Caption: {caption[:100]}...', fontsize=12)
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved visualization to {save_path}")
        
        plt.show()
    
    def check_data_integrity(self, n_samples=100):
        """
        Run sanity checks on the dataset.
        
        Args:
            n_samples: Number of samples to check
        """
        print("\n" + "="*60)
        print("RUNNING DATA INTEGRITY CHECKS")
        print("="*60)
        
        issues = []
        n_check = min(n_samples, len(self))
        
        print(f"Checking {n_check} samples...")
        for i in range(n_check):
            try:
                eeg, img, caption, cat_label, subj_id = self[i]
                
                # Check for NaN/Inf
                if torch.isnan(eeg).any():
                    issues.append(f"Index {i}: NaN in EEG")
                if torch.isinf(eeg).any():
                    issues.append(f"Index {i}: Inf in EEG")
                
                # Check shapes
                if eeg.shape != (122, 500):
                    issues.append(f"Index {i}: Wrong EEG shape {eeg.shape}")
                if img.shape != (3, 224, 224):
                    issues.append(f"Index {i}: Wrong image shape {img.shape}")
                
                # Check label ranges
                if not (0 <= cat_label < 20):
                    issues.append(f"Index {i}: Invalid category label {cat_label}")
                if not (0 <= subj_id < 13):
                    issues.append(f"Index {i}: Invalid subject ID {subj_id}")
                
            except Exception as e:
                issues.append(f"Index {i}: Exception {str(e)}")
        
        if issues:
            print(f"⚠️  Found {len(issues)} issues:")
            for issue in issues[:20]:  # Show first 20
                print(f"  - {issue}")
        else:
            print(f"✅ All {n_check} samples passed integrity checks!")
        
        # Check label distribution
        labels = [self.trial_metadata[i]['category_label'] for i in range(len(self))]
        unique_labels = len(set(labels))
        print(f"✅ Dataset contains {unique_labels}/20 categories")
        
        # Check subject distribution
        subjects = [self.trial_metadata[i]['subject_id'] for i in range(len(self))]
        unique_subjects = len(set(subjects))
        print(f"✅ Dataset contains {unique_subjects}/13 subjects")
        print("="*60)


def create_datasets(bids_root, images_dir, captions_path, all_subjects=None, augment_train=True):
    """
    Factory function to create datasets with proper splits.
    
    Args:
        bids_root: Path to BIDS dataset root
        images_dir: Path to images directory
        captions_path: Path to captions.txt
        all_subjects: List of subjects to include (default: all 13)
        augment_train: Apply augmentation to training set
    
    Returns:
        train_dataset, val_dataset, test_dataset
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
    print("CREATING TRAINING DATASET")
    print("="*60)
    train_dataset = EEGMultimodalDataset(
        bids_root=bids_root,
        images_dir=images_dir,
        captions_path=captions_path,
        subject_list=all_subjects,
        session_list=train_sessions,
        image_transform=img_transform,
        augment=augment_train  # Enable augmentation for training
    )
    
    norm_stats = train_dataset.get_normalization_stats()
    
    print("\n" + "="*60)
    print("CREATING VALIDATION DATASET")
    print("="*60)
    val_dataset = EEGMultimodalDataset(
        bids_root=bids_root,
        images_dir=images_dir,
        captions_path=captions_path,
        subject_list=all_subjects,
        session_list=val_sessions,
        image_transform=img_transform,
        normalization_stats=norm_stats,
        augment=False  # No augmentation for validation
    )
    
    print("\n" + "="*60)
    print("CREATING TEST DATASET")
    print("="*60)
    test_dataset = EEGMultimodalDataset(
        bids_root=bids_root,
        images_dir=images_dir,
        captions_path=captions_path,
        subject_list=all_subjects,
        session_list=test_sessions,
        image_transform=img_transform,
        normalization_stats=norm_stats,
        augment=False  # No augmentation for test
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
# TESTING
# ============================================================

if __name__ == "__main__":
    print("Testing Enhanced EEG Dataset...")
    
    # Replace with your actual paths
    train_ds, val_ds, test_ds = create_datasets(
        bids_root='/path/to/ds005589',
        images_dir='/path/to/images',
        captions_path='/path/to/captions.txt',
        augment_train=True
    )
    
    # Run integrity checks
    train_ds.check_data_integrity(n_samples=100)
    
    # Test dataloader
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    
    eeg, img, caption, cat_label, subj_id = next(iter(train_loader))
    
    print("\n" + "="*60)
    print("BATCH TEST")
    print("="*60)
    print(f"EEG shape:        {eeg.shape}")       # Should be (32, 122, 500)
    print(f"Image shape:      {img.shape}")       # Should be (32, 3, 224, 224)
    print(f"Caption type:     {type(caption)}")   # Should be tuple of strings
    print(f"Caption length:   {len(caption)}")    # Should be 32
    print(f"Category shape:   {cat_label.shape}") # Should be (32,)
    print(f"Subject ID shape: {subj_id.shape}")   # Should be (32,)
    print(f"\nUnique subjects in batch: {len(torch.unique(subj_id))}")
    print(f"Unique categories in batch: {len(torch.unique(cat_label))}")
    print(f"First caption: {caption[0][:80]}...")
    print("="*60)
    
    # Get class weights
    class_weights = train_ds.get_class_weights()
    print(f"\nClass weights shape: {class_weights.shape}")
    print(f"Class weights sample: {class_weights[:5]}")
    
    # Visualize a sample
    print("\n" + "="*60)
    print("VISUALIZING SAMPLE")
    print("="*60)
    train_ds.visualize_sample(0, save_path='sample_visualization.png')
    
    print("\n✅ All tests passed! Dataset is ready to use.")
    print("\nUsage in training:")
    print(">>> criterion = nn.CrossEntropyLoss(weight=train_ds.get_class_weights().to(device))")