import os
import lmdb
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import json
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class SpectralAugmentation:
    """
    Data augmentation class for spectral image data.
    
    Supports augmentations for spectral images with shape (channels, height, width).
    Designed for spectral data where:
    - Axis 0: channels (e.g., gv_img, gi_img)
    - Axis 1: height/y-axis (spatial dimension)
    - Axis 2: width/x-axis (wavelength dimension)
    
    Args:
        augmentation_types: List of augmentation types to apply
        shift_x_range: Range for wavelength shifts (default: 20 pixels)
        shift_y_range: Range for y-axis shifts (default: 5 pixels)
        noise_std: Standard deviation for Gaussian noise (default: 0.01)
        flip_prob: Probability of applying vertical flip (default: 0.5)
        deterministic: If True, apply augmentations deterministically based on seed
    """
    
    def __init__(
        self,
        augmentation_types: Optional[List[str]] = None,
        shift_x_range: int = 48,
        shift_y_range: int = 4,
        seed: Optional[int] = None
    ):
        self.augmentation_types = augmentation_types or []
        self.shift_x_range = shift_x_range
        self.shift_y_range = shift_y_range
        self.seed = seed
        
        # Validate augmentation types
        valid_types = ['flip_vertical', 'shift_x', 'shift_y']
        for aug_type in self.augmentation_types:
            if aug_type not in valid_types:
                raise ValueError(
                    f"Unknown augmentation type: {aug_type}. "
                    f"Valid types are: {valid_types}"
                )
    
    def apply_single_augmentation(self, image_data: np.ndarray, aug_type: str, seed: Optional[int] = None) -> np.ndarray:
        """
        Apply a single augmentation type to image data.
        
        Args:
            image_data: numpy array of shape (channels, height, width)
            aug_type: Type of augmentation to apply (must be one of: 'flip_vertical', 'shift_x', 'shift_y')
            seed: Optional seed for deterministic augmentation
        
        Returns:
            Augmented image data
        """
        # Handle None case (should not happen, but be safe)
        if aug_type is None:
            return image_data.copy()
        
        # Validate augmentation type
        valid_types = ['flip_vertical', 'shift_x', 'shift_y']
        if aug_type not in valid_types:
            raise ValueError(
                f"Unknown augmentation type: {aug_type}. "
                f"Valid types are: {valid_types}"
            )
        
        augmented = image_data.copy()
        
        # if seed is not None:
            # rng_state = np.random.get_state()
            # np.random.seed(seed)
        
        try:
            if aug_type == 'flip_vertical':
                augmented = np.flip(augmented, axis=1)
            
            elif aug_type == 'shift_x':
                shift_amount = np.random.randint(-self.shift_x_range, self.shift_x_range + 1)
                if shift_amount != 0:
                    augmented = np.roll(augmented, shift_amount, axis=2)
            
            elif aug_type == 'shift_y':
                shift_amount = np.random.randint(-self.shift_y_range, self.shift_y_range + 1)
                if shift_amount != 0:
                    augmented = np.roll(augmented, shift_amount, axis=1)
        
        except Exception as e:
            print(f"Error applying augmentation {aug_type}: {e}")
            raise
        
        # finally:
        #     if seed is not None:
        #         np.random.set_state(rng_state)
        
        augmented = augmented.copy()
        
        return augmented
    
    def __call__(self, image_data: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Apply augmentations to image data.
        
        Args:
            image_data: numpy array of shape (channels, height, width)
            seed: Optional seed for deterministic augmentation
        
        Returns:
            Augmented image data
        """
        # Make a copy to avoid modifying original
        augmented = image_data.copy()
        
        for aug_type in self.augmentation_types:
            augmented = self.apply_single_augmentation(augmented, aug_type, seed=seed)
        
        return augmented
    
    def __repr__(self) -> str:
        return (
            f"SpectralAugmentation("
            f"types={self.augmentation_types}, "
            f"shift_x_range={self.shift_x_range}, "
            f"shift_y_range={self.shift_y_range}, "
            f"seed={self.seed})"
        )


class LMDBDataset(Dataset):
    """
    PyTorch Dataset for reading data from multiple LMDB databases.
    
    Args:
        lmdb_dir: Path to directory containing LMDB databases
        transform: Optional transform to apply to data samples
        cache_size: Size of LMDB cache in bytes (default: 1GB)
        indices: Optional list of indices to use (filters dataset to these indices).
                Can contain duplicates if same sample should appear multiple times with different augmentations.
        augmentation_types: Optional list of augmentation types, one per index in indices.
                          Should have same length as indices. Use 'original' for no augmentation.
                          Valid types: 'original', 'flip_vertical', 'shift_x', 'shift_y', 'noise'
        augmenter: Optional SpectralAugmentation instance (will be created automatically if not provided)
    """
    
    def __init__(
        self, 
        lmdb_dir: str = "../sls/output/lmdb_data",
        transform: Optional[callable] = None,
        cache_size: int = 1024**3,
        indices: Optional[List[int]] = None,
        augmentation_types: Optional[List[str]] = None,
        augmenter: Optional[SpectralAugmentation] = None,
    ):
        self.lmdb_dir = Path(lmdb_dir)
        self.transform = transform
        self.cache_size = cache_size
        
        # Store indices (if None, use all indices)
        self.indices = indices
        
        # Store augmentation types - should match length of indices if provided
        if augmentation_types is not None:
            if indices is not None and len(augmentation_types) != len(indices):
                raise ValueError(
                    f"augmentation_types length ({len(augmentation_types)}) must match "
                    f"indices length ({len(indices)})"
                )
            # Validate augmentation types
            valid_types = ['original', 'flip_vertical', 'shift_x', 'shift_y']
            for aug_type in augmentation_types:
                if aug_type not in valid_types:
                    raise ValueError(
                        f"Unknown augmentation type: {aug_type}. "
                        f"Valid types are: {valid_types}"
                    )
            self.augmentation_types = augmentation_types
        else:
            self.augmentation_types = None
        
        # Create or use provided augmenter
        if augmenter is not None:
            self.augmenter = augmenter
        else:
            self.augmenter = SpectralAugmentation(
                augmentation_types=['flip_vertical', 'shift_x', 'shift_y'],
                shift_x_range=48,
                shift_y_range=4,
                seed=42,
            )
        
        # Find all LMDB databases
        self.lmdb_paths = sorted([
            d for d in self.lmdb_dir.iterdir() 
            if d.is_dir() and d.suffix == '.lmdb'
        ])
        
        if not self.lmdb_paths:
            raise ValueError(f"No LMDB databases found in {self.lmdb_dir}")
        
        # Open all LMDB environments
        self.envs = []
        self.txns = []
        self.cursors = []
        self.db_sizes = []
        
        for lmdb_path in self.lmdb_paths:
            env = lmdb.open(
                str(lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1,
                map_size=self.cache_size
            )
            self.envs.append(env)
            
            # Get database size
            with env.begin(buffers=True) as txn:
                db_size = txn.stat()['entries']
                self.db_sizes.append(db_size)
        
        # Calculate cumulative sizes for indexing
        self.cumulative_sizes = np.cumsum([0] + self.db_sizes)
        self.total_size = sum(self.db_sizes)
        
        # Process indices if provided
        if self.indices is not None:
            # Convert to numpy array for easier handling
            self.indices = np.array(self.indices)
            # Validate indices
            if np.any(self.indices < 0) or np.any(self.indices >= self.total_size):
                raise ValueError(
                    f"Indices must be in range [0, {self.total_size})"
                )
            self.dataset_size = len(self.indices)
        else:
            self.dataset_size = self.total_size
        
        # # print(f"Loaded {len(self.lmdb_paths)} LMDB databases with {self.total_size} total entries")
        # for i, (path, size) in enumerate(zip(self.lmdb_paths, self.db_sizes)):
        #     print(f"  {path.name}: {size} entries")
        
        if self.indices is not None:
            print(f"Using {self.dataset_size} samples from provided indices")
        
        if self.augmentation_types is not None:
            unique_aug_types = set(self.augmentation_types)
            print(f"Augmentation types used: {sorted(unique_aug_types)}")

    def __len__(self) -> int:
        return self.dataset_size
    
    def _get_db_and_local_idx(self, idx: int) -> Tuple[int, int]:
        """Convert global index to (database_index, local_index) pair."""
        if idx < 0 or idx >= self.total_size:
            raise IndexError(f"Index {idx} out of range [0, {self.total_size})")
        
        # Find which database contains this index
        db_idx = np.searchsorted(self.cumulative_sizes[1:], idx, side='right')
        local_idx = idx - self.cumulative_sizes[db_idx]
        
        return db_idx, local_idx
    
    def __getitem__(self, idx: int):
        """
        Retrieve item at given index.
        
        Args:
            idx: Index into the dataset (0 to len(dataset)-1)
        
        Returns:
            Data sample (transformed and augmented if specified)
        """
        # Get the LMDB index for this dataset index
        if self.indices is not None:
            lmdb_idx = self.indices[idx]
        else:
            lmdb_idx = idx
        
        # Get the augmentation type for this dataset index
        if self.augmentation_types is not None:
            aug_type = self.augmentation_types[idx]
            # 'original' means no augmentation
            if aug_type == 'original':
                aug_type = None
        else:
            aug_type = None
        
        db_idx, local_idx = self._get_db_and_local_idx(lmdb_idx)
        
        # Open transaction for this access
        with self.envs[db_idx].begin(buffers=True) as txn:
            cursor = txn.cursor()
            
            # Move cursor to the local index position
            cursor.first()
            for _ in range(local_idx):
                cursor.next()
            
            key, value = cursor.item()
            
            # Convert memoryview to bytes and deserialize
            key_bytes = bytes(key)
            value_bytes = bytes(value)
            
            # Try to unpickle the value (common format for LMDB storage)
            try:
                data = pickle.loads(value_bytes)
            except:
                # If not pickled, return raw bytes
                data = value_bytes
        
        # data is a dict 
        
        # image: 2, 40, 480
        image_data = data['stacked_img']
        infos = data['infos'] # dict
        
        z = infos['REDSHIFT']
        ID = infos['ID']
    
        # Apply transform if provided
        if self.transform is not None:
            image_data = self.transform(image_data)
        
        # Apply augmentation if needed
        if aug_type is not None and self.augmenter is not None:
            # Use deterministic augmentation with seed based on lmdb_idx and aug_type
            # This ensures reproducibility while creating different augmented versions
            image_data = self.augmenter.apply_single_augmentation(image_data, aug_type, seed=None)
        
        image_data = torch.from_numpy(image_data.copy())
        z = torch.tensor(z)
        
        return image_data, z, ID
    
    def close(self):
        """Close all LMDB environments."""
        for env in self.envs:
            env.close()
    
    def __del__(self):
        """Cleanup when dataset is destroyed."""
        self.close()

def create_dataloader(
    lmdb_dir: str = "../sls/output/lmdb_data",
    dataframe_path: str = "../sls/output/lmdb_data/valid_sources.csv",
    batch_size: int = 32,
    num_workers: int = 10,
    train_split: float = 0.8,
    random_seed: int = 42,
    train_augment: bool = True,
    test_augment: bool = False,
    augmentation_types: str = None,
    save_path: str = None,
) -> Tuple[DataLoader, DataLoader]:
    
    dataframe = pd.read_csv(dataframe_path)
    total_size = len(dataframe)
    
    print(f"Total size: {total_size}")
    
    dist = np.loadtxt('../datasets/CSST/1d_points.d')
    interp = interp1d(dist[:, 0], dist[:, 1], kind='linear',
                      bounds_error=False, fill_value="extrapolate")
    
    redshifts = dataframe['REDSHIFT'].values

    # Fast histogram-based density estimation
    # Create bins covering the redshift range
    n_bins = 200  # Adjust based on your data size
    redshift_min, redshift_max = redshifts.min(), redshifts.max()
    # Add small padding to avoid edge effects
    padding = (redshift_max - redshift_min) * 0.01
    bins = np.linspace(redshift_min - padding, redshift_max + padding, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Compute histogram (density)
    hist_counts, _ = np.histogram(redshifts, bins=bins, density=True)

    # Interpolate to get density at each redshift value
    data_density_interp = interp1d(bin_centers, hist_counts, kind='linear',
                                    bounds_error=False, fill_value=1e-10)
    data_density = data_density_interp(redshifts)
    data_density[data_density < 1e-10] = 1e-10  # Avoid division by zero

    # Compute importance weights: target_density / data_density
    target_density = interp(redshifts)
    # target_density[(target_density < 0) & (redshifts > 1.5)] = 0
    target_density[target_density < 0] = 0

    # Importance sampling weights
    importance_weights = target_density / data_density
    importance_weights = importance_weights / importance_weights.sum()

    train_split = 0.8
    test_split = 1 - train_split

    random_seed = 42
    np.random.seed(random_seed)
    test_indices = np.random.choice(
        np.arange(total_size), 
        size=int(test_split * total_size), 
        replace=False, 
        p=importance_weights
    )

    train_indices = np.setdiff1d(np.arange(total_size), test_indices)

    print(len(test_indices))
    print(len(train_indices))

    plt.figure(figsize=(8, 5))
    _ = plt.hist(redshifts, bins=100, histtype='step', label='Total', density=True)
    _ = plt.hist(redshifts[test_indices], bins=100, histtype='step', label='Test', density=True)
    _ = plt.hist(redshifts[train_indices], bins=100, histtype='step', label='Train', density=True)
    # Also plot the target distribution
    plt.plot(dist[:, 0], dist[:, 1] / dist[:, 1].sum() / np.diff(dist[:, 0]).mean(), 
            'k--', label='Target', linewidth=2)
    plt.xlim(0, 1.5)
    plt.legend()
    plt.xlabel('Redshift')
    plt.ylabel('Density')
    plt.title('Redshift Distribution')
    plt.savefig(os.path.join(save_path, 'redshift_distribution.png'))
    
    print(f"Train indices: {len(train_indices)}")
    print(f"Test indices: {len(test_indices)}")
    
    # base_augmentation_types = ['original', 'flip_vertical', 
    #                            'shift_x', 'shift_y', 
    #                            'shift_x', 'shift_y', 
    #                            'shift_x', 'shift_y']
    
    base_augmentation_types = augmentation_types
    
    train_aug_indices = []
    train_aug_types = []
    if train_augment:
        for i in train_indices:
            for aug_type in base_augmentation_types:
                train_aug_indices.append(int(i))
                train_aug_types.append(aug_type)
    else:
        train_aug_indices = train_indices.tolist()
        train_aug_types = None
    
    test_aug_indices = []
    test_aug_types = []
    if test_augment:
        for i in test_indices:
            for aug_type in base_augmentation_types:
                test_aug_indices.append(int(i))
                test_aug_types.append(aug_type)
    else:
        test_aug_indices = test_indices.tolist()
        test_aug_types = None
        
    if save_path is not None:
        with open(os.path.join(save_path, 'data_splits.json'), 'w') as f:
            json.dump({
                'train_indices': train_aug_indices,
                'train_aug_types': train_aug_types,
                'test_indices': test_aug_indices,
                'test_aug_types': test_aug_types,
            }, f, indent=4)
        
    train_dataset = LMDBDataset(
        lmdb_dir=lmdb_dir,
        transform=None,
        indices=train_aug_indices,
        augmentation_types=train_aug_types,
    )
    
    test_dataset = LMDBDataset(
        lmdb_dir=lmdb_dir,
        transform=None,
        indices=test_aug_indices,
        augmentation_types=test_aug_types,
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    return train_dataset, test_dataset
    
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=num_workers,
    #     pin_memory=True,
    # )
    
    # test_dataloader = DataLoader(
    #     test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     pin_memory=True,
    # )
    
    # return train_dataloader, test_dataloader
    
# Example usage
if __name__ == "__main__":
    
    train_dataloader, test_dataloader = create_dataloader()
