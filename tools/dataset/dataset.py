"""Dataset class for Mirflickr image loading.

Handles loading, error handling, and transformation pipeline for training and evaluation.
"""

import os
from typing import Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MirflickrDataset(Dataset):
    """Custom dataset for loading and preprocessing images from a specified directory."""
    
    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None, limit: Optional[int] = None):
        """Initializes the dataset.

        Args:
            root_dir (str): Path to directory containing images.
            transform (Optional[transforms.Compose]): Image transform pipeline.
            limit (Optional[int]): Limit on the number of images to load.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if limit:
            self.image_files = self.image_files[:limit]
            
    def __len__(self) -> int:
        """Returns the number of images in the dataset.

        Returns:
            int: Dataset length.
        """
        return len(self.image_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Loads and transforms an image by its index.

        Args:
            idx (int): Image index.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
