from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment=False):
        self.root_dir = root_dir
        self.augment = augment
        
        # 默认变换
        default_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        
        self.transform =  default_transform
        
        # 支持多目录数据集
        self.image_paths = []
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        if isinstance(root_dir, list):
            for directory in root_dir:
                self._load_images_from_directory(directory, valid_extensions)
        else:
            self._load_images_from_directory(root_dir, valid_extensions)
        
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {root_dir}")
        print(f"Loaded {len(self.image_paths)} images")

    def _load_images_from_directory(self, directory, valid_extensions):
        for fname in os.listdir(directory):
            if os.path.splitext(fname)[1].lower() in valid_extensions:
                self.image_paths.append(os.path.join(directory, fname))

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return torch.rand(3, 224, 224)
    