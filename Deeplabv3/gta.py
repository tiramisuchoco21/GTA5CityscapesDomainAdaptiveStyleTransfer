import os
import torch
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms

class GTA(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        
        self.color_map = {
            (128, 64, 128): 0,   # Road
            (244, 35, 232): 1,   # Sidewalk
            (70, 70, 70): 2,     # Building
            (102, 102, 156): 3,  # Wall
            (190, 153, 153): 4,  # Fence
            (153, 153, 153): 5,  # Pole
            (250, 170, 30): 6,   # Traffic Light
            (220, 220, 0): 7,    # Traffic Sign
            (107, 142, 35): 8,   # Vegetation
            (152, 251, 152): 9,  # Terrain
            (70, 130, 180): 10,  # Sky
            (220, 20, 60): 11,   # Person
            (255, 0, 0): 12,     # Rider
            (0, 0, 142): 13,     # Car
            (0, 0, 70): 14,      # Truck
            (0, 60, 100): 15,    # Bus
            (0, 80, 100): 16,    # Train
            (0, 0, 230): 17,     # Motorcycle
            (119, 11, 32): 18    # Bicycle
        }
        
        self.image_dir = os.path.join(root, 'images') 
        self.label_dir = os.path.join(root, 'labels')
        
        self.files = sorted(os.listdir(self.image_dir))
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        img_name = self.files[index]
        
        # Load Image
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Load Label (Assume same filename)
        lbl_path = os.path.join(self.label_dir, img_name)
        label = Image.open(lbl_path).convert('RGB')
        
        # Encode Label (RGB -> Train ID)
        label = self.encode_target(label)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
        
    def encode_target(self, target):
        """
        Convert RGB image to 2D ID tensor
        """
        target = np.array(target)
        # Initialize mask with 255 (Ignore)
        mask = np.full(target.shape[:2], 255, dtype=np.longlong)
        
        for color, class_id in self.color_map.items():
            # Find pixels matching the color
            matches = np.all(target == np.array(color), axis=-1)
            mask[matches] = class_id
            
        return torch.from_numpy(mask)