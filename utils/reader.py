import os 
import cv2 
import torch 
import numpy as np 
from PIL import Image 
from config import configs
from IPython import embed
class SegDataset(torch.utils.data.Dataset):
    def __init__(self,files,phase="train",transforms=None):
        self.files = files
        self.phase = phase
        self.transforms = transforms
        if phase=="test":
            self.image_path = configs.test_folder + "images/"
            self.mask_path = configs.test_folder + "masks/"
        else:
            self.image_path = configs.dataset + "images/"
            self.mask_path = configs.dataset + "masks/"

    def __getitem__(self,index):
        image_name = self.image_path + self.files[index].replace(".npy",".png")
        mask_name = self.mask_path + self.files[index]
        image = cv2.imread(image_name)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raw_mask = np.load(mask_name)
        mask = np.stack([(raw_mask==v) for v in range(configs.num_classes)],axis=-1).astype(float)
        # do augmentation
        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        else:
            img = image
            mask = mask
        mask = mask[0].permute(2, 0, 1)  # 1x4x256x1600
        return img, mask
    def __len__(self):
        return len(self.files)
