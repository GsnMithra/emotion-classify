import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FERDataset(Dataset):
    def __init__(self, csv_file, img_dir, datatype, transform=None):
        self.csv_file = pd.read_csv(csv_file)
        self.labels = self.csv_file['emotion']
        self.img_dir = img_dir
        self.transform = transform
        self.datatype = datatype

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx

        img = Image.open(f'{os.path.join(self.img_dir, self.datatype)}{idx}.jpg')
        labels = np.array(self.labels[idx])
        labels = torch.from_numpy(labels).long()

        if self.transform:
            img = self.transform(img)
        return (img, labels)
