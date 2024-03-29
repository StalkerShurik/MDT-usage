import porespy as ps
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

N = 100
N_samples = 100
shape = [N, N]

def generate_samples(k): #2k will be generated
    dataset = np.zeros(shape=[2*k,2,N,N])
    for i in range(k):
        im1 = ps.generators.blobs(shape=shape, blobiness = [2,1], porosity=0.4)
        im2 = ps.generators.blobs(shape=shape, blobiness = [1,4], porosity=0.6)
        dataset[2*i][0] = im1
        dataset[2*i][1] = 0
        dataset[2*i+1][0] = im2
        dataset[2*i+1][1] = 1
    np.random.shuffle(dataset)
    return dataset
transform = transforms.Compose([transforms.ToTensor()])

import os
import pandas as pd
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, dataset, transform=None, target_transform=None):
        self.img_labels = [dataset[i][1][0][0].astype(int) for i in range(dataset.shape[0])]
        #print("type", self.img_labels)
        self.images = [dataset[i][0].astype(np.single) for i in range(dataset.shape[0])]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx])
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
