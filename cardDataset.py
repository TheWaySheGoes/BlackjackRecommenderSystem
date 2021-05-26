import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.models as models
from torchvision.io import read_image
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import csv
from PIL import Image
import pandas as pd
import numpy as np
from skimage import io, transform
#  https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n = 4
cards = pd.read_csv('dataset/card2/train_labels.csv')
img_name = cards.iloc[n, 0]
suit = cards.iloc[n, 3]
landmarks = cards.iloc[n, 4:]
landmarks = np.asarray(landmarks)
landmarks = landmarks.astype('float').reshape(-1, 2)
print('Image name: {}'.format(img_name))
print('Suit {}'.format(suit))
print('Landmarks shape: {}'.format(landmarks.shape))
print('The 4 Landmarks: {}'.format(landmarks[:4]))


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=50, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

plt.figure()
show_landmarks(io.imread(os.path.join('dataset/card2/train', img_name)),
               landmarks)
plt.show()


class CardDataset(Dataset):
    

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.cards = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.cards)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.cards.iloc[idx, 0])

        image = io.imread(img_name)
       # suit = self.cards.iloc[idx, 3]
        landmarks = self.cards.iloc[idx, 4:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}
        #sample = {'image': image,'suit': suit , 'landmarks': landmarks}
        if self.transform:
            sample = self.transform(sample)

        return sample

"""
Insantiate class and look through data samples.
Prints the first 4 samples.

card_dataset = CardDataset(csv_file='dataset/card2/train_labels.csv',
                                    root_dir='dataset/card2/train')

fig = plt.figure()

for i in range(len(card_dataset)):
    sample = card_dataset[i]
   # print("sample", sample)
    print(i, sample['image'].shape, sample['landmarks'].shape)
    
    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break
    
"""