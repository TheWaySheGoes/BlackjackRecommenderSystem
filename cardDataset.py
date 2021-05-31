import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.models as models
from torchvision.io import read_image
from torchvision import datasets, models, transforms
import matplotlib
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
#cards = pd.read_csv('dataset/card2/train_labels.csv')
cards = pd.read_csv('new_dataset/card2/new_train_label.csv')
img_name = cards.iloc[n, 0]
suit = cards.iloc[n, 3]
labels = cards.iloc[n, -1]
landmarks = cards.iloc[n, 4:8]
landmarks = np.asarray(landmarks)
landmarks = landmarks.astype('float').reshape(-1, 2)
print('Image name: {}'.format(img_name))
print('Suit {}'.format(suit))
print('Label {}'.format(labels))
print('Landmarks shape: {}'.format(landmarks.shape))
print('The 4 Landmarks: {}'.format(landmarks[:4]))


def show_landmarks(image, landmarks, suit, labels=None):
    """Show image with landmarks"""
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.imshow(image)
    ax.text(2,-40,'Card suit: {}'.format(suit), fontsize=10 )
    ax.text(2,-10,'Card value: {}'.format(labels), fontsize=10 )
    landmarks_test = []
    #landmarks_test = landmarks[:, 0]
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=50, marker='.', c='r')
    
    plt.pause(0.001)  # pause a bit so that plots are updated

show_landmarks(io.imread(os.path.join('new_dataset/card2/train', img_name)),
               landmarks, suit, labels)
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

        #image = io.imread(img_name)
        image = read_image(img_name) # convert to tensor
        suit = self.cards.iloc[idx, 3]
        labels = self.cards.iloc[idx, -1]
        landmarks = self.cards.iloc[idx, 4:8]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        #sample = {'image': image, 'landmarks': landmarks}
        #sample = {'image': image,'suit': suit , 'landmarks': landmarks}
        sample = {'image': image, 'landmarks': landmarks, 'suit': suit, 'labels': labels}
        #print("sample e e  e", sample)
        if self.transform:
            #sample = self.transform(sample)
            image = self.transform(image)

        #return sample
        return image

"""
Insantiate class and look through data samples.
Prints the first 4 samples.
"""

data_transforms = {
    
    'train':
    transforms.Compose([
    torchvision.transforms.RandomCrop((224,224), pad_if_needed = True),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomInvert(),
    transforms.RandomRotation(45),
    #transforms.ToTensor(),
    ]),
    
    'test':transforms.Compose([
        torchvision.transforms.RandomCrop((224,224)),
        transforms.ToTensor(),
        ]),
}



