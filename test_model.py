import torchvision
from torchvision import transforms
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#model=torch.load('BlackjackRecommenderSystem-main/models/model_resnet.m')
#device = torch.device("cpu")
class CardDataset(Dataset):
    

    def __init__(self,  transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform=transform
        csv_path='new_dataset\\card1\\card1_new_train_label.csv'
        self.img_path='new_dataset\\card1\\train\\'
        cards_csv = pd.read_csv(csv_path)
        self.file_names=cards_csv['filename']
        self.labels=cards_csv['labels']
        self.klass=cards_csv['class']
        self.xmin=cards_csv['xmin']
        self.ymin=cards_csv['ymin']
        self.xmax=cards_csv['xmax']
        self.ymax=cards_csv['ymax']

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
             
        img=Image.open(self.img_path+self.file_names[idx]).convert("RGB") #(read_image(self.img_path+self.file_names[idx])).float().to(device)
        
        if self.transform:
            img=self.transform(img)

        img = img.to(device)
        box=torch.as_tensor([self.xmin[idx],self.ymin[idx],self.xmax[idx],self.ymax[idx]],dtype=torch.float32).to(device)
        lbl=torch.as_tensor((self.labels[idx]),dtype= torch.int64).to(device)
        #box=(torch.tensor([int(self.xmin[idx]),int(self.ymin[idx]),int(self.xmax[idx]),int(self.ymax[idx])])).to(device)
        #box =box.float()
        #lbl=(torch.tensor([int(self.labels[idx])])).to(device)
        #lbl=(torch.tensor(int(self.labels[idx]))).to(device)
        target={'boxes':box,'labels':lbl}
        #print("size", target['labels'].size())
        return img,target



transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )])
train_data = CardDataset(transform=transforms)


batch_size=1
train_data_loader = DataLoader(train_data, batch_size=batch_size)
#val_data_loader = data.DataLoader(val_data, batch_size=batch_size)
#test_data_loader = data.DataLoader(test_data, batch_size=batch_size)

#model= torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False, progress=True, num_classes=14, pretrained_backbone=False, trainable_backbone_layers=None)
#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=14, pretrained_backbone=False, trainable_backbone_layers=None)

model= torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False, num_classes=14)
#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=14)

model_path='BlackjackRecommenderSystem-main/models/test_model50.m'
model = model.to(device)


import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, optimizer, loss_fn, train_loader, epochs=1):
    
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        print("Training... current epoch: ", epoch)
    
        for batch in train_loader:
            model.train()
            optimizer.zero_grad()
            inputs, target = batch
            
            #print('inputs',inputs)
            output = model(inputs,[target])
            #print(target)
            #print(output)
            #loss = loss_fn(output,target)
            #loss.backward()
            #optimizer.step()
            #training_loss += loss.data.item()
        #training_loss /= len(train_iterator)
           # model.eval()
            #predictions = model(inputs)
            #if torch.eq(predictions[0]["labels"], target["labels"]):
            #print("pred labels ", predictions)#[0]["labels"])
            #print("target ", target["labels"])


    #valid_loss /= len(valid_iterator)
    #print('Epoch: {}, Training Loss: {:.2f},Validation Loss: {:.2f},accuracy = {:.2f}'.format(epoch, training_loss,valid_loss, num_correct / num_examples))

train(model, optimizer, torch.nn.NLLLoss(),train_data_loader)
print("Saving model...")
#torch.save(model,model_path)
#model.eval()
#predictions = model(inputs)
#print("pred labels ", predictions)#[0]["labels"])

