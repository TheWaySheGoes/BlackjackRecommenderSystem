import torchvision
from torchvision import transforms
import torch
from torchvision.io import read_image
import pandas as pd


aaaa=   [[[0.0116, 0.03857, 0.9454, 0.3402],
         [0.04021, 0.01515, 0.6301, 0.0718], 
         [0.06871, 0.06273, 0.8718, 0.3487], 
         [0.07753, 0.01358, 0.9282, 0.5879], 
         [0.04447, 0.07606, 0.7718, 0.2089], 
         [0.05661, 0.04690, 0.1704, 0.8201], 
         [0.01610, 0.04800, 0.0623, 0.5784], 
         [0.09735, 0.02063, 0.4258, 0.9247], 
         [0.09349, 0.03573, 0.9079, 0.9017], 
         [0.01685, 0.02096, 0.9077, 0.0664], 
         [0.07279, 0.01533, 0.4141, 0.8476]],
         
         [[0.0116, 0.03857, 0.9454, 0.3402],
         [0.04021, 0.01515, 0.6301, 0.0718], 
         [0.06871, 0.06273, 0.8718, 0.3487], 
         [0.07753, 0.01358, 0.9282, 0.5879], 
         [0.04447, 0.07606, 0.7718, 0.2089], 
         [0.05661, 0.04690, 0.1704, 0.8201], 
         [0.01610, 0.04800, 0.0623, 0.5784], 
         [0.09735, 0.02063, 0.4258, 0.9247], 
         [0.09349, 0.03573, 0.9079, 0.9017], 
         [0.01685, 0.02096, 0.9077, 0.0664], 
         [0.07279, 0.01533, 0.4141, 0.8476]],

         [[0.0116, 0.03857, 0.9454, 0.3402],
         [0.04021, 0.01515, 0.6301, 0.0718], 
         [0.06871, 0.06273, 0.8718, 0.3487], 
         [0.07753, 0.01358, 0.9282, 0.5879], 
         [0.04447, 0.07606, 0.7718, 0.2089], 
         [0.05661, 0.04690, 0.1704, 0.8201], 
         [0.01610, 0.04800, 0.0623, 0.5784], 
         [0.09735, 0.02063, 0.4258, 0.9247], 
         [0.09349, 0.03573, 0.9079, 0.9017], 
         [0.01685, 0.02096, 0.9077, 0.0664], 
         [0.07279, 0.01533, 0.4141, 0.8476]],

         [[0.0116, 0.03857, 0.9454, 0.3402],
         [0.04021, 0.01515, 0.6301, 0.0718], 
         [0.06871, 0.06273, 0.8718, 0.3487], 
         [0.07753, 0.01358, 0.9282, 0.5879], 
         [0.04447, 0.07606, 0.7718, 0.2089], 
         [0.05661, 0.04690, 0.1704, 0.8201], 
         [0.01610, 0.04800, 0.0623, 0.5784], 
         [0.09735, 0.02063, 0.4258, 0.9247], 
         [0.09349, 0.03573, 0.9079, 0.9017], 
         [0.01685, 0.02096, 0.9077, 0.0664], 
         [0.07279, 0.01533, 0.4141, 0.8476]]]






print("test: works")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,num_classes=11)
# For training
images, boxes = torch.rand(4, 3, 600, 1200), torch.tensor(aaaa)#torch.rand(4, 11, 4)


labels = torch.randint(1, 91, (4, 11))
images = list(image for image in images)
targets = []
for i in range(len(images)):
    d = {}
    d['boxes'] = boxes[i]
    d['labels'] = labels[i]
    targets.append(d)
#print(targets)

#########inputs############
csv_path='new_dataset\\card2\\new_train_label.csv'
img_path='new_dataset\\card2\\train\\'
cards_csv = pd.read_csv(csv_path)
img_limit=20
#print(cards_csv.head())
file_names=cards_csv['filename']
#img_limit=len(file_names)
print('image_limit:',img_limit)
#print(file_names)
imgs=[] #list of imgae tensors
data_transform=transforms.Compose([transforms.Normalize(mean=[0.485],std=[0.229])])

for i in range(0,img_limit):
    #print(file_names[i])
    img=(read_image(img_path+file_names[i])).float()
    print(img.shape)
    img=data_transform(img)
    print(img.shape)
    imgs.append(img)

#########targets############
labels=cards_csv['labels']
klass=cards_csv['class']
xmin=cards_csv['xmin']
ymin=cards_csv['ymin']
xmax=cards_csv['xmax']
ymax=cards_csv['ymax']
klass_length=len(klass.unique())
print("klass length:",klass_length)

print("label length:",len(labels.unique()))
print("label max val:", max(labels))

boxs=[]
for i in range(0,img_limit):
    boxs.append(torch.tensor([[xmin[i],ymin[i],xmax[i],ymax[i]]]))
    #print(boxs[i])
lbls=[]
for i in range(0,img_limit):
    lbls.append(torch.tensor([int(labels[i])]))
    #print(lbls[i])    
kls=[]
#for i in range(0,img_limit):
#    kls.append(torch.tensor(klass[i]))
#    print(kls[i])    
print("labels: ",lbls)


trgts=[] # list of target dictionaries with tensors

for i in range(0,img_limit):
    d={}
    d['boxes']=boxs[i]
    d['labels']=lbls[i]
    trgts.append(d)
#print(trgts)

output = model(imgs, trgts)

#torch.save(model,'models\\model1.m')

model.eval()
for i in range(0,int(img_limit/3)):
    predictions = model([imgs[i]])
    print("actual label:",lbls[i])
    print(predictions)

# For inference
#model.eval()
#x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]



# optionally, if you want to export the model to ONNX:
#torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)
