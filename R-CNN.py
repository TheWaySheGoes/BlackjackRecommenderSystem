import torchvision
from torchvision import transforms
import torch
from torchvision.io import read_image
import pandas as pd

device = torch.device("cpu")

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

model_path='models\\model1.m'




#print("test: works")

#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,num_classes=11)
model=torch.load('models\\model1.m')
#model= torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False, progress=True, num_classes=12, pretrained_backbone=False, trainable_backbone_layers=6)
#torch.save(model,model_path)
model = model.to(device)
# For training
images, boxes = torch.rand(4, 3, 600, 1200), torch.tensor(aaaa)#torch.rand(4, 11, 4)
print(model)

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
csv_path='new_dataset/card1/new_train_label.csv'
img_path='new_dataset/card1/train/'
cards_csv = pd.read_csv(csv_path)
print('dataset:',cards_csv)



#print(cards_csv.head())
file_names=cards_csv['filename']
#img_limit=len(file_names)
#print(file_names)

data_transform=transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
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

################################ set this for batches
img_from=0
img_to=1
img_steps=1
img_max_val=50

while img_to < img_max_val:
    #model=torch.load(model_path)
    #model.train()
    imgs=[] #list of image tensors inputs
    boxs=[]
    lbls=[]
    kls=[]
    trgts=[] # list of target dictionaries with tensors

    for i in range(img_from,img_to):
        print(file_names[i])
        img=(read_image(img_path+file_names[i])).float()
        #print(img.shape)
        img=data_transform(img)
        #print(img.shape)
        img=img.to(device)
        imgs.append(img)
    #imgs=imgs.to(device)
    #########targets############
    for i in range(img_from,img_to):
        boxs.append(torch.tensor([[xmin[i],ymin[i],xmax[i],ymax[i]]]).to(device))
        #print(boxs[i])
    for i in range(img_from,img_to):
        lbls.append(torch.tensor([int(labels[i])]).to(device))
        #print(lbls[i])    
    #lbls=lbls.to(device)
    #for i in range(img_from,img_to):
    #    kls.append(torch.tensor(klass[i]))
    #    print(kls[i])    
    print("labels: ",lbls)

    for i in range(0,len(boxs)):
        d={}
        d['boxes']=boxs[i]
        d['labels']=lbls[i]
        trgts.append(d)
    #print(trgts)
    #trgts=trgts.to(device)
    output = model(imgs, trgts)
    torch.save(model,model_path)
    img_from=img_to
    img_to=img_to+img_steps


    #model=torch.load('models\\model1.m')
#    if img_to%10==0:
model.eval()

predictions = model([imgs[0]])
print("actual label:",lbls[0])
print(predictions)

#for i in range(img_from,img_to):
#    predictions = model([imgs[i]])
#    print("actual label:",lbls[i])
#    print(predictions)

# For inference
#model.eval()
#x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# optionally, if you want to export the model to ONNX:
#torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)
