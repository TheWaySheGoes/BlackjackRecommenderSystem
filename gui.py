import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.io import read_image
from torchvision import datasets, models, transforms
import matplotlib
import matplotlib.pyplot as plt
import time
import os
import copy
import csv
import pandas as pd
import numpy as np
from skimage import io, transform
from pynput.mouse import Listener as MouseListener
from pynput.mouse import Button, Controller
from pynput.keyboard import Listener as KeyboardListener
#from pynput.keyboard import Key, Controller
import PySimpleGUI as sg
import threading
from PIL import ImageGrab, Image, ImageDraw
import recommender


class GUI(threading.Thread):
   
    def __init__(self):
        threading.Thread.__init__(self)
        self.model=torch.load('model\\test_model100.m',map_location=torch.device('cpu'))
        self.model.eval()
        self.list=[]
        self.hit=[sg.Text('x'),sg.In(size=(5,1),enable_events=True,key='hit_x'),sg.Text('y'),sg.In(size=(5,1),enable_events=True,key='hit_y'),sg.Radio(text='hit',group_id='check',enable_events=False,key='hit_check')]
        self.stand=[sg.Text('x'),sg.In(size=(5,1),enable_events=True,key='stand_x'),sg.Text('y'),sg.In(size=(5,1),enable_events=True,key='stand_y'),sg.Radio(text='stand',group_id='check',enable_events=False,key='stand_check')]
        self.split=[sg.Text('x'),sg.In(size=(5,1),enable_events=True,key='split_x'),sg.Text('y'),sg.In(size=(5,1),enable_events=True,key='split_y'),sg.Radio(text='split',group_id='check',enable_events=False,key='split_check')]
        self.double=[sg.Text('x'),sg.In(size=(5,1),enable_events=True,key='double_x'),sg.Text('y'),sg.In(size=(5,1),enable_events=True,key='double_y'),sg.Radio(text='double',group_id='check',enable_events=False,key='double_check')]
        self.deal=[sg.Text('x'),sg.In(size=(5,1),enable_events=True,key='deal_x'),sg.Text('y'),sg.In(size=(5,1),enable_events=True,key='deal_y'),sg.Radio(text='deal',group_id='check',enable_events=False,key='deal_check')]
        self.extra1=[sg.Text('x'),sg.In(size=(5,1),enable_events=True,key='extra1_x'),sg.Text('y'),sg.In(size=(5,1),enable_events=True,key='extra1_y'),sg.Radio(text='extra1',group_id='check',enable_events=False,key='extra1_check')]
        self.extra2=[sg.Text('x'),sg.In(size=(5,1),enable_events=True,key='extra2_x'),sg.Text('y'),sg.In(size=(5,1),enable_events=True,key='extra2_y'),sg.Radio(text='extra2',group_id='check',enable_events=False,key='extra2_check')]
        self.start_end_XY_Radio=[sg.Radio(text='grab area',group_id='grab_check',enable_events=False,key='grab_check')]
        self.start_end_Y=[sg.Text('start x'),sg.In(size=(5,1),enable_events=True,key='mouse_x_start'),sg.Text('end x'),sg.In(size=(5,1),enable_events=True,key='mouse_x_end')]
        self.start_end_X=[sg.Text('start y'),sg.In(size=(5,1),enable_events=True,key='mouse_y_start'),sg.Text('end y'),sg.In(size=(5,1),enable_events=True,key='mouse_y_end')]
        self.separator=[sg.HorizontalSeparator()]
        self.image_window = [sg.Image(filename="data\MatejkoKonst3Maj1791.png",background_color='white',enable_events=True,size=(300,300),key='_image_')]

        self.column_1=[
                        self.start_end_XY_Radio,
                        self.start_end_Y,
                        self.start_end_X,
                        self.separator,
                        [sg.Text('in text'), sg.In(size=(25, 1),enable_events=False,key='in')],
                        [sg.Button('grab',key='_grab_'),sg.Button('calc',key='_calc_'),sg.Button('remove',key='_remove_') ],
                        [sg.Listbox(values=self.list,enable_events=True,size=(30,5), key='_list_',auto_size_text=True)]
                    
        ]

#        self.column_3 = [                    
#                    ,
#                    [sg.Text('out text '), sg.In(size=(25, 1),enable_events=True,key='out')],
#                    [sg.Text('extra '), sg.In(size=(25, 1),enable_events=True,key='extra')],
#                    ]

        
        self.column_2 = [
                        self.image_window
                    ]

        self.layout = [
                [sg.Column(self.column_1),sg.Column(self.column_2,key='column_2')]                
                ]

        self.window = sg.Window('BlackJack',self.layout,resizable=True)
        self.keyboard_listener=KeyboardListener(on_press=self.on_press, on_release=self.on_release)
        self.mouse_listener = MouseListener(on_move=self.on_move, on_click=self.on_click,onscroll=self.on_scroll)
        self.mouse = Controller()
        self.mouse_button = Button
        self.keyboard = Controller()
        #self.keyboard_key=Key



    def click_hit(self):
        print((self.window['hit_x'].get()))
        print((self.window['hit_y'].get()))
        print(int(self.window['hit_x'].get()))
        print(int(self.window['hit_y'].get()))
        self.mouse.position=(int(self.window['hit_x'].get()),int(self.window['hit_y'].get()))
        self.mouse.press(self.mouse_button.left)
        self.mouse.release(self.mouse_button.left)
        self.mouse.position=(int(self.window['hit_x'].get()),int(self.window['hit_y'].get()))
        print(self.mouse.position)
        
    def click_stand(self):
        print((self.window['stand_x'].get()))
        print((self.window['stand_y'].get()))
        print(int(self.window['stand_x'].get()))
        print(int(self.window['stand_y'].get()))
        self.mouse.position=(int(self.window['stand_x'].get()),int(self.window['stand_y'].get()))
        self.mouse.press(self.mouse_button.left)
        self.mouse.release(self.mouse_button.left)
        self.mouse.position=(int(self.window['stand_x'].get()),int(self.window['stand_y'].get()))
        print(self.mouse.position)
    
    def click_split(self):
        print((self.window['split_x'].get()))
        print((self.window['split_y'].get()))
        print(int(self.window['split_x'].get()))
        print(int(self.window['split_y'].get()))
        self.mouse.position=(int(self.window['split_x'].get()),int(self.window['split_y'].get()))
        self.mouse.press(self.mouse_button.left)
        self.mouse.release(self.mouse_button.left)
        self.mouse.position=(int(self.window['split_x'].get()),int(self.window['split_y'].get()))
        print(self.mouse.position)

    def click_double(self):
        print((self.window['double_x'].get()))
        print((self.window['double_y'].get()))
        print(int(self.window['double_x'].get()))
        print(int(self.window['double_y'].get()))
        self.mouse.position=(int(self.window['double_x'].get()),int(self.window['double_y'].get()))
        self.mouse.press(self.mouse_button.left)
        self.mouse.release(self.mouse_button.left)
        self.mouse.position=(int(self.window['double_x'].get()),int(self.window['double_y'].get()))
        print(self.mouse.position)

    def click_deal(self):
        print((self.window['deal_x'].get()))
        print((self.window['deal_y'].get()))
        print(int(self.window['deal_x'].get()))
        print(int(self.window['deal_y'].get()))
        self.mouse.position=(int(self.window['deal_x'].get()),int(self.window['deal_y'].get()))
        self.mouse.press(self.mouse_button.left)
        self.mouse.release(self.mouse_button.left)
        self.mouse.position=(int(self.window['deal_x'].get()),int(self.window['deal_y'].get()))
        print(self.mouse.position)

    def click_extra1(self):
        print((self.window['extra1_x'].get()))
        print((self.window['extra1_y'].get()))
        print(int(self.window['extra1_x'].get()))
        print(int(self.window['extra1_y'].get()))
        self.mouse.position=(int(self.window['extra1_x'].get()),int(self.window['extra1_y'].get()))
        self.mouse.press(self.mouse_button.left)
        self.mouse.release(self.mouse_button.left)
        self.mouse.position=(int(self.window['extra1_x'].get()),int(self.window['extra1_y'].get()))
        print(self.mouse.position)

    def click_extra2(self):
        print((self.window['extra2_x'].get()))
        print((self.window['extra2_y'].get()))
        print(int(self.window['extra2_x'].get()))
        print(int(self.window['extra2_y'].get()))
        self.mouse.position=(int(self.window['extra2_x'].get()),int(self.window['extra2_y'].get()))
        self.mouse.press(self.mouse_button.left)
        self.mouse.release(self.mouse_button.left)
        self.mouse.position=(int(self.window['extra2_x'].get()),int(self.window['extra2_y'].get()))
        print(self.mouse.position)


    def run(self):
        self.start_gui()

    def start_gui(self):
        #start mouse and keyboard listeners
        self.keyboard_listener.start()
        self.mouse_listener.start()
        #######this blocks dont use########
        #self.keyboard_listener.join()
        #self.mouseListener.join()
        ###################################

        in_text=""
        out_text=""
        extra=""

        while True:
            event, values = self.window.read()

            if event=='in':
                print()

            if event=='out':
                out_text=values['out']

            if event=='extra':
                extra=values['extra']

            if event == "_grab_" and self.grab_area_isSet()==True:
                img=ImageGrab.grab(bbox=(int(self.window['mouse_x_start'].get()),int(self.window['mouse_y_start'].get()),int(self.window['mouse_x_end'].get()),int(self.window['mouse_y_end'].get())),include_layered_windows=True,all_screens=True)
                #img.show()
                img.save('data\grab1.png')
                self.window['_image_'].update(filename='data\grab1.png')
                #print("grab")
               
            if event=='_calc_':
                #send to model                
                img=Image.open('data\grab1.png').convert("RGB")
                #img=(read_image('data\grab1.png')).float()
                a = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )])
                img=a(img)
                predictions = self.model([img])
                pred_boxes=predictions[0]['boxes'].detach().numpy()
                pred_labels=predictions[0]['labels'].detach().numpy()
                pred_scores=predictions[0]['scores'].detach().numpy()
                half_image=(int(self.window['mouse_y_end'].get())-int(self.window['mouse_y_start'].get()))/2
                print('half image:',half_image)
                dealer_pred_boxes=[]
                dealer_pred_labels=[]
                dealer_pred_scores=[]
                player_pred_boxes=[]
                player_pred_labels=[]
                player_pred_scores=[]
                out_img=Image.open('data\grab1.png')
                draw=ImageDraw.Draw(out_img)
                for i in range(0,len(pred_scores)):
                    if pred_boxes[i][1]< half_image and pred_boxes[i][3]<half_image:
                        #dealers
                        dealer_pred_boxes.append(pred_boxes[i])
                        draw.rectangle([(pred_boxes[i][0],pred_boxes[i][1]),(pred_boxes[i][2],pred_boxes[i][3])],outline='red',width=2)
                        dealer_pred_labels.append(pred_labels[i])
                        dealer_pred_scores.append(pred_scores[i])
                    elif pred_boxes[i][1]> half_image and pred_boxes[i][3]>half_image:
                        #players
                        player_pred_boxes.append(pred_boxes[i])
                        draw.rectangle([(pred_boxes[i][0],pred_boxes[i][1]),(pred_boxes[i][2],pred_boxes[i][3])],outline='black',width=2)
                        player_pred_labels.append(pred_labels[i])
                        player_pred_scores.append(pred_scores[i])
                out_img.save('data\grab1.png')
                self.window['_image_'].update(filename='data\grab1.png')
                print('dealer:')
                print(dealer_pred_boxes)
                print(dealer_pred_labels)
                print(dealer_pred_scores)
                print('player:')
                print(player_pred_boxes)
                print(player_pred_labels)
                print(player_pred_scores)
                self.list=[]
                self.window['_list_'].update(self.list)
                
                

                dealer_sum=(dealer_pred_labels[dealer_pred_scores.index(max(dealer_pred_scores))] if len(dealer_pred_scores)>0  else -1) if (dealer_pred_labels[dealer_pred_scores.index(max(dealer_pred_scores))] if len(dealer_pred_scores)>0  else -1) <10 else 10
            
                n=int(self.window['in'].get())
                print('nbr of cards:',n)
                player_sum=0
                player_cards=[]
                for i in range(0,n):
                    player_sum=player_sum+(player_pred_labels[i] if player_pred_labels[i] <10 else 10)
                    player_cards.append(player_pred_labels[i])
                    print(player_sum)

                self.list.append(('dealer sum:',  dealer_sum ))
                self.list.append(('dealer card:',  (dealer_pred_labels[dealer_pred_scores.index(max(dealer_pred_scores))] if len(dealer_pred_scores)>0  else -1)))
                self.window['_list_'].update(self.list)
                self.list.append(('player sum:',player_sum))
                self.list.append(('player cards:',player_cards))
                self.window['_list_'].update(self.list)

                if player_sum <21:
                    #send to recommender system
                    recommendation = recommender.recommender(a=0,p=player_sum,d=dealer_sum) 
                    #make a decision
                    if recommendation.thorpe()==0:
                        self.list.append(('recommendation: STAND'))
                        self.window['_list_'].update(self.list)
                        #self.click_stand()
                    elif recommendation.thorpe()==1:
                        self.list.append(('recommendation: HIT'))
                        self.window['_list_'].update(self.list)
                        #self.click_hit()
                    elif recommendation.thorpe()==2:
                        self.list.append(('recommendation: DOUBLE'))
                        self.window['_list_'].update(self.list)
                        #self.click_double()
                else:
                    self.list.append(('!!! BUST !!!!'))
                    self.window['_list_'].update(self.list)

                


            if event=='_remove_':
                if len(values['_list_'])>0:
                    self.list.remove(in_text+';'+out_text+';'+extra)
                    self.window['_list_'].update(self.list)

            if event=='_list_':
                
                if len(values['_list_'])>1:
                    list_item_split=values['_list_'][0].strip().split(';')
                    self.window['in'].update(list_item_split[0])
                    in_text=values['in']
                    self.window['out'].update(list_item_split[1])
                    out_text=values['out']
                    self.window['extra'].update(list_item_split[2])
                    extra=values['extra']


            if event == sg.WIN_CLOSED:
                break




#mouse and keyboard events
    def on_press(self,key):
        print(key)
    def on_release(self,key):
        print(key)
    def on_move(self,x,y):
        x
        
    def on_click(self,x,y,button,pressed):
        #self.list.append(button)
        #self.list.append(pressed)
        #self.window['_list_'].update(self.list)
        if len(self.list) >100000: self.list=[]
        #on press and release  get pointer position      
        if pressed:
            if self.window['grab_check'].get()==True:
                self.window['mouse_x_start'].update(x)
                self.window['mouse_y_start'].update(y)

        else:
            if self.window['grab_check'].get()==True:
                self.window['mouse_x_end'].update(x)
                self.window['mouse_y_end'].update(y)
                self.window['grab_check'].update(value=False)

            

                 

    def on_scroll(self,x,y,dx,dy):
        print("scroll")

    def grab_area_isSet(self):
        ret=False
        if int(self.window['mouse_x_start'].get()) < int(self.window['mouse_x_end'].get()) and int(self.window['mouse_y_start'].get()) < int(self.window['mouse_y_end'].get()):
            ret=True
        else:
            msg="Grab area must be set n 1. Check the area set button \n 2. Click and drag mouse pointer to mark the area \n 3. Do so from upper left to lower right "
            self.list.append(msg)
            self.window['_list_'].update(self.list)
            print(msg)
            ret=False
        return ret

##### testing gui

gui=GUI()
gui.start()
