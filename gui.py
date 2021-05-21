import os
from pynput.mouse import Listener as MouseListener
from pynput.keyboard import Listener as KeyboardListener
import PySimpleGUI as sg


class GUI():
   
    def __init__(self):
        self.list=[]


        self.column_1 = [
                    [sg.Text('mouse x'),sg.In(size=(25,1),enable_events=True,key='mouse_x')],
                    [sg.Text('mouse y'),sg.In(size=(25,1),enable_events=True,key='mouse_y')],
                    [sg.Text('in text         '), sg.In(size=(25, 1),enable_events=True,key='in')],
                    [sg.Text('out text '), sg.In(size=(25, 1),enable_events=True,key='out')],
                    [sg.Text('extra '), sg.In(size=(25, 1),enable_events=True,key='extra')],
                    [sg.Button('ok',key='_ok_'),sg.Button('add',key='_add_'),sg.Button('remove',key='_remove_') ],
                    [sg.Listbox(values=self.list,enable_events=True,size=(60,20), key='_list_')]
                    ]

        self.column_2 = [
                    [sg.Image(background_color='red',enable_events=True,size=(60,20),key='_image_')]
                    ]

        self.layout = [
                [sg.Column(self.column_1),sg.Column(self.column_2)]                
                ]

        self.window = sg.Window('TEST test test',self.layout)
        
        self.keyboard_listener=KeyboardListener(on_press=self.on_press, on_release=self.on_release)
        self.mouse_listener = MouseListener(on_move=self.on_move, on_click=self.on_click,onscroll=self.on_scroll)


    def start(self):
        #start mouse and keyboard listeners
        self.keyboard_listener.start()
        self.mouse_listener.start()
        #self.keyboard_listener.join()
        #self.mouseListener.join()

        in_text=""
        out_text=""
        extra=""

        while True:
            
            event, values = self.window.read()
            if event=='in':
                in_text=values['in']

            if event=='out':
                out_text=values['out']

            if event=='extra':
                extra=values['extra']
            if event == "_ok_":
                print("ok")
               
            if event=='_add_':
                self.list.append(in_text+';'+out_text+';'+extra)
                self.window['_list_'].update(self.list)

            if event=='_remove_':
                self.list.remove(in_text+';'+out_text+';'+extra)
                self.window['_list_'].update(self.list)


            if event=='_list_':
                list_item_split=values['_list_'][0].strip().split(';')
                self.window['in'].update(list_item_split[0])
                in_text=values['in']
                self.window['out'].update(list_item_split[1])
                out_text=values['out']
                self.window['extra'].update(list_item_split[2])
                extra=values['extra']

            if event == sg.WIN_CLOSED:
                break

    def on_press(self,key):
        print(key)
    def on_release(self,key):
        print(key)
    def on_move(self,x,y):
        self.window['mouse_x'].update(x)
        self.window['mouse_y'].update(y)
        
    def on_click(self,x,y,button,pressed):
        print(x)
        print(y)
        print(button)
        print(pressed)
    def on_scroll(self,x,y,dx,dy):
        print("scroll")



##### testing gui
gui=GUI()
gui.start()