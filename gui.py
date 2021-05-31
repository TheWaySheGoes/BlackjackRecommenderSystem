import os
from pynput.mouse import Listener as MouseListener
from pynput.mouse import Button, Controller
from pynput.keyboard import Listener as KeyboardListener
#from pynput.keyboard import Key, Controller
import PySimpleGUI as sg
import threading
from PIL import ImageGrab, Image
'test'
import recommender


class GUI(threading.Thread):
   
    def __init__(self):
        threading.Thread.__init__(self)
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

        self.column_1=[
                        self.start_end_XY_Radio,
                        self.start_end_Y,
                        self.start_end_X,
                        self.separator,
                        self.hit,
                        self.stand,
                        self.split,
                        self.double,
                        self.deal,
                        self.extra1,
                        self.extra2
        ]

        self.column_2 = [                    
                    [sg.Text('in text'), sg.In(size=(25, 1),enable_events=True,key='in')],
                    [sg.Text('out text '), sg.In(size=(25, 1),enable_events=True,key='out')],
                    [sg.Text('extra '), sg.In(size=(25, 1),enable_events=True,key='extra')],
                    [sg.Button('grab',key='_grab_'),sg.Button('add',key='_add_'),sg.Button('remove',key='_remove_') ],
                    [sg.Listbox(values=self.list,enable_events=True,size=(30,20), key='_list_',auto_size_text=True)]
                    ]

        self.image_window = [sg.Image(filename="data\MatejkoKonst3Maj1791.png",background_color='white',enable_events=True,size=(300,300),key='_image_')]

        self.column_3 = [
                        self.image_window
                    ]

        self.layout = [
                [sg.Column(self.column_1),sg.Column(self.column_2,key='column_2'),sg.Column(self.column_3)]                
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
        
    def stand_hit(self):
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
                recommendation = recommender.recommender(a=1,p=21,d=2) 
                if recommendation.thorpe()==0:
                    self.stand_hit()
                in_text=values['in']

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
               
            if event=='_add_':

                self.list.append(in_text+';'+out_text+';'+extra)
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

            if self.window['hit_check'].get()==True:
                self.window['hit_x'].update(x)
                self.window['hit_y'].update(y)
                print('hit check', self.window['hit_check'].get())
                self.window['hit_check'].update(value=False)

            if self.window['stand_check'].get()==True:
                self.window['stand_x'].update(x)
                self.window['stand_y'].update(y)
                print('stand check', self.window['stand_check'].get())
                self.window['stand_check'].update(value=False)

            if self.window['split_check'].get()==True:
                self.window['split_x'].update(x)
                self.window['split_y'].update(y)
                print('split check', self.window['split_check'].get())
                self.window['split_check'].update(value=False)

            if self.window['double_check'].get()==True:
                self.window['double_x'].update(x)
                self.window['double_y'].update(y)
                print('double check', self.window['double_check'].get())
                self.window['double_check'].update(value=False)

            if self.window['deal_check'].get()==True:
                self.window['deal_x'].update(x)
                self.window['deal_y'].update(y)
                print('deal check', self.window['deal_check'].get())
                self.window['deal_check'].update(value=False)

            if self.window['extra1_check'].get()==True:
                self.window['extra1_x'].update(x)
                self.window['extra1_y'].update(y)
                print('extra1 check', self.window['extra1_check'].get())
                self.window['extra1_check'].update(value=False)

            if self.window['extra2_check'].get()==True:
                self.window['extra2_x'].update(x)
                self.window['extra2_y'].update(y)
                print('extra2 check', self.window['extra2_check'].get())
                self.window['extra2_check'].update(value=False)
                


                 

    def on_scroll(self,x,y,dx,dy):
        print("scroll")

    def grab_area_isSet(self):
        ret=False
        if self.window['mouse_x_start'].get() < self.window['mouse_x_end'].get() and self.window['mouse_y_start'].get() < self.window['mouse_y_end'].get():
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
