import os
import PySimpleGUI as sg


class GUI():
   
    def __init__(self):
        self.list=[]


        self.column_1 = [
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
        

    def start(self):
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





##### testing gui
gui=GUI()
gui.start()