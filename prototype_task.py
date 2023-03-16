import os
import tkinter as tk
import numpy as np
import cv2
import random
from preattentive_object import PreattentiveObject

class ProtoTypeTask:
    def __init__(self, task):
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        self.background = np.zeros((self.screen_height,self.screen_width), dtype='uint8')
        self.background_color = np.zeros((self.screen_height,self.screen_width,3), dtype='uint8')
        self.ready = False
        self.nonfamiliar_list = os.listdir('nonfamiliar')
        self.familiar_list = os.listdir('familiar')
        self.preattentive_list = os.listdir('IdentiGaze-Stimuli')
        self.preattentive_object = PreattentiveObject(self.screen_width, self.screen_height, 'black')
        # self.preattentive_object.set_set_size(2)

        if task=='familiar':
            self.familiar()
        elif task=='preattentive':
            self.preattentive()
        elif task=='linearface':
            self.linearface()

    def preattentive(self):
        Interface_x = 0
        Interface_y = 0
        bg = self.background_color.copy()
        # self.preattentive_object.random_control = False
        cv2.putText(bg, "Press the key to start the test", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow('image', Interface_x, Interface_y)
        cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('image', bg)
        cv2.waitKey(0) & 0xff
        while True:
            if self.ready==0:
                bg = self.get_cross()
                cv2.imshow('image', bg)
                key = cv2.waitKey(800) & 0xff
            elif self.ready == 1:
                bg = self.background.copy()
                self.ready = 2
                cv2.imshow('image', bg)
                key = cv2.waitKey(200) & 0xff
            else:
                if len(self.preattentive_object.grid_index_list)==0:
                    self.preattentive_object.grid_index_list = list(range(self.preattentive_object.set_size**2))
                # bg, _, = self.preattentive_object.stimuli_hue(target_index)
                # self.preattentive_object.set_set_size(random.choice(list(range(3,7))))
                target_index = random.choice(self.preattentive_object.grid_index_list)
                self.preattentive_object.grid_index_list.remove(target_index)

                task = random.choice(list(range(5)))
                if task == 0:
                    bg, _, = self.preattentive_object.stimuli_shape(target_index)
                elif task == 1:
                    bg, _, = self.preattentive_object.stimuli_size(target_index)
                elif task == 2:
                    bg, _, = self.preattentive_object.stimuli_hue(target_index)
                elif task == 3:
                    bg, _, = self.preattentive_object.stimuli_brightness(target_index)
                elif task == 4:
                    bg, _, = self.preattentive_object.stimuli_orientation(target_index)
                self.ready = 0
                cv2.imshow('image', bg)
                key = cv2.waitKey(300) & 0xff

            # cv2.setMouseCallback('image', event_start)
            if key == ord('q'):
                cv2.destroyAllWindows()
                print('End')
                break
            else:
                continue

    def preattentive_old(self):
        Interface_x = 0
        Interface_y = 0
        bg = self.background_color.copy()
        cv2.putText(bg, "Press the key to start the test", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow('image', Interface_x, Interface_y)
        cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('image', bg)
        cv2.waitKey(0) & 0xff
        while True:
            if self.ready==False:
                bg = self.get_cross_white()
                cv2.imshow('image', bg)
                key = cv2.waitKey(250) & 0xff

            else:
                bg = self.center_stimuli()
                cv2.imshow('image', bg)
                key = cv2.waitKey(300) & 0xff

            # cv2.setMouseCallback('image', event_start)
            if key == ord('q'):
                cv2.destroyAllWindows()
                print('End')
                break
            else:
                continue


    def familiar(self):
        Interface_x = 0
        Interface_y = 0
        bg = self.background.copy()
        cv2.putText(bg, "Press the key to start the test", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow('image', Interface_x, Interface_y)
        cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('image', bg)
        cv2.waitKey(0) & 0xff

        while True:
            if self.ready==0:
                bg = self.get_cross()
                cv2.imshow('image', bg)
                key = cv2.waitKey(250) & 0xff

            else:
                bg = self.grid_face()
                cv2.imshow('image', bg)
                key = cv2.waitKey(550) & 0xff

            # cv2.setMouseCallback('image', event_start)
            if key == ord('q'):
                cv2.destroyAllWindows()
                print('End')
                break
            else:
                continue

    def linearface(self):
        Interface_x = 0
        Interface_y = 0
        bg = self.background.copy()
        cv2.putText(bg, "Press the key to start the test", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow('image', Interface_x, Interface_y)
        cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('image', bg)
        cv2.waitKey(0) & 0xff
        
        while True:
            if self.ready==0:
                bg = self.get_cross()
                cv2.imshow('image', bg)
                key = cv2.waitKey(250) & 0xff

            else:
                bg = self.center_face()
                cv2.imshow('image', bg)
                key = cv2.waitKey(100) & 0xff

            # cv2.setMouseCallback('image', event_start)
            if key == ord('q'):
                cv2.destroyAllWindows()
                print('End')
                break
            else:
                continue

    def center_face(self):
        x = self.screen_width/2-300
        y = self.screen_height/2-300
        if len(self.familiar_list)>0:
            if self.dicetoss()==True:
                this_familiar_face = random.choice(self.familiar_list)
                self.familiar_list.remove(this_familiar_face)
                img = cv2.imread(os.path.join('familiar', this_familiar_face), cv2.IMREAD_GRAYSCALE)
            else:
                this_nonfamiliar_face = random.choice(self.nonfamiliar_list)
                self.nonfamiliar_list.remove(this_nonfamiliar_face)
                img = cv2.imread(os.path.join('nonfamiliar', this_nonfamiliar_face), cv2.IMREAD_GRAYSCALE)
            
        else:
            this_nonfamiliar_face = random.choice(self.nonfamiliar_list)
            self.nonfamiliar_list.remove(this_nonfamiliar_face)
            img = cv2.imread(os.path.join('nonfamiliar', this_nonfamiliar_face), cv2.IMREAD_GRAYSCALE)
        # img = cv2.resize(img, dsize=(600,600))
        # bg = self.background.copy()
        # bg[int(y):int(y)+600,int(x):int(x)+600] = img
        space = random.choice(list(range(8)))
        img = cv2.resize(img, dsize=(200,200))
        bg = self.background.copy()
        bg = self.put_grid(bg, img, space)
        self.ready = 0
        return bg

    def grid_face(self):
        show_face = []
        if len(self.familiar_list)>0:
            if self.cointoss()=='up':
                this_familiar_face = random.choice(self.familiar_list)
                show_face.append(['familiar',this_familiar_face])
                self.familiar_list.remove(this_familiar_face)
            else:
                pass
        iter_num = 8-len(show_face)
        for _ in range(iter_num):
            this_nonfamiliar_face = random.choice(self.nonfamiliar_list)
            show_face.append(['nonfamiliar',this_nonfamiliar_face])
            self.nonfamiliar_list.remove(this_nonfamiliar_face)
        if len(show_face)!=8:
            raise Exception('show faces are not 8 elements')
        random.shuffle(show_face)
        bg = self.background.copy()
        for n, [state, image_name] in enumerate(show_face):
            img = cv2.imread(os.path.join(state, image_name), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, dsize=(200,200))
            bg = self.put_grid(bg, img, n)

        self.ready = 0
        return bg

    def put_grid(self, bg, img, number):
        if number == 0:
            x = self.screen_width/2-300
            y = self.screen_height/2-300
        elif number == 1:
            x = self.screen_width/2-100
            y = self.screen_height/2-300
        elif number == 2:
            x = self.screen_width/2+100
            y = self.screen_height/2-300
        elif number == 3:
            x = self.screen_width/2-300
            y = self.screen_height/2-100
        elif number == 4:
            x = self.screen_width/2+100
            y = self.screen_height/2-100
        elif number == 5:
            x = self.screen_width/2-300
            y = self.screen_height/2+100
        elif number == 6:
            x = self.screen_width/2-100
            y = self.screen_height/2+100
        elif number == 7:
            x = self.screen_width/2+100
            y = self.screen_height/2+100
        else:
            raise Exception("face list length is over 8")
        bg[int(y):int(y)+200,int(x):int(x)+200] = img
        return bg

    def center_stimuli(self):
        x = self.screen_width/2-300
        y = self.screen_height/2-300
        bg = self.background_color.copy()
        bg.fill(255)
        if len(self.preattentive_list)>0:
            this_preattentive = random.choice(self.preattentive_list)
            self.preattentive_list.remove(this_preattentive)
            img = cv2.imread(os.path.join('IdentiGaze-Stimuli', this_preattentive), cv2.IMREAD_COLOR)
        else:
            cv2.putText(bg, "Stimuli Over", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2)
            return bg
        img = cv2.resize(img, dsize=(600,600))
        bg[int(y):int(y)+600,int(x):int(x)+600] = img
        self.ready = False
        return bg

    def cointoss(self):
        toss_list = ['up', 'down']
        return random.choice(toss_list)

    def dicetoss(self):
        toss_list = [True, False, False, False, False, False]
        return random.choice(toss_list)

    def get_cross(self):
        cross = np.zeros((self.screen_height,self.screen_width), dtype='uint8')
        cross[int(self.screen_height/2)-30:int(self.screen_height/2)+30,int(self.screen_width/2)-5:int(self.screen_width/2)+5].fill(255)
        cross[int(self.screen_height/2)-5:int(self.screen_height/2)+5,int(self.screen_width/2)-30:int(self.screen_width/2)+30].fill(255)
        self.ready = 1
        return cross

    def get_cross_white(self):
        cross = np.zeros((self.screen_height,self.screen_width,3), dtype='uint8')
        cross.fill(255)
        cross[int(self.screen_height/2)-30:int(self.screen_height/2)+30,int(self.screen_width/2)-5:int(self.screen_width/2)+5].fill(0)
        cross[int(self.screen_height/2)-5:int(self.screen_height/2)+5,int(self.screen_width/2)-30:int(self.screen_width/2)+30].fill(0)
        self.ready = 1
        return cross

if __name__=='__main__':
    # print("Hello, World!")
    # myProto = ProtoTypeTask("familiar")
    # myProto = ProtoTypeTask("linearface")
    myProto = ProtoTypeTask("preattentive")