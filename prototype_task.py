import os
import tkinter as tk
import numpy as np
import cv2
import random
from preattentive_object import PreattentiveObject
from familiar_object import FamiliarObject

class ProtoTypeTask:
    def __init__(self, task):
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        self.background = np.zeros((self.screen_height,self.screen_width), dtype='uint8')
        self.background_color = np.zeros((self.screen_height,self.screen_width,3), dtype='uint8')
        self.ready = False
        self.preattentive_list = os.listdir('IdentiGaze-Stimuli')
        self.preattentive_object = PreattentiveObject(self.screen_width, self.screen_height, 'black')
        self.familiar_object = FamiliarObject(self.screen_width, self.screen_height, 'black')
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
                self.ready = 1
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
                bg, _ = self.preattentive_object.stimuli_hue(target_index)
                # task = random.choice(list(range(5)))
                # if task == 0:
                #     bg, _, = self.preattentive_object.stimuli_shape(target_index)
                # elif task == 1:
                #     bg, _, = self.preattentive_object.stimuli_size(target_index)
                # elif task == 2:
                #     bg, _, = self.preattentive_object.stimuli_hue(target_index)
                # elif task == 3:
                #     bg, _, = self.preattentive_object.stimuli_brightness(target_index)
                # elif task == 4:
                #     bg, _, = self.preattentive_object.stimuli_orientation(target_index)
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
                self.ready = 1
                cv2.imshow('image', bg)
                key = cv2.waitKey(250) & 0xff

            elif self.ready == 1:
                bg = self.background.copy()
                self.ready = 2
                cv2.imshow('image', bg)
                key = cv2.waitKey(200) & 0xff

            else:
                bg = self.center_stimuli()
                cv2.imshow('image', bg)
                key = cv2.waitKey(300) & 0xff
                self.ready = 0

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
                self.ready = 1
                cv2.imshow('image', bg)
                key = cv2.waitKey(800) & 0xff

            elif self.ready == 1:
                bg = self.background.copy()
                self.ready = 2
                cv2.imshow('image', bg)
                key = cv2.waitKey(200) & 0xff

            else:
                bg = self.familiar_object.grid_face()
                self.ready = 0
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
                self.ready = 1
                cv2.imshow('image', bg)
                key = cv2.waitKey(800) & 0xff

            elif self.ready == 1:
                bg = self.background.copy()
                self.ready = 2
                cv2.imshow('image', bg)
                key = cv2.waitKey(200) & 0xff

            else:
                bg = self.familiar_object.center_face()
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

    def get_cross(self):
        cross = np.zeros((self.screen_height,self.screen_width), dtype='uint8')
        cross[int(self.screen_height/2)-30:int(self.screen_height/2)+30,int(self.screen_width/2)-5:int(self.screen_width/2)+5].fill(255)
        cross[int(self.screen_height/2)-5:int(self.screen_height/2)+5,int(self.screen_width/2)-30:int(self.screen_width/2)+30].fill(255)
        return cross

    def get_cross_white(self):
        cross = np.zeros((self.screen_height,self.screen_width,3), dtype='uint8')
        cross.fill(255)
        cross[int(self.screen_height/2)-30:int(self.screen_height/2)+30,int(self.screen_width/2)-5:int(self.screen_width/2)+5].fill(0)
        cross[int(self.screen_height/2)-5:int(self.screen_height/2)+5,int(self.screen_width/2)-30:int(self.screen_width/2)+30].fill(0)
        return cross

if __name__=='__main__':
    # print("Hello, World!")
    # myProto = ProtoTypeTask("familiar")
    # myProto = ProtoTypeTask("linearface")
    myProto = ProtoTypeTask("preattentive")