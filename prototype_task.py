import os
import time 
import json
import itertools
import tkinter as tk
import numpy as np
import cv2
import random
from preattentive_object import PreattentiveObject
from familiar_object import FamiliarObject
from pynput.keyboard import Controller

keyboard_button = Controller()

def keyboard_A_btn():
    keyboard_button.press('a')
    keyboard_button.release('a')


def keyboard_B_btn():
    keyboard_button.press('b')
    keyboard_button.release('b')


def keyboard_C_btn():
    keyboard_button.press('c')
    keyboard_button.release('c')


def keyboard_D_btn():
    keyboard_button.press('d')
    keyboard_button.release('d')


class ProtoTypeTask:
    def __init__(self, task):
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        self.background = np.zeros((self.screen_height,self.screen_width), dtype='uint8')
        self.background_color = np.zeros((self.screen_height,self.screen_width,3), dtype='uint8')
        self.ready = False
        self.task = task
        self.preattentive_list = os.listdir('IdentiGaze-Stimuli')
        self.preattentive_object = PreattentiveObject(self.screen_width, self.screen_height, 'black')
        self.familiar_object = FamiliarObject(self.screen_width, self.screen_height, 'black')
        self.gather_information()
        # self.preattentive_object.set_set_size(2)

        if self.task=='familiar':
            self.familiar()
        elif self.task=='preattentive':
            self.preattentive()
        elif self.task=='linearface':
            self.linearface()

    def preattentive(self):
        Interface_x = 0
        Interface_y = 0
        bg = self.background_color.copy()
        # self.preattentive_object.random_control = False
        cv2.putText(bg, "Please focus on an odd element (shape, color, size, orientation).", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        cv2.putText(bg, "Press any key to start the test", (100,300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow('image', Interface_x, Interface_y)
        cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('image', bg)
        cv2.waitKey(0) & 0xff
        stimuli_amount = 100
        task_count = 0
        self.make_task_list(stimuli_amount, 5)
        cross_bg = self.get_cross()
        empty_bg = self.background.copy()
        log_data = {}

        while task_count<stimuli_amount:
            if self.ready == 2:
                if len(self.preattentive_object.grid_index_list)==0:
                    self.preattentive_object.grid_index_list = list(range(self.preattentive_object.set_size**2))
                # bg, _, = self.preattentive_object.stimuli_hue(target_index)
                self.preattentive_object.set_set_size(random.choice(list(range(4,7))))
                target_index = random.choice(self.preattentive_object.grid_index_list)
                self.preattentive_object.grid_index_list.remove(target_index)
                # bg, _ = self.preattentive_object.stimuli_hue(target_index)
                task = self.choice_task()
                if task == 0:
                    bg, log, = self.preattentive_object.stimuli_shape(target_index)
                elif task == 1:
                    bg, log, = self.preattentive_object.stimuli_size(target_index)
                elif task == 2:
                    bg, log, = self.preattentive_object.stimuli_hue(target_index)
                elif task == 3:
                    bg, log, = self.preattentive_object.stimuli_brightness(target_index)
                elif task == 4:
                    bg, log, = self.preattentive_object.stimuli_orientation(target_index)
            start_time = time.time()
            while True:
                if self.ready==0:
                    cv2.imshow('image', cross_bg)
                    key = cv2.waitKey(60) & 0xff
                    time_passed = time.time() - start_time
                    if time_passed > 0.8:
                        keyboard_A_btn()
                        self.ready = 1
                        break

                elif self.ready == 1:
                    cv2.imshow('image', empty_bg)
                    key = cv2.waitKey(60) & 0xff
                    time_passed = time.time() - start_time
                    if time_passed > 0.2:
                        keyboard_B_btn()
                        self.ready = 2
                        break

                else:
                    cv2.imshow('image', bg)
                    key = cv2.waitKey(60) & 0xff
                    time_passed = time.time() - start_time
                    if time_passed > self.stimuli_time:
                        keyboard_C_btn()
                        self.ready = 0
                        task_count += 1
                        log_data[task_count] = log
                        break

        cv2.destroyAllWindows()
        self.save_log_data(log_data)


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
        cv2.putText(bg, "Please focus on an odd element.", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        cv2.putText(bg, "Press the key to start the test", (100,300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
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
    
    def make_task_list(self, total_stimuli, task):
        task_list = [[i]*(total_stimuli//task) for i in range(task)]
        flatten = list(itertools.chain(*task_list))
        random.shuffle(flatten)
        self.task_list = flatten

    def choice_task(self):
        this_task = random.choice(self.task_list)
        self.task_list.remove(this_task)
        if len(self.task_list) == 0:
            print("End")
        return this_task
    
    def save_log_data(self, log_data):
        if self.task and self.participant and self.stimuli_time:
            tm = time.gmtime(time.time())
            with open(os.path.join('results',f'{tm.tm_mon}.{tm.tm_mday}.{tm.tm_hour}.{tm.tm_min}.{tm.tm_sec}_{self.task}_{self.participant}_{self.stimuli_time}.json'), 'w') as f:
                json.dump(log_data, f)
        else:
            raise Exception("뭔가 문제가 생겼어요. task / participant / stimuli 를 제대로 적어주시지 않았나봐요.")

    def gather_information(self):
        participant = str(input("Enter your name in English: "))
        stimuli = int(input("Enter the number with your task. 1(0.5 sec), 2(0.7 sec) : "))
        if stimuli == 1:
            self.stimuli_time = 0.5
        elif stimuli == 2:
            self.stimuli_time = 0.7
        else:
            raise Exception("Enter the number for 1 or 2.")
        self.participant = participant

if __name__=='__main__':
    # print("Hello, World!")
    # myProto = ProtoTypeTask("familiar")
    # myProto = ProtoTypeTask("linearface")
    myProto = ProtoTypeTask("preattentive")
    # task_list = ([i]*(100//5) for i in range(5))
    # print(list(itertools.chain(*task_list)))
    # print(time.localtime(time.time()))