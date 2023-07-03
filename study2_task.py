from audioop import cross
from operator import index
import os
import time 
import json
import itertools
import tkinter as tk
import numpy as np
import cv2
import json
from pynput.keyboard import Controller
from preattentive_second import PreattentiveObjectSecond


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


class Study2Task:
    def __init__(self):
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        self.background = np.zeros((self.screen_height,self.screen_width), dtype='uint8')
        self.background_color = np.zeros((self.screen_height,self.screen_width,3), dtype='uint8')
        self.ready = False
        # self.preattentive_second = PreattentiveObjectSecond(1980,1080, 'black')
        self.preattentive_second = PreattentiveObjectSecond(self.screen_width,self.screen_height, 'black')
        self.cross = self.get_cross()
        self.gather_information()
        if self.stimuli == 'different':
            self.taskStart_different()
        elif self.stimuli == 'similar':
            self.taskStart_similar()

    def get_cross(self):
        cross = np.zeros((1080,1980,3), dtype=np.uint8)
        cross[int(1080/2)-30:int(1080/2)+30,int(1980/2)-5:int(1980/2)+5].fill(255)
        cross[int(1080/2)-5:int(1080/2)+5,int(1980/2)-30:int(1980/2)+30].fill(255)
        return cross

    def gather_information(self):
        participant = str(input("Enter participant number: "))
        session = str(input("Enter session number: "))
        stimuli = str(input("Enter the kind of stimuli (different or similar) : "))
        if stimuli == 'different':
            self.stimuli = stimuli
        elif stimuli == 'similar':
            self.stimuli = stimuli
        else:
            raise Exception("Enter the string 'different' or 'similar'.")
        rest = int(input("Rest -> 1 or 2: "))
        if rest == 1:
            if stimuli == 'different':
                indexList = list(range(0,128))
            elif stimuli == 'similar':
                indexList = list(range(0,98))
        elif rest == 2:
            if stimuli == 'different':
                indexList = list(range(128,256))
            elif stimuli == 'similar':
                indexList = list(range(98,196))
        else:
            raise Exception("Enter the integer '1' or '2'.")

        self.participant = participant
        self.session = session
        self.indexList = indexList
        if stimuli == 'different':
            with open(os.path.join('madeSet', f'{participant}', f'session{session}', 'different_set.json')) as f:
                indexData = json.load(f)
        elif stimuli == 'similar':
            with open(os.path.join('madeSet', f'{participant}', f'session{session}', 'similar_set.json')) as f:
                indexData = json.load(f)
        self.indexData = indexData

    def taskStart_different(self):
        Interface_x = 0
        Interface_y = 0
        bg = self.background_color.copy()
        # self.preattentive_object.random_control = False
        cv2.putText(bg, "Please focus on every odd element (shape, brightness, hue, size).", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        cv2.putText(bg, "Press 'z' key to start the test", (100,300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow('image', Interface_x, Interface_y)
        cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        while True:
            cv2.imshow('image', bg)
            key = cv2.waitKey(60) & 0xff
            if key == ord('z'):
                break
            else:
                continue
        cross_bg = self.get_cross()
        empty_bg = self.background_color.copy()
        task_count = 0
        while task_count<128:
            if self.ready == 2:
                index = self.indexList[task_count]
                levelIndex = self.indexData[str(index)]['level_index']
                targetList = self.indexData[str(index)]['target_list']
                bg = self.preattentive_second.stimuli_shape(targetList, levelIndex)
                

            start_time = time.time()
            while True:
                if self.ready==0:
                    cross_copy = cross_bg.copy()
                    (textW, textH),_ = cv2.getTextSize(f"{task_count+1}/128", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    cv2.putText(cross_copy, f"{task_count+1}/128", (int(990-textW/2),620), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.imshow('image', cross_copy)
                    key = cv2.waitKey(60) & 0xff
                    time_passed = time.time() - start_time
                    if time_passed > 0.8:
                        # print(time_passed)
                        keyboard_A_btn()
                        self.ready = 1
                        break

                elif self.ready == 1:
                    cv2.imshow('image', empty_bg)
                    key = cv2.waitKey(60) & 0xff
                    time_passed = time.time() - start_time
                    if time_passed > 0.2:
                        # print(time_passed)
                        keyboard_B_btn()
                        self.ready = 2
                        break

                elif self.ready == 3:
                    cv2.imshow('image', empty_bg)
                    key = cv2.waitKey(60) & 0xff
                    time_passed = time.time() - start_time
                    if time_passed > 0.3:
                        # print(time_passed)
                        keyboard_D_btn()
                        self.ready = 0
                        break
                else:
                    cv2.imshow('image', bg)
                    key = cv2.waitKey(60) & 0xff
                    time_passed = time.time() - start_time
                    if time_passed > 0.7:
                        # print(time_passed)
                        keyboard_C_btn()
                        self.ready = 3
                        task_count += 1
                        break

        cv2.destroyAllWindows()


    def taskStart_similar(self):
        Interface_x = 0
        Interface_y = 0
        bg = self.background_color.copy()
        # self.preattentive_object.random_control = False
        cv2.putText(bg, "Please focus on every odd element (shape, brightness, hue, size).", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        cv2.putText(bg, "Press 'z' key to start the test", (100,300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow('image', Interface_x, Interface_y)
        cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        while True:
            cv2.imshow('image', bg)
            key = cv2.waitKey(60) & 0xff
            if key == ord('z'):
                break
            else:
                continue
        cross_bg = self.get_cross()
        empty_bg = self.background_color.copy()
        task_count = 0
        while task_count<98:
            if self.ready == 2:
                index = self.indexList[task_count]
                indexData = self.indexData[str(index)]['index_data']
                targetList = self.indexData[str(index)]['target_list']
                visualComponent = self.indexData[str(index)]['stimuli']
                bg = self.preattentive_second.stimuli_similar(visualComponent, targetList, indexData)
                
            start_time = time.time()
            while True:
                if self.ready==0:
                    cross_copy = cross_bg.copy()
                    (textW, textH),_ = cv2.getTextSize(f"{task_count+1}/98", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    cv2.putText(cross_copy, f"{task_count+1}/98", (int(990-textW/2),620), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.imshow('image', cross_copy)
                    key = cv2.waitKey(60) & 0xff
                    time_passed = time.time() - start_time
                    if time_passed > 0.8:
                        # print(time_passed)
                        keyboard_A_btn()
                        self.ready = 1
                        break

                elif self.ready == 1:
                    cv2.imshow('image', empty_bg)
                    key = cv2.waitKey(60) & 0xff
                    time_passed = time.time() - start_time
                    if time_passed > 0.2:
                        # print(time_passed)
                        keyboard_B_btn()
                        self.ready = 2
                        break

                elif self.ready == 3:
                    cv2.imshow('image', empty_bg)
                    key = cv2.waitKey(60) & 0xff
                    time_passed = time.time() - start_time
                    if time_passed > 0.3:
                        # print(time_passed)
                        keyboard_D_btn()
                        self.ready = 0
                        break
                else:
                    cv2.imshow('image', bg)
                    key = cv2.waitKey(60) & 0xff
                    time_passed = time.time() - start_time
                    if time_passed > 0.7:
                        # print(time_passed)
                        keyboard_C_btn()
                        self.ready = 3
                        task_count += 1
                        break

        cv2.destroyAllWindows()




if __name__ == "__main__":
    myProto = Study2Task()
