import os
import numpy as np
import cv2
import random

class FamiliarObject:
    def __init__(self, screen_width, screen_height, bg_color='black'):
        self.nonfamiliar_list = os.listdir('nonfamiliar')
        self.familiar_list = os.listdir('familiar')
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.background = np.zeros((self.screen_height,self.screen_width), dtype='uint8')
        self.set_bg_color(bg_color)

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

        return bg

    def set_bg_color(self, bg_color):
        # Setting background color
        self.bg_color = bg_color
        if self.bg_color == 'white':
            self.background.fill(255)
        elif isinstance(self.bg_color, int) and self.bg_color<=255:
            self.background.fill(self.bg_color)
        elif self.bg_color == 'black':
            pass
        else:
            raise Exception("bg_color should be 'white' or 'black' or any integer under 255.")
    
    def cointoss(self):
        toss_list = ['up', 'down']
        return random.choice(toss_list)

    def dicetoss(self):
        toss_list = [True, False, False, False, False, False]
        return random.choice(toss_list)

if __name__=='__main__':
    myfamiliar = FamiliarObject(100,100)
    print("Hello, World!")
    # myProto = ProtoTypeTask("familiar")
    # myProto = ProtoTypeTask("linearface")