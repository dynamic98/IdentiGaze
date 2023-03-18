import os
import numpy as np
import random
import math
import cv2

class PreattentiveObject:
    def __init__(self, screen_width, screen_height, bg_color='black'):
        # Basic background color is white
        # Basic FOV size is 600*600
        # Basic set size is 6*6
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.background = np.zeros((self.screen_height,self.screen_width,3), dtype='uint8')

        self.size_list = [30, 40, 50, 60, 70]
        self.shape_list = [1,2,3,4,5]
        self.hue_list = ['red','yellow','green','blue']
        self.hue_level = ['very low', 'low', 'mid', 'high', 'very high']
        self.brightness_level = ['very low', 'low', 'mid', 'high', 'very high']
        self.orientation_list = [-30, -15, 0, 15, 30]

        self.set_bg_color(bg_color)
        self.set_FOV(600,600)
        self.set_set_size(6)
        self.set_random_control()

    def stimuli_shape(self, target_num):
        size_list = [30, 40, 50, 60, 70]
        color_task = ['hue', 'brightness']
        color_level = ['very low', 'low', 'mid', 'high', 'very high']
        if self.random_control==True:
            element_size = random.choice(size_list)
            color = self.random_color()
            # color = self.convert_color(random.choice(color_task), random.choice(color_level))
        else:
            element_size = 50
            color = (255,255,255)
        shape_distractor = self.select_shape()
        shape_target = self.select_shape()
        grid_list = self.calc_grid(element_size)
        # random.shuffle(grid_list)
        bg = self.background.copy()
        for n, i in enumerate(grid_list):
            if n==target_num:
                # target object
                bg = self.shape_draw(shape_target, bg, i[0], i[1], element_size, color)
                target_x = i[0]
                target_y = i[1]
            else:
                bg= self.shape_draw(shape_distractor, bg, i[0], i[1], element_size, color)
        stimuli_log = {'task':'shape', 'shape_target':shape_target, 'shape_distractor':shape_distractor,
                       'set_size':self.set_size, 'target_cnt':(target_x, target_y), 'target_size':element_size,
                       'distractor_size':element_size, 'target_color':color, 'distractor_color':color, 
                       'target_orientation':None, 'distractor_orientation':None}
        return bg, stimuli_log
                
    def stimuli_size(self, target_num):
        shape_list = [1,2,3,4,5]
        color_task = ['hue', 'brightness']
        color_level = ['very low', 'low', 'mid', 'high', 'very high']
        if self.random_control==True:
            shape = random.choice(shape_list)
            color = self.random_color()
            # color = self.convert_color(random.choice(color_task), random.choice(color_level))
        else:
            shape = 1
            color = (255,255,255)
        size_distractor = self.select_size()
        size_target = self.select_size()
        grid_list = self.calc_grid(size_distractor)
        bg = self.background.copy()
        for n, i in enumerate(grid_list):
            if n==target_num:
                # target object
                bg = self.shape_draw(shape, bg, i[0], i[1], size_target, color)
                target_x = i[0]
                target_y = i[1]
            else:
                bg = self.shape_draw(shape, bg, i[0], i[1], size_distractor, color)
        stimuli_log = {'task':'size', 'shape_target':shape, 'shape_distractor':shape,
                'set_size':self.set_size, 'target_cnt':(target_x, target_y), 'target_size':size_target,
                'distractor_size':size_distractor, 'target_color':color, 'distractor_color':color,
                'target_orientation':None, 'distractor_orientation':None}

        return bg, stimuli_log
    
    def stimuli_hue(self, target_num):
        shape_list = [1,2,3,4,5]
        size_list = [30, 40, 50, 60, 70]
        if self.random_control==True:
            element_size = random.choice(size_list)
            shape = random.choice(shape_list)
        else:
            element_size = 50
            shape = 1
        hue = self.select_hue()
        hue_level_distractor = self.select_hue_level()
        color_distactor = self.convert_hue(hue, hue_level_distractor)
        hue_level_target = self.select_hue_level()
        color_target = self.convert_hue(hue, hue_level_target)
        grid_list = self.calc_grid(element_size)
        bg = self.background.copy()
        for n, i in enumerate(grid_list):
            if n==target_num:
                # target object
                bg = self.shape_draw(shape, bg, i[0], i[1], element_size, color_target)
                target_x = i[0]
                target_y = i[1]
            else:
                bg = self.shape_draw(shape, bg, i[0], i[1], element_size, color_distactor)
        stimuli_log = {'task':'hue', 'shape_target':shape, 'shape_distractor':shape,
                'set_size':self.set_size, 'target_cnt':(target_x, target_y), 'target_size':element_size,
                'distractor_size':element_size, 'target_color':color_target, 'distractor_color':color_distactor,
                'target_orientation':None, 'distractor_orientation':None}

        return bg, stimuli_log

    def stimuli_brightness(self, target_num):
        shape_list = [1,2,3,4,5]
        size_list = [30, 40, 50, 60, 70]
        if self.random_control==True:
            element_size = random.choice(size_list)
            shape = random.choice(shape_list)
        else:
            element_size = 50
            shape = 1
        brightness_distractor = self.select_brightness_level()
        color_distactor = self.convert_brightness(brightness_distractor)
        brightness_target = self.select_brightness_level()
        color_target = self.convert_brightness(brightness_target)
        grid_list = self.calc_grid(element_size)
        bg = self.background.copy()
        for n, i in enumerate(grid_list):
            if n==target_num:
                # target object
                bg = self.shape_draw(shape, bg, i[0], i[1], element_size, color_target)
                target_x = i[0]
                target_y = i[1]
            else:
                bg = self.shape_draw(shape, bg, i[0], i[1], element_size, color_distactor)
        stimuli_log = {'task':'brightness', 'shape_target':shape, 'shape_distractor':shape,
                'set_size':self.set_size, 'target_cnt':(target_x, target_y), 'target_size':element_size,
                'distractor_size':element_size, 'target_color':color_target, 'distractor_color':color_distactor,
                'target_orientation':None, 'distractor_orientation':None}

        return bg, stimuli_log
    
    def stimuli_orientation(self, target_num):
        size_list = [30, 40, 50, 60, 70]
        color_task = ['hue', 'brightness']
        color_level = ['very low', 'low', 'mid', 'high', 'very high']
        if self.random_control==True:
            element_size = random.choice(size_list)
            color = self.random_color()
            # color = self.convert_color(random.choice(color_task), random.choice(color_level))
        else:
            element_size = 50
            color = (255,255,255)
        orientation_distractor = self.select_orientation()
        orientation_target = self.select_orientation()
        grid_list = self.calc_grid(element_size)
        bg = self.background.copy()
        for n, i in enumerate(grid_list):
            if n==target_num:
                # target object
                bg = self.orientation(bg, i[0], i[1], element_size, orientation_target, color)
                target_x = i[0]
                target_y = i[1]
            else:
                bg = self.orientation(bg, i[0], i[1], element_size, orientation_distractor, color)
        stimuli_log = {'task':'orientation', 'shape_target':'orientation', 'shape_distractor':'orientation',
                'set_size':self.set_size, 'target_cnt':(target_x, target_y), 'target_size':element_size,
                'distractor_size':element_size, 'target_color':color, 'distractor_color':color,
                'target_orientation':orientation_target, 'distractor_orientation':orientation_distractor}

        return bg, stimuli_log


    def test(self):
        # bg = self.cross(self.background, 800, 500, 50, (100,200,50))
        bg = self.background.copy()
        bg = cv2.rectangle(bg, (self.FOV_x1, self.FOV_y1), (self.FOV_x2, self.FOV_y2), (0,0,0), 3)
        # bg = self.triangle(bg, 200,300,50,(100,200,50))
        grid_list = self.calc_grid(50)
        for grid in grid_list:
        #     bg = self.triangle(bg, grid[0], grid[1], 50, (100,200,50))
            # bg = self.cross(bg, grid[0], grid[1], 50, (100,200,50))
            # bg = self.rectangle(bg, grid[0], grid[1], 50, (100,200,50))
            bg = self.hexagon(bg, grid[0], grid[1], 50, (100,200,50))
        cv2.imshow('image',bg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def calc_grid(self, distractor_size):
        interval_size_width = int((self.FOV_width-(distractor_size*self.set_size))/(self.set_size-1))
        interval_size_height = int((self.FOV_height-(distractor_size*self.set_size))/(self.set_size-1))
        grid_list = []
        for i in range(self.set_size):
            for j in range(self.set_size):
                x = self.FOV_x1+ i*(distractor_size+interval_size_width)+distractor_size/2
                y = self.FOV_y1+ j*(distractor_size+interval_size_height)+distractor_size/2
                grid_list.append([int(x),int(y)])
        return grid_list

    def set_FOV(self, width, height):
        # Setting display size of stimuli
        if width > self.screen_width or height > self.screen_height:
            raise Exception("FOV is larger than your screen size")
        self.FOV_width = width
        self.FOV_height = height
        self.FOV_x1 = int(self.screen_width/2-width/2)
        self.FOV_x2 = int(self.screen_width/2+width/2)
        self.FOV_y1 = int(self.screen_height/2-height/2)
        self.FOV_y2 = int(self.screen_height/2+height/2)

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

    def set_set_size(self, set_size):
        # Actual set size is set_size**2
        self.set_size = set_size
        self.grid_index_list = list(range(set_size**2))
    
    def shape_cross_draw(self, shape, bg, x1, y1, size, color):
        if shape == 1:
            bg = self.rectangle(bg, x1, y1, size, color)
        else:
            bg = self.cross_detail(shape, bg, x1, y1, size, color)
        return bg

    def shape_draw(self, shape, bg, x1, y1, size, color):
        if shape == 1:
            bg = self.triangle(bg, x1, y1, size, color)
        elif shape == 2:
            bg = self.rectangle(bg, x1, y1, size, color)
        elif shape == 3:
            bg = self.pentagon(bg, x1, y1, size, color)
        elif shape == 4:
            bg = self.hexagon(bg, x1, y1, size, color)
        elif shape == 5:
            bg = cv2.circle(bg, (x1, y1), int(size/2), color, -1)
        else:
            raise Exception("shape is not in defined set")
        return bg

    def cross_detail(self, shape, bg, x1, y1, size, color):
        # color is triplet tuple. e.g.,(255,255,255)
        # color tuple contains (Blue, Green, Red)
        ratio = shape*2
        for i, channel in enumerate(color):
            bg[int(y1-size/2):int(y1+size/2),int(x1-size/ratio):int(x1+size/ratio),i].fill(channel)
            bg[int(y1-size/ratio):int(y1+size/ratio),int(x1-size/2):int(x1+size/2),i].fill(channel)
        return bg

    def cross(self, bg, x1, y1, size, color):
        # color is triplet tuple. e.g.,(255,255,255)
        # color tuple contains (Blue, Green, Red)
        for i, channel in enumerate(color):
            bg[int(y1-size/2):int(y1+size/2),int(x1-size/8):int(x1+size/8),i].fill(channel)
            bg[int(y1-size/8):int(y1+size/8),int(x1-size/2):int(x1+size/2),i].fill(channel)
        return bg

    def rectangle(self, bg, x1, y1, size, color):
        bg = cv2.rectangle(bg, (int(x1-size/2),int(y1-size/2)), (int(x1+size/2),int(y1+size/2)), color, -1)
        return bg

    def triangle(self, bg, x1, y1, size, color):
        radius = size/2
        ax = x1 - radius*math.cos(math.pi*(30/180))
        ay = y1 + radius*math.sin(math.pi*(30/180))
        bx = x1 + radius*math.cos(math.pi*(30/180))
        by = y1 + radius*math.sin(math.pi*(30/180))
        cx = x1
        cy = y1 - size/2
        triangle_cnt = np.array([(int(ax), int(ay)), (int(bx), int(by)), (int(cx), int(cy))])
        bg = cv2.drawContours(bg, [triangle_cnt], 0, color, -1)
        return bg

    def pentagon (self, bg, x1, y1, size, color):
        radius = size/2
        ax = x1
        ay = y1 - radius
        pentagon_cnt = [(int(ax), int(ay))]
        for i in range(1,5):
            rotated_coordinate = self.rotate((x1,y1),(ax,ay),math.pi*(i*72/180))
            pentagon_cnt.append(rotated_coordinate)
        pentagon_cnt = np.array(pentagon_cnt)
        bg = cv2.drawContours(bg, [pentagon_cnt], 0, color, -1)
        return bg
    
    def hexagon (self, bg, x1, y1, size, color):
        radius = size/2
        ax = x1
        ay = y1 - radius
        hexagon_cnt = [(int(ax), int(ay))]
        for i in range(1,6):
            rotated_coordinate = self.rotate((x1,y1),(ax,ay),math.pi*(i*60/180))
            hexagon_cnt.append(rotated_coordinate)
        hexagon_cnt = np.array(hexagon_cnt)
        bg = cv2.drawContours(bg, [hexagon_cnt], 0, color, -1)
        return bg

    def orientation(self, bg, x1, y1, size, direction, color):
        origin = (x1, y1)
        half_length = size/2
        half_thickness = 2.5
        direction = math.pi*(direction/180)
        ax = x1-half_thickness
        ay = y1-half_length
        bx = x1+half_thickness
        by = y1+half_length
        if direction == 0:
            orientation_cnt = np.array([(int(ax), int(ay)), (int(ax), int(by)), (int(bx), int(by)), (int(bx), int(ay))])
        else:
            orientation_cnt = np.array([self.rotate(origin, (ax, ay), direction),self.rotate(origin, (ax, by), direction),
                                        self.rotate(origin, (bx, by), direction),self.rotate(origin, (bx, ay), direction)])
        bg = cv2.drawContours(bg, [orientation_cnt], 0, color, -1)
        return bg

    def rotate(self, origin, point, angle):
        ox, oy = origin
        px, py = point
        qx = ox + math.cos(angle)*(px-ox) - math.sin(angle)*(py-oy)
        qy = oy + math.sin(angle)*(px-ox) + math.cos(angle)*(py-oy)
        return (int(qx), int(qy))

    def set_random_control(self, random:bool=True):
        self.random_control = random

    def select_hue(self):
        if len(self.hue_list)==1:
            self.hue_list = ['red','yellow','green','blue']
        hue = random.choice(self.hue_list)
        self.hue_list.remove(hue)
        return hue

    def select_hue_level(self):
        if len(self.hue_level)==1:
            self.hue_level = ['very low', 'low', 'mid', 'high', 'very high']
        hue_level = random.choice(self.hue_level)
        self.hue_level.remove(hue_level)
        return hue_level

    def select_brightness_level(self):
        if len(self.brightness_level)==1:
            self.brightness_level = ['very low', 'low', 'mid', 'high', 'very high']
        brightness_level = random.choice(self.brightness_level)
        self.brightness_level.remove(brightness_level)
        return brightness_level

    def select_shape(self):
        if len(self.shape_list)==1:
            self.shape_list = [1,2,3,4,5]
        shape = random.choice(self.shape_list)
        self.shape_list.remove(shape)
        return shape
    
    def select_size(self):
        if len(self.size_list)==1:
            self.size_list = [30, 40, 50, 60, 70]
        size = random.choice(self.size_list)
        self.size_list.remove(size)
        return size

    def select_orientation(self):
        if len(self.orientation_list)==1:
            self.orientation_list = [-30, -15, 0, 15, 30]
        orientation = random.choice(self.orientation_list)
        self.orientation_list.remove(orientation)
        return orientation

    def random_color(self):
        task = random.choice(list(range(2)))
        if task==0: 
            hue = random.choice(['red','yellow','green','blue'])
            level = random.choice(['very low', 'low', 'mid', 'high', 'very high'])
            color = self.convert_hue(hue, level)
        else:
            level = random.choice(['very low', 'low', 'mid', 'high', 'very high'])
            color = self.convert_brightness(level)
        return color

    def convert_hue(self, hue:str, level:str):
        # task: 'hue', 'brightness'
        # level: 'very low', 'low', 'mid', 'high', 'very high'
        if hue == 'red':
            if level == 'very low':
                color = (62, 101, 127)
            elif level == 'low':
                color = (62, 82, 127)
            elif level == 'mid':
                color = (62, 62, 127)
            elif level == 'high':
                color = (82, 62, 127)
            elif level == 'very high':
                color = (101, 62, 127)
        elif hue == 'yellow':
            if level == 'very low':
                color = (62, 127, 88)
            elif level == 'low':
                color = (62, 127, 107)
            elif level == 'mid':
                color = (62, 127, 127)
            elif level == 'high':
                color = (62, 107, 127)
            elif level == 'very high':
                color = (62, 88, 127)
        elif hue == 'green':
            if level == 'very low':
                color = (101, 127, 62)
            elif level == 'low':
                color = (82, 127, 62)
            elif level == 'mid':
                color = (62, 127, 62)
            elif level == 'high':
                color = (62, 127, 82)
            elif level == 'very high':
                color = (62, 127, 101)
        elif hue == 'blue':
            if level == 'very low':
                color = (127, 62, 101)
            elif level == 'low':
                color = (127, 62, 82)
            elif level == 'mid':
                color = (127, 62, 62)
            elif level == 'high':
                color = (127, 82, 62)
            elif level == 'very high':
                color = (127, 101, 62)
        return color
    
    def convert_brightness(self, level:str):
        # level: 'very low', 'low', 'mid', 'high', 'very high'
        if level == 'very low':
            color = (51, 51, 51)
        elif level == 'low':
            color = (102, 102, 102)
        elif level == 'mid':
            color = (153, 153, 153)
        elif level == 'high':
            color = (204, 204, 204)
        elif level == 'very high':
            color = (255, 255, 255)
        return color

    def convert_color(self, task:str, level:str):
        # task: 'hue', 'brightness'
        # level: 'very low', 'low', 'mid', 'high', 'very high'
        if task == 'hue':
            if level == 'very low':
                color = (101, 63, 127)
            elif level == 'low':
                color = (82, 63, 127)
            elif level == 'mid':
                color = (63, 63, 127)
            elif level == 'high':
                color = (63, 82, 127)
            elif level == 'very high':
                color = (63, 101, 127)
        elif task == 'brightness':
            if level == 'very low':
                color = (51, 51, 51)
            elif level == 'low':
                color = (102, 102, 102)
            elif level == 'mid':
                color = (153, 153, 153)
            elif level == 'high':
                color = (204, 204, 204)
            elif level == 'very high':
                color = (255, 255, 255)
        return color
        
if __name__=='__main__':
    myPreattentiveObject = PreattentiveObject(1280,1080,254)
    # myPreattentiveObject.set_set_size(4)
    # myPreattentiveObject.set_set_size(8)
    myPreattentiveObject.test()
    # Saturation : 129
    # Brightness : 127
    # sample_image = np.zeros((1000,1000,3), dtype='uint8')
    # sample_image[:,:,1].fill(129)
    # sample_image[:,:,2].fill(127)
    # for hue in [0, 30, 60, 120]:
    #     print("=============")
    #     print(hue)
    #     for i in range(-2,3,1):
    #         change_hue = hue - i*9
    #         if change_hue < 0:
    #             change_hue = 180+change_hue
    #         change_hue = int(change_hue)
    #         sample_image[:,:,0].fill(change_hue)
    #         print(cv2.cvtColor(sample_image, cv2.COLOR_HSV2BGR)[0,0,:])


