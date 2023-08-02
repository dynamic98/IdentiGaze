import os
import numpy as np
import random
import math
import cv2
import matplotlib.pyplot as plt
import ML_util


class PreattentiveObjectSecond:
    def __init__(self, screen_width, screen_height, bg_color='black'):
        # Basic background color is black
        # Basic FOV size is 600*600
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.background = np.zeros((self.screen_height,self.screen_width,3), dtype='uint8')

        self.size_list = [30, 40, 50, 60, 70]
        self.shape_list = [1,2,3,4,5]
        self.hue = "blue"
        self.hue_level = ['very low', 'low', 'mid', 'high', 'very high']
        self.brightness_level = ['very low', 'low', 'mid', 'high', 'very high']

        self.set_bg_color(bg_color)
        self.set_FOV(600,600)
        self.levels = self.level_combinations()
        self.count = 1

        # self.set_random_control()

    def stimuli_shape(self, targetList, levelIndex):
        size_list = [30, 40, 50, 60, 70]
        color_task = ['hue', 'brightness']
        color_level = ['very low', 'low', 'mid', 'high', 'very high']

        color = (255,153,153)
        grid_list = self.calc_grid()
        bg = self.background.copy()

        targetShape = self.levels[levelIndex][0]
        targetSize = self.levels[levelIndex][1]
        targetHue_str = color_level[self.levels[levelIndex][2]]
        targetBrightness_str = color_level[self.levels[levelIndex][3]]

        targetHue = self.convert_hue(targetHue_str)
        targetBrightness = self.convert_brightness(targetBrightness_str)

        for n, i in enumerate(grid_list):
            if targetList[0] == n:
                bg= self.shape_draw(targetShape, bg, i[0], i[1], 70, color)
            elif targetList[1] == n:
                bg= self.shape_draw(5, bg, i[0], i[1], 70, targetHue)
            elif targetList[2] == n:
                bg= self.shape_draw(5, bg, i[0], i[1], 70, targetBrightness)
            elif targetList[3] == n:
                bg= self.shape_draw(5, bg, i[0], i[1], targetSize, color)
            else:
                bg= self.shape_draw(5, bg, i[0], i[1], 70, color)
        self.count += 1
        return bg

    def stimuli_similar(self,stimuli, targetList, indexData):
        shape_list = [1,2,3,4,5]
        size_list = [30, 40, 50, 60, 70]
        color_task = ['hue', 'brightness']
        color_level = ['very low', 'low', 'mid', 'high', 'very high']

        color = (255,153,153)
        grid_list = self.calc_grid()
        bg = self.background.copy()
        # targetShape = self.levels[levelIndex][0]
        # targetSize = self.levels[levelIndex][1]
        # targetHue_str = color_level[self.levels[levelIndex][2]]
        # targetBrightness_str = color_level[self.levels[levelIndex][3]]

        # targetHue = self.convert_hue(targetHue_str)
        # targetBrightness = self.convert_brightness(targetBrightness_str)
        if stimuli == 'shape':
            # print(f"indexData: {indexData}, targetList: {targetList}")
            for n, i in enumerate(grid_list):
                if n in targetList:
                    index_targetList = targetList.index(n)
                    targetShape = shape_list[indexData[index_targetList]]
                    # print(f"n:{n}, index:{indexData[index_targetList]}, targetShape:{targetShape}")
                    bg= self.shape_draw(targetShape, bg, i[0], i[1], 70, color)
                else:
                    bg= self.shape_draw(5, bg, i[0], i[1], 70, color)
        
        elif stimuli == 'size':
            for n, i in enumerate(grid_list):
                if n in targetList:
                    index_targetList = targetList.index(n)
                    targetSize = size_list[indexData[index_targetList]]
                    bg= self.shape_draw(5, bg, i[0], i[1], targetSize, color)
                else:
                    bg= self.shape_draw(5, bg, i[0], i[1], 70, color)

        elif stimuli == 'hue':
            for n, i in enumerate(grid_list):
                if n in targetList:
                    index_targetList = targetList.index(n)
                    targetHue = self.convert_hue(color_level[indexData[index_targetList]])
                    bg= self.shape_draw(5, bg, i[0], i[1], 70, targetHue)
                else:
                    bg= self.shape_draw(5, bg, i[0], i[1], 70, color)

        elif stimuli == 'brightness':
            for n, i in enumerate(grid_list):
                if n in targetList:
                    index_targetList = targetList.index(n)
                    targetBrightness = self.convert_brightness(color_level[indexData[index_targetList]])
                    bg= self.shape_draw(5, bg, i[0], i[1], 70, targetBrightness)
                else:
                    bg= self.shape_draw(5, bg, i[0], i[1], 70, color)

        self.count += 1
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
        radius = size/2
        ax = x1
        ay = y1 - radius
        rectangle_cnt = []
        for i in range(1,5):
            rotated_coordinate = self.rotate((x1,y1),(ax,ay),math.pi*((i*90+45)/180))
            rectangle_cnt.append(rotated_coordinate)
        rectangle_cnt = np.array(rectangle_cnt)
        bg = cv2.drawContours(bg, [rectangle_cnt], 0, color, -1)

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
    
    def rotate(self, origin, point, angle):
        ox, oy = origin
        px, py = point
        qx = ox + math.cos(angle)*(px-ox) - math.sin(angle)*(py-oy)
        qy = oy + math.sin(angle)*(px-ox) + math.cos(angle)*(py-oy)
        return (int(qx), int(qy))

    def calc_grid(self):
        radius = min(self.FOV_width, self.FOV_height)/2
        center_x = self.screen_width/2
        center_y = self.screen_height/2
        blockAngle = math.pi*22.5/180
        biasAngle = math.pi*11.25/180
        grid_list = []

        for i in range(16):
            angle = i*blockAngle+biasAngle
            x = radius*math.cos(angle) + center_x
            y = radius*math.sin(angle) + center_y
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

    def convert_brightness(self, level:str):
        # level: 'very low', 'low', 'mid', 'high', 'very high'
        if level == 'very low':
            color = (255, 102, 102)
        elif level == 'low':
            color = (255, 115, 115)
        elif level == 'mid':
            color = (255, 128, 128)
        elif level == 'high':
            color = (255, 140, 140)
        elif level == 'very high':
            color = (255, 153, 153)
        return color

    def convert_hue(self, level:str):
        # level: 'very low', 'low', 'mid', 'high', 'very high'
        if level == 'very low':
            color = (255, 255, 153)
        elif level == 'low':
            color = (255, 230, 153)
        elif level == 'mid':
            color = (255, 204, 153)
        elif level == 'high':
            color = (255, 179, 153)
        elif level == 'very high':
            color = (255, 153, 153)
        return color

    def level_combinations(self):
        return ML_util.cartesian([[1,2,3,4],[30,40,50,60],[0,1,2,3],[0,1,2,3]])

def takeTargetList():
    # Take Target List excluding the nearby elements
    numList = list(range(16))
    targetList = []
    for _ in range(4):
        thisRandom = random.choice(numList)
        targetList.append(thisRandom)
        numList.remove(thisRandom)
        thisRandomLeft = thisRandom - 1
        thisRandomRight = thisRandom + 1
        if thisRandomLeft == -1:
            thisRandomLeft = 15
        if thisRandomRight == 16:
            thisRandomRight = 0
        if thisRandomLeft in numList:
            numList.remove(thisRandomLeft)
        if thisRandomRight in numList:
            numList.remove(thisRandomRight)
    return targetList


if __name__ == "__main__":
    myPreattentive = PreattentiveObjectSecond(1980,1080, 'black')

    cross = np.zeros((1080,1980,3), dtype=np.uint8)
    cross[int(1080/2)-30:int(1080/2)+30,int(1980/2)-5:int(1980/2)+5].fill(255)
    cross[int(1080/2)-5:int(1080/2)+5,int(1980/2)-30:int(1980/2)+30].fill(255)

    black = np.zeros((1080,1980,3), dtype=np.uint8)

    # cv2.namedWindow('image',cv2.WND_PROP_FULLSCREEN)
    cv2.namedWindow('image')
    cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow('image', 0, 0)
    cv2.imshow('image', cross)
    cv2.waitKey(0) & 0xff


    levelIndexList = list(range(256))
    random.shuffle(levelIndexList)

    for levelIndex in levelIndexList:
        targetList = takeTargetList()

        cross_copy = cross.copy()
        (textW, textH),_ = cv2.getTextSize(f"{myPreattentive.count}/256", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.putText(cross_copy, f"{myPreattentive.count}/256", (int(990-textW/2),620), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        img = myPreattentive.stimuli_shape(targetList, levelIndex)

        cv2.imshow('image', cross_copy)
        cv2.waitKey(800) & 0xff

        cv2.imshow('image', black)
        cv2.waitKey(200) & 0xff

        cv2.imshow('image', img)
        cv2.waitKey(700) & 0xff

        cv2.imshow('image', black)
        cv2.waitKey(300) & 0xff

    cv2.destroyAllWindows()
    # cv2.imshow("title", img)
    # cv2.destroyAllWindows()
