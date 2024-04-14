from primitives import Primitives
from core import *
import numpy as np

class BrushStroke2D(Primitives):
    def __init__(self, heightMap, color=None, theta=None, t=None):
        self.isBrushed = True
        self.theta = np.random.uniform(0, 2*np.pi)
        self.heightMap = heightMap
        
        self.color = color
        
        # y is row, x is col
        self.h = heightMap.shape[0]
        self.w = heightMap.shape[1]
        
        self.cx = self.w/2
        self.cy = self.h/2
        
        if theta == None:
            self.theta = 0
            self.R = np.eye(2)
            self.t = np.zeros(2)
            self.randomize_parameters()
        else:
            self.theta = theta
            self.set_R()
            self.t = t
        
    def set_R(self):
        self.R = np.array([[np.cos(self.theta), -np.sin(self.theta)], 
                           [np.sin(self.theta), np.cos(self.theta)]])
    
    def randomize_parameters(self): 
        self.t = np.array([np.random.uniform(0,self.w), np.random.uniform(0,self.h)])
        
        self.theta = np.random.uniform(0, np.pi*2)
        self.set_R()


    def mutate(self):
        self.theta += np.random.uniform(0, np.pi/3)
        self.set_R()
        t_mutation = np.array([np.random.uniform(-0.05*self.w, 0.05*self.w), np.random.uniform(-0.05*self.h, 0.05*self.h)])
        self.t += t_mutation
        
    def transform(self, x, y):        
        x -= self.cx
        y -= self.cy
        
        transformed = self.R @ np.array([x,y])
        transformed += np.array([self.cx, self.cy])
        
        return transformed + self.t

    # def optimal_color_sampled(self, targetImage): 
    #     num_samples = 10 
    #     average_color = np.zeros(3)
        
    #     for i in range(num_samples): 
    #         random_coordinate_brush_frame = np.array([np.random.uniform(0,1)*self.h, np.random.uniform(0,1)*self.w])
    #         opacity = self.heightMap[]

    def optimal_color_full(self, targetImage, currentCanvas): 
        optimal_color = np.zeros(3)
        
        num_pixels = 0
        img_h, img_w = targetImage.shape
        for y in range(self.h):
            for x in range(self.w): 
                opacity = self.heightMap[y, x]
                image_coordinate = self.transform(x, y)
                if 0 <= image_coordinate[0] <= img_w and 0 <= image_coordinate[1] <= img_h:
                    image_pixel = interpolate_color(image_coordinate, targetImage)
                    current_pixel = interpolate_color(image_coordinate, currentCanvas)
                    optimal_color += (image_pixel - (1-opacity)*current_pixel)/opacity
                    num_pixels += 1

        self.color = optimal_color / num_pixels
        return self.color
    
    def get_patch_error(self, targetImage, currentCanvas, optimalColor):         
        prior_error = 0
        error = 0

        for y in range(self.h):
            for x in range(self.w): 
                opacity = self.heightMap[y, x]
                image_coordinate = self.transform(x, y)
                if 0<=image_coordinate[0]<=self.h and 0<=image_coordinate[1]<=self.w:
                    image_pixel = interpolate_color(image_coordinate, targetImage)
                    current_pixel = interpolate_color(image_coordinate, currentCanvas)
                    new_pixel = opacity * optimalColor + (1 - opacity) * current_pixel #alpha composite single pixel based on color and opacity 
                    error += (new_pixel - image_pixel) ** 2
                    prior_error += (current_pixel - image_pixel) ** 2
        
        return {"newPatchError": error, "oldPatchError": prior_error}
    
    def copy(self):
        new_primitive = BrushStroke2D(self.heightMap, self.color.copy(), self.theta, self.t.copy())
        return new_primitive
