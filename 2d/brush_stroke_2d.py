from primitives import Primitives
import numpy as np

class BrushStroke2D(Primitives):
    def __init__(self, heightMap):
        self.color = np.zeros(3)
        self.isBrushed = True
        self.R = np.eye(2)
        self.heightMap = heightMap
        self.t = np.zeros(2)

        self.color = np.zeros(3)
        
        # y is row, x is col
        self.h = heightMap.shape[0]
        self.w = heightMap.shape[1]
        
        self.randomize_parameters()

    
    def randomize_parameters(self): 
        self.t = np.array([np.random.uniform(0,self.w), np.random.uniform(0,self.h)])
        theta = np.random.uniform(0, 2*np.pi)
        self.R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        
    def mutate(self):
        t_mutation = np.array([np.random.uniform(-0.05*self.w, 0.05*self.w), np.random.uniform(-0.05*self.h, 0.05*self.h)])
        self.t += t_mutation 
        
    
    def transform(self, x, y):
        return np.dot(self.R, np.array([x,y])) + self.t

    # def optimal_color_sampled(self, targetImage): 
    #     num_samples = 10 
    #     average_color = np.zeros(3)
        
    #     for i in range(num_samples): 
    #         random_coordinate_brush_frame = np.array([np.random.uniform(0,1)*self.h, np.random.uniform(0,1)*self.w])
    #         opacity = self.heightMap[]

    def optimal_color_full(self, targetImage, currentCanvas): 
        optimal_color = np.zeros(3)
        
        num_pixels = 0
        for y in range(self.h):
            for x in range(self.w): 
                opacity = self.heightMap[y, x]
                image_coordinate = self.transform(x, y)
                if 0<=image_coordinate[0]<=self.h and 0<=image_coordinate[1]<=self.w:
                    image_pixel = self.interpolate_color(image_coordinate, targetImage)
                    current_pixel = self.interpolate_color(image_coordinate, currentCanvas)
                    optimal_color += (image_pixel - (1-opacity)*current_pixel)/opacity
                    num_pixels += 1

        self.color = optimal_color / num_pixels
        return self.color

    def bilinear_interoplation(self, coordinate, targetImage): 
        h, w = targetImage.shape[:2]
        min_x = max(np.floor(coordinate[0]), 0)
        max_x = min(np.ceil(coordinate[0]), w-1)
        min_y = max(np.floor(coordinate[1]), 0)
        max_y = min(np.ceil(coordinate[1]), h-1)
        
        min_x_weight = coordinate[0] - min_x
        max_x_weight = 1 - min_x_weight
        min_y_weight = coordinate[1] - min_y
        max_y_weight = 1 - min_y_weight

        return targetImage[min_y,min_x] * min_y*min_x + targetImage[min_y,max_x] * min_y*max_x + targetImage[max_y,max_x] * max_y*max_x + targetImage[max_y,min_x] * max_y*min_x
    
    def interpolate_color(self, coordinate, targetImage): 
        return self.bilinear_interoplation(coordinate, targetImage)