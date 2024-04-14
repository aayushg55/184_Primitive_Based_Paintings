from primitives import Primitives
from core import *
import numpy as np

class BrushStroke2D(Primitives):
    def __init__(self, heightMap, canvas_h, canvas_w, color=np.zeros(3), theta=None, t=None):
        self.isBrushed = True
        self.heightMap = heightMap
        
        self.color = color
        
        # y is row, x is col
        self.canvas_h, self.canvas_w = canvas_h, canvas_w
        self.h = heightMap.shape[0]
        self.w = heightMap.shape[1]
        
        self.cx = self.w/2
        self.cy = self.h/2
        self.c = np.zeros((2,1))
        self.c[0] = self.cx
        self.c[1] = self.cy
        
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
        self.t = np.array([np.random.randint(0,self.canvas_w), np.random.randint(0,self.canvas_h)])
        self.theta = np.random.uniform(0, 180)
        self.set_R()


    def mutate(self):
        self.theta += np.random.uniform(0, 30)
        self.set_R()
        t_mutation = np.array([np.random.randint(int(-0.05*self.canvas_w), int(0.05*self.canvas_w)), np.random.randint(int(-0.05*self.canvas_h), int(0.05*self.canvas_h))])
        self.t += t_mutation
        
    def transform(self, x, y):  
        print(x,  self.cx)      
        x -= self.cx
        y -= self.cy
        
        transformed = self.R @ np.array([x,y])
        transformed += self.c
        
        return transformed + self.t.reshape((2,1))

    # def optimal_color_sampled(self, targetImage): 
    #     num_samples = 10 
    #     average_color = np.zeros(3)
        
    #     for i in range(num_samples): 
    #         random_coordinate_brush_frame = np.array([np.random.uniform(0,1)*self.h, np.random.uniform(0,1)*self.w])
    #         opacity = self.heightMap[]

    def optimal_color_full(self, targetImage, currentCanvas): 
        optimal_color = np.zeros(3)
        
        num_pixels = 0
        img_h, img_w = targetImage.shape[:2]
        for y in range(self.h):
            for x in range(self.w): 
                opacity = self.heightMap[y, x]
                # print(f"opacity {opacity}")
                if opacity == 0.0:
                    continue
                image_coordinate = self.transform(x, y)
                if 0 <= image_coordinate[0] <= img_w and 0 <= image_coordinate[1] <= img_h:
                    image_pixel = interpolate_color(image_coordinate, targetImage)
                    current_pixel = interpolate_color(image_coordinate, currentCanvas)
                    # print(f"image pixel: {image_pixel}, current pixel: {current_pixel}")
                    pix_opt_color = (image_pixel - (1-opacity)*current_pixel)/opacity
                    max_color = np.max(pix_opt_color)
                    if max_color > 1:
                        pix_opt_color /= max_color
                    # print(f"per pix opt color: {pix_opt_color}")
                    # if np.any(pix_opt_color > 1):
                        # print(f"opacity: {opacity}, image_pixel: {image_pixel}, current_pixel: {current_pixel}")
                    
                    optimal_color += pix_opt_color
                    num_pixels += 1
        print(f"num pix: {num_pixels}")
        self.color = optimal_color / num_pixels
        return self.color
    
    def optimal_color_sample(self, targetImage, currentCanvas): 
        optimal_color = np.zeros(3)
        
        num_pixels = 0
        img_h, img_w = targetImage.shape[:2]
        for y in range(self.h):
            for x in range(self.w): 
                opacity = self.heightMap[y, x]
                # print(f"opacity {opacity}")
                if opacity == 0.0:
                    continue
                image_coordinate = self.transform(x, y)
                if 0 <= image_coordinate[0] <= img_w and 0 <= image_coordinate[1] <= img_h:
                    image_pixel = interpolate_color(image_coordinate, targetImage)
                    current_pixel = interpolate_color(image_coordinate, currentCanvas)
                    # print(f"image pixel: {image_pixel}, current pixel: {current_pixel}")
                    pix_opt_color = (image_pixel - (1-opacity)*current_pixel)/opacity
                    max_color = np.max(pix_opt_color)
                    if max_color > 1:
                        pix_opt_color /= max_color
                    # print(f"per pix opt color: {pix_opt_color}")
                    # if np.any(pix_opt_color > 1):
                        # print(f"opacity: {opacity}, image_pixel: {image_pixel}, current_pixel: {current_pixel}")
                    
                    optimal_color += pix_opt_color
                    num_pixels += 1
        print(f"num pix: {num_pixels}")
        self.color = optimal_color / num_pixels
        return self.color

    
    def optimal_color_fast(self, targetImage, currentCanvas):
        # Generate a grid of coordinates
        xs, ys = np.meshgrid(np.arange(self.w, dtype=float), np.arange(self.h, dtype=float))
        print("max_dim", np.max(xs), np.max(ys))
        opacities = self.heightMap

        # Filter out zero-opacity pixels early
        mask = opacities > 0
        xs, ys = xs[mask], ys[mask]
        opacities = opacities[mask]

        # Transform coordinates and apply boundary check
        transformed = self.transform(xs, ys)
        valid_mask = (transformed[0, :] < targetImage.shape[0]) & (transformed[0, :] >= 0) & \
                    (transformed[1, :] < targetImage.shape[1]) & (transformed[1, :] >= 0)
                    
        print("PRINTING SHAPES!!")
        print("transformed before: ", transformed.shape)
        print("valid: ", valid_mask.shape)
        
        transformed = transformed[:, valid_mask]
        filtered_opacities = opacities.flatten()[valid_mask]
        
        # print(np.max(transformed[0, :]), np.max(transformed[1, :]))

        # Interpolate colors
        print("transformed shape, ", transformed.shape)
        image_pixels = np.array([interpolate_color(transformed[:, i], targetImage) for i in range(transformed.shape[1])])
        current_pixels = np.array([interpolate_color(transformed[:, i], currentCanvas) for i in range(transformed.shape[1])])
        
        # Compute optimal colors
        pix_opt_colors = (image_pixels - (1 - filtered_opacities[:, np.newaxis]) * current_pixels) / filtered_opacities[:, np.newaxis]
        
        print("image pixels: ", image_pixels.shape)
        print("current pixels: ", current_pixels.shape)
        print("opacity scaled: ", ((1 - filtered_opacities[:, np.newaxis]) * current_pixels).shape)
        print("pix opt color: ", pix_opt_colors.shape)
        # Clip and normalize if necessary
        max_colors = np.max(pix_opt_colors, axis=1, keepdims=True)
        pix_opt_colors[max_colors > 1] /= max_colors[max_colors > 1]

        # Compute the average color
        optimal_color = np.mean(pix_opt_colors, axis=0)
        self.color = optimal_color
        return self.color

    def get_patch_error(self, targetImage, currentCanvas, optimalColor):         
        prior_error = 0
        error = 0
        
        print(f"opt color {optimalColor}")

        for y in range(self.h):
            for x in range(self.w): 
                opacity = self.heightMap[y, x]
                image_coordinate = self.transform(x, y)
                if 0<=image_coordinate[0]<=self.h and 0<=image_coordinate[1]<=self.w:
                    image_pixel = interpolate_color(image_coordinate, targetImage)
                    current_pixel = interpolate_color(image_coordinate, currentCanvas)
                    new_pixel = opacity * optimalColor + (1 - opacity) * current_pixel #alpha composite single pixel based on color and opacity 
                    pix_error = np.linalg.norm((new_pixel - image_pixel))** 2
                    prior_pix_error = np.linalg.norm((current_pixel - image_pixel)) ** 2
                    if pix_error > 3:
                        print(f"new_pixel {new_pixel}")
                        print(f"new pix err {pix_error}, prior {prior_pix_error}")
                    
                    error += pix_error
                    prior_error += prior_pix_error
        
        return {"newPatchError": error, "oldPatchError": prior_error}
    
    def copy(self):
        print(self.color, self.t)
        new_primitive = BrushStroke2D(self.heightMap, self.canvas_h, self.canvas_w, self.color.copy(), self.theta, self.t.copy())
        return new_primitive
