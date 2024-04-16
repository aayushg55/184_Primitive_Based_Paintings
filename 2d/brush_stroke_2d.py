from primitives import Primitives
from core import *
import numpy as np
import logging
import time

class BrushStroke2D(Primitives):
    def __init__(self, heightMap, canvas_h, canvas_w, probability_discard_pixel, color=np.zeros(3), theta=None, t=None):
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

        self.p = probability_discard_pixel
        
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
        self.t = np.array([np.random.randint(0,self.canvas_w-25), np.random.randint(0,self.canvas_h-25)])
        self.theta = np.random.uniform(0, 180)
        self.set_R()

    def mutate(self):
        dice = np.random.randint(0, 3)
        if dice==0:
            self.theta += np.random.normal()*30
            self.set_R()
        elif dice==1:
            h = self.h
            w = self.w
            mutation_w = int(np.random.normal()*0.5*w)
            mutation_h = int(np.random.normal()*0.5*h)
            t_mutation = np.array([mutation_w, mutation_h])
            #t_mutation = np.array([np.random.randint(int(-0.05*self.canvas_w), int(0.05*self.canvas_w)), np.random.randint(int(-0.05*self.canvas_h), int(0.05*self.canvas_h))])
            self.t += t_mutation
            self.t = np.clip(self.t, np.array([0,0]), np.array([self.canvas_w-25, self.canvas_h-25]))
        else:
            h = self.h
            w = self.w
            
            mutation = np.random.normal()*0.5
            while mutation <= -0.9:
                mutation = np.random.normal()*0.5
            height_map = self.heightMap
            heightMap_mutate = cv2.resize(height_map, (int(w*(1+mutation)+1), int(h*(1+mutation)+1)), interpolation=cv2.INTER_AREA)
            self.heightMap=heightMap_mutate
            self.h = heightMap_mutate.shape[0]
            self.w = heightMap_mutate.shape[1]

        
    def transform(self, x, y): 
        points = np.stack([x, y]).astype(np.float64)
        # logging.debug(f"points: {points.shape}")
        
        # Expand dimensions to shape (2, 1) if points is 1D
        if points.ndim == 1:
            points = np.expand_dims(points, axis=1)
        
        points -= self.c  
        
        transformed = self.R @ points
        # transformed += self.c
        
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
                if opacity == 0.0:
                    continue
                image_coordinate = self.transform(x, y)
                if 0 <= image_coordinate[0] <= img_w and 0 <= image_coordinate[1] <= img_h:
                    image_pixel = interpolate_color(image_coordinate, targetImage)
                    current_pixel = interpolate_color(image_coordinate, currentCanvas)
                    pix_opt_color = (image_pixel - (1-opacity)*current_pixel)/opacity
                    max_color = np.max(pix_opt_color)
                    if max_color > 1:
                        pix_opt_color /= max_color

                    optimal_color += pix_opt_color
                    num_pixels += 1
        logging.debug(f"num pix: {num_pixels}")
        self.color = optimal_color / num_pixels
        return self.color
    
    def optimal_color_sample(self, targetImage, currentCanvas): 
        optimal_color = np.zeros(3)
        
        num_pixels = 0
        img_h, img_w = targetImage.shape[:2]
        for y in range(self.h):
            for x in range(self.w): 
                opacity = self.heightMap[y, x]
                # logging.debug(f"opacity {opacity}")
                if opacity == 0.0:
                    continue
                image_coordinate = self.transform(x, y)
                if 0 <= image_coordinate[0] <= img_w and 0 <= image_coordinate[1] <= img_h:
                    image_pixel = interpolate_color(image_coordinate, targetImage)
                    current_pixel = interpolate_color(image_coordinate, currentCanvas)
                    # logging.debug(f"image pixel: {image_pixel}, current pixel: {current_pixel}")
                    pix_opt_color = (image_pixel - (1-opacity)*current_pixel)/opacity
                    max_color = np.max(pix_opt_color)
                    if max_color > 1:
                        pix_opt_color /= max_color
                    # logging.debug(f"per pix opt color: {pix_opt_color}")
                    # if np.any(pix_opt_color > 1):
                        # logging.debug(f"opacity: {opacity}, image_pixel: {image_pixel}, current_pixel: {current_pixel}")
                    
                    optimal_color += pix_opt_color
                    num_pixels += 1
        logging.debug(f"num pix: {num_pixels}")
        self.color = optimal_color / num_pixels
        return self.color

    
    def optimal_color_fast(self, targetImage, currentCanvas):
        # Generate a grid of coordinates
        time_now = time.time() 
        xs, ys = np.meshgrid(np.arange(self.w, dtype=float), np.arange(self.h, dtype=float))
        opacities = self.heightMap
        logging.info(f"meshgrid in color comp took {time.time() - time_now:.4f} seconds")

        # Filter out zero-opacity pixels early
        time_now = time.time() 
        ##operation to zero out randomly
        base_mask = opacities > 0  # basic mask for non-zero opacities
        random_mask = np.random.rand(*opacities.shape) > self.p  # mask to randomly zero out with probability p
        mask = base_mask & random_mask  # combine masks: pixels must pass both conditions

        #
        xs, ys = xs[mask], ys[mask]
        opacities = opacities[mask]
        logging.info(f"opacity masking in color comp took {time.time() - time_now:.4f} seconds")

        num_removed = self.h*self.w - xs.shape[0]
        logging.debug(f"removed {num_removed} zero opacity pixels, remaining {xs.shape[0]} pixels, mean opacity {np.mean(opacities)}")
        # if num_removed > 0.75 * self.h * self.w:
        #     logging.warning("REMOVED MANY zero-opacity pixels!!")

        time_now = time.time() 
        # Transform coordinates and apply boundary check
        transformed = self.transform(xs, ys)
        valid_mask = (transformed[0, :] < self.canvas_w) & (transformed[0, :] >= 0) & \
                    (transformed[1, :] < self.canvas_h) & (transformed[1, :] >= 0)
        
        transformed_valid = transformed[:, valid_mask]
        filtered_opacities = opacities.flatten()[valid_mask]
        logging.info(f"valid masking in color comp took {time.time() - time_now:.4f} seconds")
        
        num_removed = xs.shape[0] - transformed_valid.shape[1]
        logging.debug(f"removed {num_removed} out of bounds pixels, remaining {transformed_valid.shape[1]} pixels")
        if transformed_valid.shape[1] < 2: #0.01 * self.h * self.w:
            logging.warning("VERY FEW VALID PIXELS!!!!")
            return None
        
        # Interpolate colors
        time_now = time.time()
        target_pixels = fast_interp(transformed_valid, targetImage)
        current_pixels = fast_interp(transformed_valid, currentCanvas)
        logging.info(f"interpolation in color comp took {time.time() - time_now:.4f} seconds")
        
        # Compute optimal colors
        pix_opt_colors = (target_pixels - (1 - filtered_opacities[:, np.newaxis]) * current_pixels) / filtered_opacities[:, np.newaxis]
        pix_opt_colors = np.clip(pix_opt_colors, 0, 1)
        
        # Clip and normalize if necessary
        # max_colors = np.max(pix_opt_colors, axis=1, keepdims=True)
        # need_to_scale = (max_colors > 1).squeeze()
        # pix_opt_colors[need_to_scale] /= max_colors[need_to_scale]

        # Compute the average color
        optimal_color = np.mean(pix_opt_colors, axis=0)
        self.color = optimal_color
        return self.color

    def get_patch_error_fast(self, targetImage, currentCanvas, optimalColor):         
        # Generate a grid of coordinates
        xs, ys = np.meshgrid(np.arange(self.w, dtype=float), np.arange(self.h, dtype=float))
        opacities = self.heightMap

        # Filter out zero-opacity pixels early
        mask = opacities > 0
        ##operation to zero out randomly
        base_mask = opacities > 0  # basic mask for non-zero opacities
        random_mask = np.random.rand(*opacities.shape) > self.p  # mask to randomly zero out with probability p
        mask = base_mask & random_mask  # combine masks: pixels must pass both conditions

        xs, ys = xs[mask], ys[mask]
        opacities = opacities[mask]

        num_removed = self.h*self.w - xs.shape[0]
        logging.debug(f"removed {num_removed} zero opacity pixels, remaining {xs.shape[0]} pixels, mean opacity {np.mean(opacities)}")
        # if num_removed > 0.75 * self.h * self.w:
        #     logging.warning("REMOVED MANY zero-opacity pixels!!")

        time_now = time.time() 
        # Transform coordinates and apply boundary check
        transformed = self.transform(xs, ys)
        valid_mask = (transformed[0, :] < self.canvas_w) & (transformed[0, :] >= 0) & \
                    (transformed[1, :] < self.canvas_h) & (transformed[1, :] >= 0)
        
        transformed_valid = transformed[:, valid_mask]
        filtered_opacities = opacities.flatten()[valid_mask]
        logging.info(f"valid masking in color comp took {time.time() - time_now:.4f} seconds")

        
        num_removed = xs.shape[0] - transformed_valid.shape[1]
        logging.debug(f"removed {num_removed} out of bounds pixels, remaining {transformed_valid.shape[1]} pixels")
        if transformed_valid.shape[1] < 2: #0.01 * self.h * self.w:
            logging.warning("VERY FEW VALID PIXELS!!!!")
            return None
        
        # Interpolate colors
        time_now = time.time()
        target_pixels = fast_interp(transformed_valid, targetImage)
        current_pixels = fast_interp(transformed_valid, currentCanvas)
        logging.info(f"interpolation in color comp took {time.time() - time_now:.4f} seconds")
        
        # Compute color errors
        new_pixels = (filtered_opacities[:, np.newaxis]) * optimalColor + (1 - filtered_opacities[:, np.newaxis]) * current_pixels
        
        error = np.linalg.norm((new_pixels - target_pixels)) ** 2
        prior_error = np.linalg.norm((current_pixels - target_pixels)) ** 2
        
        return {"newPatchError": error, "oldPatchError": prior_error}

    def get_patch_error(self, targetImage, currentCanvas, optimalColor):         
        prior_error = 0
        error = 0
        num_pixels = 0
        
        logging.debug(f"patch err opt color {optimalColor}")
        logging.debug(f"patch err with t as {self.t} and theta as {self.theta}")

        for y in range(self.h):
            for x in range(self.w): 
                opacity = self.heightMap[y, x]
                image_coordinate = self.transform(x, y)
                # logging.debug(f"x, y {x}, {y}; image coord {image_coordinate}")
                if 0<=image_coordinate[0] < self.canvas_w and 0 <= image_coordinate[1] < self.canvas_h:
                    image_pixel = interpolate_color(image_coordinate, targetImage)
                    current_pixel = interpolate_color(image_coordinate, currentCanvas)
                    new_pixel = opacity * optimalColor + (1 - opacity) * current_pixel #alpha composite single pixel based on color and opacity 
                    pix_error = np.linalg.norm((new_pixel - image_pixel))** 2
                    prior_pix_error = np.linalg.norm((current_pixel - image_pixel)) ** 2
                    if pix_error > 3:
                        logging.debug(f"new_pixel {new_pixel}")
                        logging.debug(f"new pix err {pix_error}, prior {prior_pix_error}")
                    
                    # logging.debug(f"pix err {pix_error}, prior {prior_pix_error}")
                    error += pix_error
                    prior_error += prior_pix_error
                    num_pixels += 1
        logging.debug(f"num pix: {num_pixels}")
        return {"newPatchError": error, "oldPatchError": prior_error}
    
    def copy(self):
        new_primitive = BrushStroke2D(
            self.heightMap, 
            self.canvas_h, self.canvas_w, 
            self.p,
            self.color.copy(), self.theta, 
            self.t.copy()
        )
        return new_primitive
            
        return new_primitive

    """
    alternative for interpolation, need to compare speed
    def create_interp_grid(img):
        X, Y = np.mgrid[:img.shape[0], :img.shape[1]]
        points = np.dstack((X.ravel(), Y.ravel())).squeeze()
        # Flatten img to use for interpolation
        flat_img = einops.rearrange(img, ('h w c -> (h w) c'))
        color_interp = scipy.interpolate.NearestNDInterpolator(points, flat_img)
        return color_interp
    
    pixel_colors = color_interp((r_n, c_n))
    """