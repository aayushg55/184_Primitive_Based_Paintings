from primitives import Primitives
from core import *
import numpy as np
import logging
import time
import matplotlib.pyplot as plt

class BrushStroke2D(Primitives):
    def __init__(self, heightMap, canvas_h, canvas_w, probability_discard_pixel, color=np.zeros(3), theta=None, t=None):
        self.isBrushed = True
        # self.heightMap = (heightMap * np.random.uniform(0.5,1)).clip(0,1)
        self.heightMap = heightMap

        self.color = color
        
        # y is row, x is col
        self.canvas_h, self.canvas_w = canvas_h, canvas_w
        self.h = heightMap.shape[0]
        self.w = heightMap.shape[1]
        self.hc = heightMap.shape[0]
        self.wc = heightMap.shape[1]
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
        rad = np.deg2rad(self.theta)
        self.R = np.array([[np.cos(rad), -np.sin(rad)], 
                           [np.sin(rad), np.cos(rad)]])
    
    def randomize_parameters(self): 
        # self.t = np.array([404, 273])
        # self.theta = 45
        self.t = np.array([np.random.randint(0,self.canvas_w-25), np.random.randint(0,self.canvas_h-25)])
        self.theta = np.random.uniform(0, 180)
        self.set_R()

    def mutate(self):
        dice = np.random.randint(0, 3) # 3 before
        if dice==0:
            self.theta += np.random.normal()*15 # 30 before
            self.set_R()
        elif dice==1:
            h = self.h
            w = self.w
            mutation_w = int(np.random.normal()*0.2*w) # 0.5 before
            mutation_h = int(np.random.normal()*0.2*h)
            t_mutation = np.array([mutation_w, mutation_h])
            #t_mutation = np.array([np.random.randint(int(-0.05*self.canvas_w), int(0.05*self.canvas_w)), np.random.randint(int(-0.05*self.canvas_h), int(0.05*self.canvas_h))])
            self.t += t_mutation
            self.t = np.clip(self.t, np.array([0,0]), np.array([self.canvas_w-25, self.canvas_h-25]))
        else:
            h = self.h
            w = self.w
            mutation_w = np.clip(np.random.normal(1, 0.2), 0, 5)
            mutation_h = np.clip(np.random.normal(1, 0.2), 0, 5)
            w_ = int(w*(mutation_w)+1)
            h_ = int(h*(mutation_h)+1)
            if w_>5*self.wc: w_=5*self.wc
            if h_>5*self.hc: h_=5*self.hc
            height_map = self.heightMap
            heightMap_mutate = cv2.resize(height_map, (w_, h_), interpolation=cv2.INTER_AREA)
            self.heightMap=heightMap_mutate
            self.h = heightMap_mutate.shape[0]
            self.w = heightMap_mutate.shape[1]
        #mutate opacity
        random_number = np.random.uniform(0.9, 1.1)
        # while (random_number*np.max(self.heightMap)>1):
        #     random_number = np.random.uniform(0.9, 1.1)
        height_map_alpha = self.heightMap * random_number
        height_map_alpha /= np.max(height_map_alpha)
        self.heightMap = height_map_alpha

        
    def transform(self, x, y): 
        points = np.stack([x, y]).astype(np.float64)
        # logging.info(f"points: {points.shape}")
        
        # Expand dimensions to shape (2, 1) if points is 1D
        if points.ndim == 1:
            points = np.expand_dims(points, axis=1)
        
        self.c[0] = self.w/2
        self.c[1] = self.h/2
        
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
        logging.info(f"num pix: {num_pixels}")
        self.color = optimal_color / num_pixels
        return self.color
    
    def optimal_color_sample(self, targetImage, currentCanvas): 
        optimal_color = np.zeros(3)
        
        num_pixels = 0
        img_h, img_w = targetImage.shape[:2]
        for y in range(self.h):
            for x in range(self.w): 
                opacity = self.heightMap[y, x]
                # logging.info(f"opacity {opacity}")
                if opacity == 0.0:
                    continue
                image_coordinate = self.transform(x, y)
                if 0 <= image_coordinate[0] <= img_w and 0 <= image_coordinate[1] <= img_h:
                    image_pixel = interpolate_color(image_coordinate, targetImage)
                    current_pixel = interpolate_color(image_coordinate, currentCanvas)
                    # logging.info(f"image pixel: {image_pixel}, current pixel: {current_pixel}")
                    pix_opt_color = (image_pixel - (1-opacity)*current_pixel)/opacity
                    max_color = np.max(pix_opt_color)
                    if max_color > 1:
                        pix_opt_color /= max_color
                    # logging.info(f"per pix opt color: {pix_opt_color}")
                    # if np.any(pix_opt_color > 1):
                        # logging.info(f"opacity: {opacity}, image_pixel: {image_pixel}, current_pixel: {current_pixel}")
                    
                    optimal_color += pix_opt_color
                    num_pixels += 1
        logging.info(f"num pix: {num_pixels}")
        self.color = optimal_color / num_pixels
        return self.color

    def optimal_color_and_error_fast(self, targetImage, currentCanvas):
        # Generate a grid of coordinates
        total_time = 0.0
        xs, ys = np.meshgrid(np.arange(self.w, dtype=float), np.arange(self.h, dtype=float))
        opacities = self.heightMap

        # Filter out zero-opacity pixels early
        time_now = time.time()
        # TODO: is p mask giving speedup? remove potentially (66.7s without vs 61.64 with at 1500 prim 5 exp, 10 hc, 10 rs)
        mask = (opacities > 0) & (np.random.rand(*opacities.shape) > self.p) # mask to randomly zero out with probability p
        xs, ys = xs[mask], ys[mask]
        opacities = opacities[mask]
        logging.debug(f"opacity mask creation and masking took {time.time() - time_now:.6f} seconds")
        total_time += time.time() - time_now

        num_removed = self.h*self.w - xs.shape[0]
        logging.info(f"removed {num_removed} zero opacity pixels, remaining {xs.shape[0]} pixels")

        # Transform coordinates and apply boundary check
        time_now = time.time()
        transformed = self.transform(xs, ys)
        valid_mask = (transformed[0, :] < self.canvas_w) & (transformed[0, :] >= 0) & \
                    (transformed[1, :] < self.canvas_h) & (transformed[1, :] >= 0)
        transformed_valid = transformed[:, valid_mask]
        filtered_opacities = opacities.flatten()[valid_mask]
        logging.debug(f"valid masking in color comp took {time.time() - time_now:.6f} seconds")
        total_time += time.time() - time_now

        num_removed = xs.shape[0] - transformed_valid.shape[1]
        logging.info(f"removed {num_removed} out of bounds pixels, remaining {transformed_valid.shape[1]} pixels")
        if transformed_valid.shape[1] < 2:
            logging.critical("VERY FEW VALID PIXELS!!!!")
            return None, None

        # Interpolate colors
        time_now = time.time()
        target_pixels, current_pixels = fast_interp_two(transformed_valid, targetImage, currentCanvas)
        # target_pixels = fast_interp(transformed_valid, targetImage)
        # current_pixels = fast_interp(transformed_valid, currentCanvas)
        logging.debug(f"interpolation in color comp took {time.time() - time_now:.6f} seconds")
        total_time += time.time() - time_now

        # Compute optimal colors
        time_now = time.time()
        pix_opt_colors = (target_pixels - (1 - filtered_opacities[:, np.newaxis]) * current_pixels) / filtered_opacities[:, np.newaxis]
        pix_opt_colors = np.clip(pix_opt_colors, 0, 1)
        optimal_color = np.mean(pix_opt_colors, axis=0)
        self.color = optimal_color
        logging.debug(f"actual optimal color computation took {time.time() - time_now:.6f} seconds")
        total_time += time.time() - time_now

        # Compute color errors
        time_now = time.time()
        # return optimal_color, self.get_patch_error(targetImage, currentCanvas, optimal_color)
        new_pixels = (filtered_opacities[:, np.newaxis]) * optimal_color + (1 - filtered_opacities[:, np.newaxis]) * current_pixels
        error = np.sum((new_pixels - target_pixels) ** 2)
        prior_error = np.sum((current_pixels - target_pixels) ** 2)
        logging.debug(f"actual error computation took {time.time() - time_now:.6f} seconds")
        total_time += time.time() - time_now

        logging.debug(f"total time inside optimal color and error fast: {total_time:.6f} seconds")
        return optimal_color, {"newPatchError": error, "oldPatchError": prior_error}

    def get_patch_error(self, targetImage, currentCanvas, optimalColor):         
        prior_error = 0
        error = 0
        num_pixels = 0
        
        logging.info(f"patch err opt color {optimalColor}")
        logging.info(f"patch err with t as {self.t} and theta as {self.theta}")

        error_heatmap = np.zeros((self.h, self.w))
        transformed_error_heatmap = np.zeros((self.canvas_h, self.canvas_w))
        
        prior_error_heatmap = np.zeros((self.h, self.w))
        prior_transformed_error_heatmap = np.zeros((self.canvas_h, self.canvas_w))

        brush_stroke_mask = np.zeros((self.canvas_h, self.canvas_w))

        for y in range(self.h):
            for x in range(self.w): 
                opacity = self.heightMap[y, x]
                if opacity == 0.0:
                    continue
                image_coordinate = self.transform(x, y)
                # logging.info(f"x, y {x}, {y}; image coord {image_coordinate}")
                if 0<=image_coordinate[0] < self.canvas_w and 0 <= image_coordinate[1] < self.canvas_h:
                    image_pixel = interpolate_color(image_coordinate, targetImage)
                    current_pixel = interpolate_color(image_coordinate, currentCanvas)
                    new_pixel = opacity * optimalColor + (1 - opacity) * current_pixel #alpha composite single pixel based on color and opacity 
                    pix_error = np.linalg.norm((new_pixel - image_pixel))** 2
                    prior_pix_error = np.linalg.norm((current_pixel - image_pixel)) ** 2
                    # logging.info(f"x,y: {x,y} new_pixel {new_pixel}, image_pixel {image_pixel}, current_pixel {current_pixel}")
                    if pix_error > 3:
                        logging.info(f"new_pixel {new_pixel}")
                        logging.info(f"new pix err {pix_error}, prior {prior_pix_error}")
                    # logging.info(f"pix err {pix_error}, prior {prior_pix_error}")
                    # logging.info(f"pix err {pix_error}, prior {prior_pix_error}")
                    error += pix_error
                    prior_error += prior_pix_error
                    num_pixels += 1
                    error_heatmap[y, x] = pix_error
                    brush_stroke_mask[int(image_coordinate[1]), int(image_coordinate[0])] = 1
                    transformed_error_heatmap[int(image_coordinate[1]), int(image_coordinate[0])] = pix_error
                    
                    prior_error_heatmap[y, x] = prior_pix_error
                    prior_transformed_error_heatmap[int(image_coordinate[1]), int(image_coordinate[0])] = prior_pix_error
                    
        # logging.info(f"num pix: {num_pixels}")
        # plt.figure(figsize=(8, 8))
        # plt.imshow(targetImage, cmap='gray')
        # plt.imshow(brush_stroke_mask, cmap='viridis', alpha=0.5)
        # plt.title('Original Canvas with Brush Stroke Mask Overlay')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.show()
        
        # # Visualize the error heatmaps
        # plt.figure(figsize=(12, 6))

        # plt.subplot(2, 2, 1)
        # plt.imshow(error_heatmap, cmap='viridis')
        # plt.colorbar(label='Error')
        # plt.title('Error Heatmap (Original Frame)')
        # plt.xlabel('X')
        # plt.ylabel('Y')

        # plt.subplot(2, 2, 2)
        # plt.imshow(prior_error_heatmap, cmap='viridis')
        # plt.colorbar(label='Error')
        # plt.title('Prior Heatmap')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        
        # plt.subplot(2, 2, 3)
        # plt.imshow(targetImage, cmap='gray')
        # plt.imshow(brush_stroke_mask, cmap='viridis', alpha=0.5)
        # plt.title('Original Canvas with Brush Stroke Mask Overlay')
        # plt.xlabel('X')
        # plt.ylabel('Y')

        # plt.tight_layout()
        # plt.show()
        
        # plt.figure(figsize=(12, 6))
        # plt.subplot(2, 2, 1)
        # plt.imshow(transformed_error_heatmap, cmap='viridis')
        # plt.colorbar(label='Error')
        # plt.title('Transformed Error Heatmap')
        # plt.xlabel('X')
        # plt.ylabel('Y')

        # plt.subplot(2, 2, 2)
        # plt.imshow(prior_transformed_error_heatmap, cmap='viridis')
        # plt.colorbar(label='Error')
        # plt.title('Prior Transformed Error Heatmap')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        
        # plt.subplot(2, 2, 3)
        # plt.imshow(targetImage, cmap='gray')
        # plt.imshow(brush_stroke_mask, cmap='viridis', alpha=0.5)
        # plt.title('Original Canvas with Brush Stroke Mask Overlay')
        # plt.xlabel('X')
        # plt.ylabel('Y')

        # plt.tight_layout()
        # plt.show()
        
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