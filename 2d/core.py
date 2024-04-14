import math
import numpy as np
import cv2

def differenceFull(current, target):
    h, w = target.shape[0], target.shape[1]

    #Assuming 'current' and 'target' are both NumPy arrays of shape [h,w,3]
    difference=np.abs(current-target)
    loss = np.sum(np.square(difference))
    
    return loss
            
# def differencePartial(current, target, primitive, score, colour):
#     h, w = target.shape[0], target.shape[1]
#     loss=score**2*w*h
#     total=w*h
#     difference = np.array([0, 0, 0])
#     for y in range(primitive.h):
#         for x in range(primitive.w):
#             wx, wy = primitive.transform(x, y)
#             alpha=primitive.height_map[y,x]
#             original_colour=current[wy,wx]
#             blended_colour=(1-alpha)*original_colour+alpha*colour
#             difference = (blended_colour-target[wx,wy]).abs()
#             loss += np.linalg.norm(difference, squared=True)
#     loss= np.sqrt(loss/total)
    
#     return loss
    
def drawShape(current, primitive, colour, height_map):
    h, w=current.shape[0], current.shape[1]
    for y in range(primitive.h):
        for x in range(primitive.w):
            wx, wy = primitive.transform(x, y)
            alpha=height_map[y,x]
            original_colour=current[wy,wx]
            blended_colour=(1-alpha)*original_colour+alpha*colour
            current[wy,wx]=blended_colour
            

def bilinear_interoplation(coordinate, targetImage): 
    h, w = targetImage.shape[:2]
    min_x = int(max(np.floor(coordinate[0]), 0))
    max_x = int(min(np.ceil(coordinate[0]), w-1))
    min_y = int(max(np.floor(coordinate[1]), 0))
    max_y = int(min(np.ceil(coordinate[1]), h-1))
    
    min_x_weight = coordinate[0] - min_x
    max_x_weight = 1 - min_x_weight
    min_y_weight = coordinate[1] - min_y
    max_y_weight = 1 - min_y_weight

    return targetImage[min_y,min_x] * min_y_weight*min_x_weight + targetImage[min_y,max_x] * min_y_weight*max_x_weight + targetImage[max_y,max_x] * max_y_weight*max_x_weight + targetImage[max_y,min_x] * max_y_weight*min_x_weight

def interpolate_color(coordinate, targetImage): 
    return bilinear_interoplation(coordinate, targetImage)


def rotate_image(image, angle, scale=1.0):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    diagonal = int(np.sqrt(w**2 + h**2))
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += (diagonal - w) // 2
    M[1, 2] += (diagonal - h) // 2
    rotated = cv2.warpAffine(image, M, (diagonal, diagonal), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return rotated

def translate_and_pad_image(image, x, y, output_size):
    (h, w) = output_size
    canvas = np.zeros((h, w, 4), dtype=np.float32)
    x_offset = int(max(0, min(w, x)))
    y_offset = int(max(0, min(h, y)))
    end_x = min(x_offset + image.shape[1], w)
    end_y = min(y_offset + image.shape[0], h)
    canvas[y_offset:end_y, x_offset:end_x, :] = image[:end_y-y_offset, :end_x-x_offset, :]
    return canvas

def alpha_composite(base, overlay):
    alpha_overlay = overlay[:,:,3]
    alpha_overlay = np.stack([alpha_overlay]*3, axis=-1)
    composite = overlay[:,:,:3] * alpha_overlay + base[:,:,:3] * (1 - alpha_overlay)
    return composite

def addStroke(height_map, color, rotation, xshift, yshift, base_image):
    """
    overlay_image should be the 
    """
    
    color_overlay = np.zeros((height_map.shape[0], height_map.shape[1], 3), dtype=np.float32)
    color_overlay[:,:,0] = color[0]  # Blue channel
    color_overlay[:,:,1] = color[1]  # Green channel
    color_overlay[:,:,2] = color[2]  # Red channel
    overlay_image = cv2.merge((color_overlay[:, :, 0], color_overlay[:, :, 1], color_overlay[:, :, 2], height_map))

    rotated_overlay = rotate_image(overlay_image, rotation)
    translated_and_padded_overlay = translate_and_pad_image(rotated_overlay, xshift, yshift, base_image.shape[:2])

    composite_image = alpha_composite(base_image, translated_and_padded_overlay)  # Assuming base_image is uint8

    return composite_image  # If needed in uint8 for display or further processing
