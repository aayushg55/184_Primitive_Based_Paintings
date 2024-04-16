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

def fast_interp(coordinates, image):
    return fast_interpolate_colors_nn(coordinates, image)

def fast_interpolate_colors_bilinear(coordinates, image):
    x = coordinates[0]
    y = coordinates[1]
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, image.shape[1] - 1)
    x1 = np.clip(x1, 0, image.shape[1] - 1)
    y0 = np.clip(y0, 0, image.shape[0] - 1)
    y1 = np.clip(y1, 0, image.shape[0] - 1)

    Ia = image[y0, x0]
    Ib = image[y1, x0]
    Ic = image[y0, x1]
    Id = image[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa[:, np.newaxis] * Ia + wb[:, np.newaxis] * Ib + wc[:, np.newaxis] * Ic + wd[:, np.newaxis] * Id

def fast_interpolate_colors_nn(coordinates, image):
    x = coordinates[0]
    y = coordinates[1]
    
    # Round coordinates to nearest integer
    x_round = np.round(x).astype(int)
    y_round = np.round(y).astype(int)
    
    # Clip coordinates to image bounds
    x_clipped = np.clip(x_round, 0, image.shape[1] - 1)
    y_clipped = np.clip(y_round, 0, image.shape[0] - 1)
    
    # Use integer array indexing to get nearest-neighbor pixel values
    return image[y_clipped, x_clipped]            

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
    #M[0, 2] += (diagonal - w) // 2
    M[1, 2] += (diagonal - h) // 2
    rotated = cv2.warpAffine(image, M, (diagonal, diagonal), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return rotated

def translate_and_pad_image(image, x, y, prim_shape, output_size):
    (h, w) = output_size
    prim_center = (prim_shape[1] // 2, prim_shape[0] // 2)
    canvas = np.zeros((h, w, image.shape[2]), dtype=image.dtype)  # Ensure matching number of channels and data type

    # Calculate the centered translation offsets
    x_offset = -prim_center[0] + x
    y_offset = -prim_center[1] + y

    # Ensure the coordinates are within the bounds of the canvas
    x_start = max(0, x_offset)
    y_start = max(0, y_offset)
    x_end = min(w, x_offset + image.shape[1])
    y_end = min(h, y_offset + image.shape[0])

    # Calculate the parts of the image to be copied based on the offsets
    image_x_start = max(0, -x_offset)  # Start copying from here if offset is negative
    image_y_start = max(0, -y_offset)
    image_x_end = x_end - x_offset  # End copying here if offset + image width exceeds canvas width
    image_y_end = y_end - y_offset

    # Copy the image to the canvas
    canvas[y_start:y_end, x_start:x_end] = image[image_y_start:image_y_end, image_x_start:image_x_end]
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
    translated_and_padded_overlay = translate_and_pad_image(rotated_overlay, xshift, yshift, rotated_overlay.shape, base_image.shape[:2])

    composite_image = alpha_composite(base_image, translated_and_padded_overlay)  # Assuming base_image is uint8

    return composite_image  # If needed in uint8 for display or further processing


###############

# def rotate_image(image, angle, scale=1.0):
#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)
#     diagonal = int(np.sqrt(w**2 + h**2))
#     M = cv2.getRotationMatrix2D(center, angle, scale)
#     M[0, 2] += (diagonal - w) // 2
#     M[1, 2] += (diagonal - h) // 2
#     rotated = cv2.warpAffine(image, M, (diagonal, diagonal), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
#     return rotated

# def translate_and_pad_image(image, x, y, prim_shape, output_size):
#     (h, w) = output_size
#     prim_center = (prim_shape[1] // 2, prim_shape[0] // 2)
#     canvas = np.zeros((h, w, image.shape[2]), dtype=image.dtype)  # Ensure matching number of channels and data type

#     # Calculate the centered translation offsets
#     x_offset = -prim_center[0] + x
#     y_offset = -prim_center[1] + y

#     # Ensure the coordinates are within the bounds of the canvas
#     x_start = max(0, x_offset)
#     y_start = max(0, y_offset)
#     x_end = min(w, x_offset + image.shape[1])
#     y_end = min(h, y_offset + image.shape[0])

#     # Calculate the parts of the image to be copied based on the offsets
#     image_x_start = max(0, -x_offset)  # Start copying from here if offset is negative
#     image_y_start = max(0, -y_offset)
#     image_x_end = x_end - x_offset  # End copying here if offset + image width exceeds canvas width
#     image_y_end = y_end - y_offset

#     # Copy the image to the canvas
#     canvas[y_start:y_end, x_start:x_end] = image[image_y_start:image_y_end, image_x_start:image_x_end]
#     return canvas

# def alpha_composite(base, overlay):
#     alpha_overlay = overlay[:,:,3]
#     alpha_overlay = np.stack([alpha_overlay]*3, axis=-1)
#     composite = overlay[:,:,:3] * alpha_overlay + base[:,:,:3] * (1 - alpha_overlay)
#     return composite

# def addStroke(overlay_image, color, rotation, xshift, yshift, base_image):
#     color_overlay = np.zeros((overlay_image.shape[0], overlay_image.shape[1], 3), dtype=np.float32)
#     color_overlay[:,:,0] = color[0] # Blue channel
#     color_overlay[:,:,1] = color[1] # Green channel
#     color_overlay[:,:,2] = color[2] # Red channel
#     overlay_image = cv2.merge((color_overlay[:, :, 0], color_overlay[:, :, 1], color_overlay[:, :, 2], color_overlay))

#     rotated_overlay = rotate_image(overlay_image, rotation)
#     translated_and_padded_overlay = translate_and_pad_image(rotated_overlay, xshift, yshift, rotated_overlay.shape, base_image.shape[:2])
#     print(overlay_image.shape)

#     composite_image = alpha_composite(base_image, translated_and_padded_overlay)  # Assuming base_image is uint8

#     return composite_image  # If needed in uint8 for display or further processing
