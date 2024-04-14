import argparse
import numpy as np
import cv2
from model import Model
import os
import logging

def parse_args():
    parser = argparse.ArgumentParser(description='Instantiate and run the model with command line parameters.')
    parser.add_argument('--source_img_path', type=str, default="input_images/beach.jpg", help='Path to the source image file.')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of worker processes.')
    parser.add_argument('--num_explorations', type=int, default=16, help='Number of exploration steps.')
    parser.add_argument('--num_opt_iter', type=int, default=100, help='Number of optimization iterations.')
    parser.add_argument('--num_random_state_trials', type=int, default=1000, help='Number of random state trials.')
    parser.add_argument('--output_img_path', type=str, default="out/output.jpg", help='Path to save the output image.')
    parser.add_argument('--num_primitives', type=int, default=100, help='Number of primitives to add')
    parser.add_argument('--verbosity', '-v', type=str, default='info', help='Verbosity level (debug, info, warning, error, critical)')

    return parser.parse_args()

def load_brush_jpg(name):
    return cv2.imread(name, cv2.IMREAD_UNCHANGED).astype(np.float32)

def load_brush_stroke_height_maps():
    overlay_image = load_brush_jpg('2d_stroke_heightmaps/stroke_1.jpg')  # Convert to float
    height_map = 1.0 - cv2.cvtColor(overlay_image, cv2.COLOR_BGR2GRAY) / 255.0

    return [height_map]  # Example of 5 random brush strokes

def main():
    args = parse_args()
    
    verbosity_map = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    log_level = verbosity_map.get(args.verbosity.lower(), logging.INFO)
    print(f"Setting log level to {args.verbosity.lower()}")
    logging.basicConfig(level=log_level, format='%(message)s')

    # Load source image
    source_img = cv2.imread(args.source_img_path, cv2.IMREAD_COLOR)
    if source_img is None:
        raise FileNotFoundError(f"Source image at {args.source_img_path} not found.")
    
    # Convert to appropriate format if necessary
    source_img = source_img.astype(np.float32) / 255.0  # Normalize if using float operations in model
    
    # TODO: resize source img if too large
    output_h, output_w = source_img.shape[:2]
    
    # Load brush stroke height maps
    brush_stroke_height_maps = load_brush_stroke_height_maps()

    # Instantiate the model
    model = Model(
        source_img=source_img,
        output_h=output_h,
        output_w=output_w,
        num_workers=args.num_workers,
        brush_stroke_height_maps=brush_stroke_height_maps,
        num_explorations=args.num_explorations,
        num_opt_iter=args.num_opt_iter,
        num_random_state_trials=args.num_random_state_trials
    )
    
    # Example step or processing (add your actual method calls)
    for i in range(args.num_primitives): 
        model.step()
        logging.info(f"finished step {i}")
        logging.info("************************************")
    

    # Save the resulting image
    output_img = (model.current_img * 255).astype(np.uint8)  # Convert back to uint8
    cv2.imwrite(args.output_img_path, output_img)
    logging.info(f"Output image saved to {args.output_img_path}")

if __name__ == '__main__':
    main()
