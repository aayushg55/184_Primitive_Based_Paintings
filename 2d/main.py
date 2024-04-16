import argparse
import numpy as np
import cv2
from model import Model
import os
import logging
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Instantiate and run the model with command line parameters.')
    parser.add_argument('--source_img_path', type=str, default="input_images/beach.jpg", help='Path to the source image file.')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of worker processes.')
    parser.add_argument('--num_explorations', type=int, default=16, help='Number of exploration steps.')
    parser.add_argument('--num_opt_iter', type=int, default=100, help='Number of optimization iterations.')
    parser.add_argument('--num_random_state_trials', type=int, default=1000, help='Number of random state trials.')
    parser.add_argument('--output_img_path', '-o', type=str, default="out/output.jpg", help='Path to save the output image.')
    parser.add_argument('--num_primitives', type=int, default=100, help='Number of primitives to add')
    parser.add_argument('--sample_probability', '-p', type=float, default=.5, help='probability')

    parser.add_argument('--verbosity', '-v', type=str, default='warning', help='Verbosity level (debug, info, warning, error, critical)')

    return parser.parse_args()

def load_brush_jpg(name):
    return cv2.imread(name, cv2.IMREAD_UNCHANGED).astype(np.float32)

def load_brush_stroke_height_maps():
    overlay_image = load_brush_jpg('2d_stroke_heightmaps/stroke_1.jpg')  # Convert to float
    height_map = 1.0 - cv2.cvtColor(overlay_image, cv2.COLOR_BGR2GRAY) / 255.0
    height_map2 = cv2.resize(height_map, (3*height_map.shape[0]//4, 3*height_map.shape[1]//4), interpolation=cv2.INTER_AREA)
    height_map3 = cv2.resize(height_map, (height_map.shape[0]//2, height_map.shape[1]//2), interpolation=cv2.INTER_AREA)
    height_map4 = cv2.resize(height_map, (height_map.shape[0]//4, height_map.shape[1]//4), interpolation=cv2.INTER_AREA)
    height_map5 = cv2.resize(height_map, (height_map.shape[0]//8, height_map.shape[1]//4), interpolation=cv2.INTER_AREA)
    height_map6 = cv2.resize(height_map, (height_map.shape[0]//16, height_map.shape[1]//4), interpolation=cv2.INTER_AREA)

    # return [height_map, height_map2]  # Example of 5 random brush strokes
    return [height_map, height_map2, height_map3, height_map4, height_map5, height_map6]  # Example of 5 random brush strokes

def write_primitive_details(model, file_path):
    with open(file_path, 'w') as file:
        for i, primitive in enumerate(model.primitives):
            t_str = ','.join(map(str, primitive.t))
            theta_str = str(primitive.theta)
            color_str = ','.join(map(str, primitive.color))
            file.write(f"Primitive {i}: brush_idx={model.brush_strokes[i]}, t=[{t_str}], theta={theta_str}, color=[{color_str}]\n")
            logging.info(f"Primitive {i}: brush_idx={model.brush_strokes[i]}, t=[{t_str}], theta={theta_str}, color=[{color_str}]")


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
        num_random_state_trials=args.num_random_state_trials,
        discard_probability=args.sample_probability,
        num_steps=args.num_primitives
    )
    
    # Example step or processing (add your actual method calls)
    start_time = time.time()
    for i in range(args.num_primitives):
        cur_time = time.time()
        model.step()
        logging.warning(f"finished step {i}")
        logging.warning("************************************")
        logging.warning(f"step took {time.time() - cur_time:.6f} seconds")
    duration = time.time() - start_time

    logging.error(f"******************************** Stats ************************************")
    logging.error(f"Total time: {duration:.6f} seconds")
    logging.error(f"Avg time per step: {(duration / args.num_primitives):.6f} seconds")
    logging.error(f"Number of energy computations: {((args.num_primitives*args.num_explorations* (args.num_opt_iter + args.num_random_state_trials)))}")
    logging.error(f"****************************************************************************")

    # Save the resulting image
    output_img = (model.current_img * 255).astype(np.uint8)  # Convert back to uint8
    inp_img_name = args.source_img_path.split('/')[-1].split('.')[0]
    out_name = 'out/' + inp_img_name + f"_n_expl_{args.num_explorations}_n_o_iter_{args.num_opt_iter}_n_rs_{args.num_random_state_trials}_n_prim_{args.num_primitives}_disc_p_{args.sample_probability}_n_work_{args.num_workers}.jpg"
    cv2.imwrite(out_name, output_img)
    logging.error(f"Output image saved to {out_name}")
    
    file_path = out_name.split('.')[0] + '_primitives.txt'
    write_primitive_details(model, file_path)

if __name__ == '__main__':
    main()
