import argparse
import numpy as np
import cv2
from model import Model
import os
import logging
import time
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Instantiate and run the model with command line parameters.')
    parser.add_argument('--source_img_path', type=str, default="input_images/beach.jpg", help='Path to the source image file.')
    parser.add_argument('--brush_img_path', type=str, default="2d_stroke_heightmaps/brush_stroke_2.jpg", help='Path to the brush stroke image file.')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of worker processes.')
    parser.add_argument('--num_explorations', type=int, default=10, help='Number of exploration steps.')
    parser.add_argument('--num_opt_iter', type=int, default=100, help='Number of optimization iterations.')
    parser.add_argument('--num_random_state_trials', type=int, default=10, help='Number of random state trials.')
    parser.add_argument('--output_img_path', '-o', type=str, default="out/output.jpg", help='Path to save the output image.')
    parser.add_argument('--num_primitives', type=int, default=100, help='Number of primitives to add')
    parser.add_argument('--sample_probability', '-p', type=float, default=.9, help='probability')
    parser.add_argument('--redirect_stdout', '-r', type=int, default=0, help='Redirect stdout to a file')
    parser.add_argument('--verbosity', '-v', type=str, default='error', help='Verbosity level (debug, info, warning, error, critical)')

    return parser.parse_args()

def load_brush_jpg(name):
    brush = cv2.imread(name, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return (1.0 - cv2.cvtColor(brush, cv2.COLOR_BGR2GRAY) / 255.0)

def load_brush_stroke_height_maps(path):
    overlay_image = load_brush_jpg(path)  # Convert to float
    height_map = 0.8 * overlay_image
    # height_map2 = cv2.resize(height_map, (3*height_map.shape[0]//4, 3*height_map.shape[1]//4), interpolation=cv2.INTER_AREA)
    # height_map13 = cv2.resize(height_map, (height_map.shape[0]//4, 3*height_map.shape[1]//4), interpolation=cv2.INTER_AREA)
    # height_map31 = cv2.resize(height_map, (height_map.shape[0]//4, 3*height_map.shape[1]//4), interpolation=cv2.INTER_AREA)

    # height_map3 = cv2.resize(height_map, (height_map.shape[0]//2, height_map.shape[1]//2), interpolation=cv2.INTER_AREA)
    # height_map4 = cv2.resize(height_map, (height_map.shape[0]//4, height_map.shape[1]//4), interpolation=cv2.INTER_AREA)
    # height_map5 = cv2.resize(height_map, (height_map.shape[0]//10, height_map.shape[1]//10), interpolation=cv2.INTER_AREA)
    # height_map6 = cv2.resize(height_map, (height_map.shape[0]//8, height_map.shape[1]//4), interpolation=cv2.INTER_AREA)
    
    # height_map_big = cv2.resize(height_map, (height_map.shape[0]*2, height_map.shape[1]*2), interpolation=cv2.INTER_AREA)
    # height_map_long = cv2.resize(height_map, (int(height_map.shape[0]*0.1), int(height_map.shape[1]*2)), interpolation=cv2.INTER_AREA)
    # height_map_wide = cv2.resize(height_map, (int(height_map.shape[0]*2), int(height_map.shape[1]*0.1)), interpolation=cv2.INTER_AREA)

    # height_map8 = cv2.resize(height_map, (height_map.shape[0]*3, height_map.shape[1]*3), interpolation=cv2.INTER_AREA)
    # height_map9 = cv2.resize(height_map, (height_map.shape[0]//4, height_map.shape[1]), interpolation=cv2.INTER_AREA)

    # # return [height_map, height_map2]  # Example of 5 random brush strokes
    # return [height_map_big, height_map, height_map2, height_map13, height_map31, height_map3, height_map6]  # Example of 5 random brush strokes
    # # return [height_map]
    brush_strokes = []
    # Upsized by 2
    height_map_up_2 = cv2.resize(height_map, (height_map.shape[1]*2, height_map.shape[0]*2), interpolation=cv2.INTER_AREA)
    brush_strokes.append(height_map_up_2)
    
    # # Thin: downsized by 5 in width and up 2 in height
    # height_map_thin = cv2.resize(height_map, (height_map.shape[1]//5, height_map.shape[0]*2), interpolation=cv2.INTER_AREA)
    # brush_strokes.append(height_map_thin)
    
    # # Wide: upsized by 2 in width and down 5 in height
    # height_map_wide = cv2.resize(height_map, (height_map.shape[1]*2, height_map.shape[0]//5), interpolation=cv2.INTER_AREA)
    # brush_strokes.append(height_map_wide)
    
    # Original size
    brush_strokes.append(height_map)
    
    # Downsized by 2
    height_map_2 = cv2.resize(height_map, (height_map.shape[1]//2, height_map.shape[0]//2), interpolation=cv2.INTER_AREA)
    brush_strokes.append(height_map_2)
    
    # Downsized by 4
    height_map_4 = cv2.resize(height_map, (height_map.shape[1]//4, height_map.shape[0]//4), interpolation=cv2.INTER_AREA)
    brush_strokes.append(height_map_4)
    
    # Uneven scaling: downsized by 2 in width and 4 in height
    height_map_2_4 = cv2.resize(height_map, (height_map.shape[1]//2, height_map.shape[0]//4), interpolation=cv2.INTER_AREA)
    brush_strokes.append(height_map_2_4)
    
    # Uneven scaling: downsized by 4 in width and 2 in height
    height_map_4_2 = cv2.resize(height_map, (height_map.shape[1]//4, height_map.shape[0]//2), interpolation=cv2.INTER_AREA)
    brush_strokes.append(height_map_4_2)
    
    height_map_8_4 = cv2.resize(height_map, (height_map.shape[0]//8, height_map.shape[1]//4), interpolation=cv2.INTER_AREA)
    brush_strokes.append(height_map_8_4)

    height_map_10_10 = cv2.resize(height_map, (height_map.shape[0]//10, height_map.shape[1]//10), interpolation=cv2.INTER_AREA)
    brush_strokes.append(height_map_10_10)
   
    height_map_15_15 = cv2.resize(height_map, (height_map.shape[0]//12, height_map.shape[1]//4), interpolation=cv2.INTER_AREA)
    brush_strokes.append(height_map_15_15)
   
    return brush_strokes

def write_primitive_details(model, file_path):
    with open(file_path, 'w') as file:
        for i, primitive in enumerate(model.primitives):
            t_str = ','.join(map(str, primitive.t))
            theta_str = str(primitive.theta)
            color_str = ','.join(map(str, primitive.color))
            file.write(f"Primitive {i}: brush_idx={model.brush_strokes[i]}, t=[{t_str}], theta={theta_str}, color=[{color_str}]\n")
            logging.warning(f"Primitive {i}: brush_idx={model.brush_strokes[i]}, t=[{t_str}], theta={theta_str}, color=[{color_str}]")


def main():
    args = parse_args()
    
    verbosity_map = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    log_level = verbosity_map.get(args.verbosity.lower(), logging.ERROR)
    print(f"Setting log level to {args.verbosity.lower()}")
    print(f"Redirecting stdout to file: {args.redirect_stdout}")
    
    inp_img_name = args.source_img_path.split('/')[-1].split('.')[0]
    brush_img_name = args.brush_img_path.split('/')[-1].split('.')[0]
    inp_img_name = inp_img_name + '_' + brush_img_name
    
    now = datetime.now()
    date_time_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    log_file = 'log_' + inp_img_name + f"_n_expl_{args.num_explorations}_n_o_iter_{args.num_opt_iter}_n_rs_{args.num_random_state_trials}_n_prim_{args.num_primitives}_disc_p_{args.sample_probability}_n_work_{args.num_workers}_{date_time_string}.txt"

    if args.redirect_stdout == 1:
        logging.basicConfig(filename=log_file, level=log_level, format='%(message)s')
    else:
        logging.basicConfig(level=log_level, format='%(message)s')

    # Load source image
    source_img = cv2.imread(args.source_img_path, cv2.IMREAD_COLOR)
    #base_img = cv2.imread('input_images/britany_base2.jpg', cv2.IMREAD_COLOR)
    if source_img is None:
        raise FileNotFoundError(f"Source image at {args.source_img_path} not found.")
    
    # Convert to appropriate format if necessary
    source_img = source_img.astype(np.float32) / 255.0 
    #base_img = base_img.astype(np.float32) / 255.0 
    # TODO: resize source img if too large
    output_h, output_w = source_img.shape[:2]
    
    # Load brush stroke height maps
    brush_stroke_height_maps = load_brush_stroke_height_maps(args.brush_img_path)

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
        num_steps=args.num_primitives,
        #base_img=base_img
    )
    
    out_name = 'out/' + inp_img_name + f"_n_expl_{args.num_explorations}_n_o_iter_{args.num_opt_iter}_n_rs_{args.num_random_state_trials}_n_prim_{args.num_primitives}_disc_p_{args.sample_probability}_n_work_{args.num_workers}"
    os.makedirs(out_name, exist_ok=True)
    out_dir_name = out_name
    logging.error(f"Output will be saved to {out_name}")
    start_time = time.time()

    for i in range(args.num_primitives):
        cur_time = time.time()
        model.step()
        
        logging.error(f"finished step {i}")
        logging.error("*************************************************")
        logging.error(f"step took {time.time() - cur_time:.6f} seconds")
        
        mod_factor = args.num_primitives // 100
        if mod_factor == 0: mod_factor = 1
        if i % mod_factor == 0:
            iter_name = os.path.join(out_name, f"iter_{i}.jpg")
            logging.error(f"Saving image at step {i}")
            output_img = (model.current_img * 255).astype(np.uint8)
            cv2.imwrite(iter_name, output_img)

            iter_name = os.path.join(out_name, f"iter_height{i}.jpg")
            output_height_map = (model.current_height_map / np.max(model.current_height_map) * 255).astype(np.uint8)
            cv2.imwrite(iter_name, output_height_map)

    duration = time.time() - start_time

    logging.critical(f"******************************** Stats ************************************")
    logging.critical(f"Total time: {duration:.6f} seconds")
    logging.critical(f"Avg time per step: {(duration / args.num_primitives):.6f} seconds")
    logging.critical(f"Number of energy computations: {((args.num_primitives*args.num_explorations* (args.num_opt_iter + args.num_random_state_trials)))}")
    logging.critical(f"****************************************************************************")

    # Save the resulting image
    output_img = (model.current_img * 255).astype(np.uint8)  # Convert back to uint8
    
    final_out_name = out_dir_name + '/' + inp_img_name + f"_n_expl_{args.num_explorations}_n_o_iter_{args.num_opt_iter}_n_rs_{args.num_random_state_trials}_n_prim_{args.num_primitives}_disc_p_{args.sample_probability}_n_work_{args.num_workers}.jpg"
    cv2.imwrite(final_out_name, output_img)
    logging.critical(f"Output image saved to {out_name}")

    # Save the resulting heightmap
    output_height_img = (model.current_height_map / np.max(model.current_height_map) * 255).astype(np.uint8)  # Convert back to uint8
    out_height_name = out_dir_name + '/height_' + inp_img_name + f"_n_expl_{args.num_explorations}_n_o_iter_{args.num_opt_iter}_n_rs_{args.num_random_state_trials}_n_prim_{args.num_primitives}_disc_p_{args.sample_probability}_n_work_{args.num_workers}"

    cv2.imwrite(out_height_name+'.jpg', output_height_img)
    np.save(out_height_name+'.npy', model.current_height_map)
    logging.critical(f"Output heightmap image saved to {out_height_name+'.jpg'}")
    
    file_path = out_name.split('.')[0] + '_primitives.txt'
    write_primitive_details(model, file_path)

if __name__ == '__main__':
    main()
