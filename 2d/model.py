import multiprocessing
from multiprocessing import JoinableQueue
import random

import time
import numpy as np
from worker import Worker
from primitives import Primitives
from state import State
from core import *
import logging

class Model:
    def __init__(self, source_img, output_h, output_w, num_workers, brush_stroke_height_maps, 
                 num_explorations, num_opt_iter, num_random_state_trials):
        #TODO: decide format of inp & target img (np, etc.)
        self.source_img = source_img
        self.source_h = source_img.shape[0]
        self.source_w = source_img.shape[1]
        self.output_h = output_h
        self.output_w = output_w
        self.num_el = output_h * output_w * 3
        self.background_color = np.mean(source_img, axis=(0, 1))
        
        self.current_img = np.zeros_like(source_img)
        self.scores = []
        self.primitives = []
        # self.colors = []
        
        self.brush_stroke_height_maps = brush_stroke_height_maps
        self.num_brush_strokes = len(self.brush_stroke_height_maps)
        
        self.workers = []
        self.num_workers = num_workers


        for i in range(num_workers):
            worker = Worker(
                i, 
                num_explorations // num_workers, # divide work among workers
                num_opt_iter, 
                num_random_state_trials
            )
            self.workers.append(worker)
        
        
        initial_loss = differenceFull(self.current_img, self.source_img)
        self.scores.append(initial_loss)
        logging.info(f"Initial Score: {initial_loss}")
        

    def multiprocess_hill_climb(self, brush_idx):
        brush_height_map = self.brush_stroke_height_maps[brush_idx]
        
        current_state = State(
            height_map=brush_height_map, 
            target=self.source_img,
            current=self.current_img, 
            score=self.scores[-1],
            canvas_score=self.scores[-1],
        )
            
        # Start all the workers
        best_states = []
        for worker in self.workers:
            best_state = worker.run(current_state)
            best_states.append(best_state)
        
        assert len(best_states) == self.num_workers

        # Process the results
        best_energy = np.inf
        best_state = None
        
        logging.debug("checking best states from all workers' hill climb")
        for state in best_states:
            energy = state.energy()
            if energy < best_energy:
                best_energy = energy
                best_state = state
        
        logging.debug(f"best energy from all workers' hill climb: {best_energy}")
        return best_state
            
    def step(self):
        brush_idx = np.random.randint(low=0, high=self.num_brush_strokes)
        best_state = self.multiprocess_hill_climb(brush_idx)
        self.update(best_state)
	        
    def update(self, best_state):
        prev_img = self.current_img.copy()
        strokeAdded = best_state.height_map
        stroke = best_state.primitive
        
        colour = stroke.color
        logging.debug(f"opt color being added {colour}")
        rotation = stroke.theta
        translation = stroke.t
        img = addStroke(strokeAdded, colour, rotation, translation[0], translation[1], prev_img)
        
        self.scores.append(best_state.score)
        logging.debug(f"New score: {best_state.score}")
        self.primitives.append(stroke)
        self.current_img = img
        logging.debug(f"the largest element of the output image is {np.max(img)}")

