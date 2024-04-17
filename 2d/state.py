import numpy as np
from brush_stroke_2d import BrushStroke2D
from core import *
import logging
import time

class State:
    def __init__(self, height_map, target, current, pixel_discard_probability, score=np.inf, canvas_score=np.inf, primitive=None, recalculate_score=False):
        self.target = target
        self.current = current
        
        self.score = score  # score for current canvas with current primitive added
        
        # Score for current canvas without current primitive added. 
        # Set once, stays constant throughout the state's lifetime
        self.canvas_score = canvas_score 
        
        self.primitive: BrushStroke2D = primitive
        self.height_map = height_map
        
        self.h, self.w = height_map.shape[:2]
        self.canvas_h, self.canvas_w = self.target.shape[:2]
        self.recalculate_score = recalculate_score
        self.pixel_discard_probability = pixel_discard_probability
        
    def energy(self):
        if self.recalculate_score:
            start_time = time.time()
            color, err_dict = self.primitive.optimal_color_and_error_fast(self.target, self.current)
            end_time = time.time()
            logging.info(f"optimal_color_and_error_fast took {end_time - start_time:.6f} seconds")

            if np.any(color) == None or err_dict is None:
                return np.inf

            self.primitive.color = color
            logging.debug(f"in energy calc - color: {color}")

            logging.debug(f"err calc: {err_dict}")
            err_reduction = err_dict['newPatchError'] - err_dict['oldPatchError']
            logging.debug(f"err reduction: {err_reduction}")

            self.score = self.canvas_score + err_reduction
            self.recalculate_score = False
            logging.info(f"total energy recalc took {time.time() - start_time:.6f} seconds\n")

        return self.score


    def do_move(self):
        old_state = self.copy()
        mutate_p = self.primitive
        mutate_p.mutate()
        self.primitive = mutate_p
        #self.primitive.mutate()
        
        self.recalculate_score = True
        return old_state

    def undo_move(self, old_state):
        self.score = old_state.score
        self.primitive = old_state.primitive

    def copy(self):
        new_state = State(
            height_map=self.height_map, 
            target=self.target, 
            current=self.current,
            score=self.score, 
            pixel_discard_probability=self.pixel_discard_probability,
            canvas_score=self.canvas_score, 
            primitive=self.primitive.copy(),
            recalculate_score=self.recalculate_score
        )
        return new_state
    