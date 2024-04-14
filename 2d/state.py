import numpy as np
from brush_stroke_2d import BrushStroke2D
from core import *

class State:
    def __init__(self, height_map, target, current, score=np.inf, canvas_score=np.inf, primitive=None, recalculate_score=False):
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
        self.buffer = np.zeros_like(target)
        self.recalculate_score = recalculate_score
        
    def energy(self):
        # colour = self.primitive.optimal_color_full(self.target, self.current)
        if self.recalculate_score:
            colour = self.primitive.optimal_color_fast(self.target, self.current)
            self.primitive.color = colour
            err_dict = self.primitive.get_patch_error(self.target, self.current, colour)
            print("err calc: ", err_dict)
            err_reduction = err_dict['newPatchError'] - err_dict['oldPatchError']
            print(f"err reduction: {err_reduction}")
            self.score = self.canvas_score + err_reduction # score for current canvas with current primitive added
            self.recalculate_score = False 
        return self.score

    def do_move(self):
        old_state = self.copy()
        self.primitive.mutate()

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
            canvas_score=self.canvas_score, 
            primitive=self.primitive.copy(),
            recalculate_score=self.recalculate_score
        )
        new_state.buffer = self.buffer.copy()
        return new_state
    