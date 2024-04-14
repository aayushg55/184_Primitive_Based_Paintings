import numpy as np
from brush_stroke_2d import BrushStroke2D
from core import *

class State:
    def __init__(self, height_map, target, current, score=np.inf, primitive=None):
        self.target = target
        self.current = current
        
        self.score = score
        self.primitive = primitive
        self.height_map = height_map
        
        self.h, self.w = height_map.shape[:2]
        self.canvas_h, self.canvas_w = self.target.shape[:2]
        self.buffer = np.zeros_like(target)
        
        
        
    def energy(self):
        #colour = self.primitive.optimal_color_full(self.target, self.current)
        colour = self.primitive.optimal_color_fast(self.target, self.current)
        self.primitive.color = colour
        err_dict = self.primitive.get_patch_error(self.target, self.current, colour)
        print("err calc: ", err_dict)
        self.score += err_dict['newPatchError'] - err_dict['oldPatchError']   
        return self.score

    def do_move(self):
        old_state = self.copy()
        self.primitive.mutate()

        # self.score = -1
        return old_state

    def undo_move(self, old_state):
        self.score = old_state.score
        self.primitive = old_state.primitive

    def copy(self):
        new_state = State(self.height_map, self.target, self.current, self.score, self.primitive.copy())
        new_state.buffer = self.buffer.copy()
        return new_state
    