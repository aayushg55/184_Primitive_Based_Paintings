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
        
        self.h, self.w = height_map.shape
        self.buffer = np.zeros_like(target)
        
    def energy(self):
        if self.score < 0:
            colour = self.primitive.optimal_color_full(self.target, self.current)
            err_dict = self.primitive.get_patch_error(self.target, self.current, colour)
             
            self.score += err_dict['newPatchError'] - err_dict['oldPatchError']
            
        return self.score

    def do_move(self):
        old_state = self.copy()
        self.primitive.mutate()

        self.score = -1
        return old_state

    def undo_move(self, old_state):
        self.state.score = old_state.score
        self.state.primitive = old_state.primitive

    def copy(self):
        new_state = State(self.height_map, self.target, self.current, self.score, self.primitive.copy())
        new_state.buffer = self.buffer.copy()
        return new_state
    