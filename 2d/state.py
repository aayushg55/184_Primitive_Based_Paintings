import numpy as np
from brush_stroke_2d import BrushStroke2D
class State:
    def __init__(self, height_map, target, current):
        self.target = target
        self.current = current
        
        self.score = np.inf
        self.primitive = BrushStroke2D(self.state.height_map, self.target, self.current)
        self.height_map = height_map
        
        self.h, self.w = height_map.shape
        self.buffer = np.zeros_like(target)

        
    def energy(self):
        colour = self.primitive.optimal_color_full(self.target, self.current)
        drawShape(self.current, self.primitive, colour, self.height_map)
        
		return differencePartial(self.current, self.target, self.primitive, self.score)

    #TODO: implement this
    def copy(self):
        new_state = State(self.height_map, self.target, self.current)
        new_state.score = self.score
        new_state.buffer = self.buffer.copy()
        return new_state
    