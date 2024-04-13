import multiprocessing
import random
import time
import numpy as np
from brush_stroke_2d import BrushStroke2D
from optimize import *

class Worker(multiprocessing.Process):
    def __init__(self, state_queue, result_queue, worker_idx, num_explorations, num_opt_iter, num_random_state_tries):
        super().__init__()
        self.seed = worker_idx
        self.state = None
        self.state_queue = state_queue
        self.result_queue = result_queue

        np.random.seed(self.seed)
        
        self.num_explorations = num_explorations
        self.num_opt_iter = num_opt_iter
        self.num_random_state_tries = num_random_state_tries

        """
        t ShapeType, a, n, age, m int
        """


    def run(self):
        self.state = self.state_queue.get()
        self.best_hill_climb_state()
        
        self.result_queue.put(self.state)
    
    def best_hill_climb_state(self):
        
        best_energy = np.inf
        best_state = None
        
        for i in range(self.num_explorations):
            state = self.best_random_state()
            before = state.energy()
            state = hill_climb(state, self.num_opt_iter)
            energy = state.energy()
            
            if i == 0 or energy < best_energy:
                best_energy = energy
                best_state = state
        
        return best_state
    
    def best_random_state(self):
        best_energy = np.inf
        best_state = None
        
        for i in range(self.num_random_state_tries):
            state = self.random_state(primitive, depth_map)
            energy = state.energy()
            
            if i == 0 or energy < best_energy:
                best_energy = energy
                best_state = state
        
        return best_state

    def compute_energy(self):
        # Compute the energy or score based on the current state
        return