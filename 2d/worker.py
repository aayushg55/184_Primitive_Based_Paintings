import multiprocessing
from multiprocessing import JoinableQueue
import random
import time
import numpy as np
from brush_stroke_2d import BrushStroke2D
from optimize import *
from state import *
import os

class Worker:
    def __init__(self, worker_idx, num_explorations, num_opt_iter, num_random_state_trials):
        self.seed = worker_idx
        self.state: State = None

        np.random.seed(self.seed)
        
        self.num_explorations = num_explorations
        self.num_opt_iter = num_opt_iter
        self.num_random_state_trials = num_random_state_trials


    def run(self, state):
        self.state = state
        self.best_hill_climb_state()
        print("finished best hill climb in worker \n")
        return self.state


    def best_hill_climb_state(self):
        best_energy = np.inf
        best_primitive = None
        
        for i in range(self.num_explorations):
            print("exploration: ", i)
            state = self.best_random_state()
            print("found best random state")
            start_energy = state.energy()
            print("about to hill climb")
            state = hill_climb(state, self.num_opt_iter)
            print("hill climbed")
            energy = state.energy()
            if energy < best_energy:
                best_energy = energy
                best_primitive = state.primitive
            print()
                
        print("finished explorations, best_energy is ", best_energy)
        self.state.primitive = best_primitive
        self.state.recalculate_score = False
        self.state.score = best_energy
        
        return self.state
    
    def best_random_state(self):
        best_energy = np.inf
        best_primitive = None
        
        for i in range(self.num_random_state_trials):
            state = self.random_state()
            print("iter: ", i, ". trying random state: ", state.primitive.t, state.primitive.theta)
            energy = state.energy()
            print("random state energy: ", energy)
            if energy < best_energy:
                best_energy = energy
                best_primitive = state.primitive
                
        # print("best_random_state energye: ", best_energy)
        # print("color after best random_state: ", best_primitive.color)
        self.state.primitive = best_primitive
        self.state.recalculate_score = state.recalculate_score
        self.state.score = state.score
        print(f"best random state energy: {best_energy}, recalculate_score: {self.state.recalculate_score}")
        print(f"best random state (t,theta,color): {best_primitive.t}, {best_primitive.theta}, {best_primitive.color}")
        return self.state
    
    def random_state(self):
        # Sets state to have a new randomly perturbed primitive
        self.state.primitive = BrushStroke2D(self.state.height_map, self.state.canvas_h, self.state.canvas_w)
        self.state.recalculate_score = True
        return self.state
