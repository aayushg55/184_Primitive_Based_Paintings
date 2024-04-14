import multiprocessing
from multiprocessing import JoinableQueue
import random
import time
import numpy as np
from brush_stroke_2d import BrushStroke2D
from optimize import *
from state import *
import os

class Worker(multiprocessing.Process):
    def __init__(self, state_queue, result_queue, worker_idx, num_explorations, num_opt_iter, num_random_state_trials):
        super().__init__()
        self.seed = worker_idx
        self.state = None
        self.state_queue = state_queue
        self.result_queue = result_queue

        np.random.seed(self.seed)
        
        self.num_explorations = num_explorations
        self.num_opt_iter = num_opt_iter
        self.num_random_state_trials = num_random_state_trials


    def run(self):
        try: 
            self.state = self.state_queue.get()
            self.best_hill_climb_state()
            print("finished best hill climb in worker")
            print(f"putting {self.state} onto queue")
            self.result_queue.put(self.state)
            print("put onto queue!!")
        except Exception as e:
            print(f"Error in worker: {e}")
        finally:
            print("Worker is exiting...")
            # os._exit(0)  # Forcefully terminates the current process



    def best_hill_climb_state(self):
        best_energy = np.inf
        best_primitive = None
        
        for i in range(self.num_explorations):
            state = self.best_random_state()
            start_energy = state.energy()
            state = hill_climb(state, self.num_opt_iter)
            energy = state.energy()
            if energy < best_energy:
                best_energy = energy
                best_primitive = state.primitive
        
        self.state.primitive = best_primitive
        return self.state
    
    def best_random_state(self):
        best_energy = np.inf
        best_primitive = None
        
        for i in range(self.num_random_state_trials):
            state = self.random_state()
            print(state.primitive.t, state.primitive.theta)
            energy = state.energy()
            print(energy)
            if energy < best_energy:
                best_energy = energy
                best_primitive = state.primitive
                
        print("best_random_state energy: ", best_energy)
        print("color after best random_state: ", best_primitive.color)
        self.state.primitive = best_primitive
        return self.state
    
    def random_state(self):
        # Sets state to have a new randomly perturbed primitive
        self.state.primitive = BrushStroke2D(self.state.height_map)
        # self.state.score = -1
        return self.state
