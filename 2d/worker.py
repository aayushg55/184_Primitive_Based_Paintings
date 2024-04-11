import multiprocessing
import random
import time

class Worker(multiprocessing.Process):
    def __init__(self, target, current, score, queue):
        super().__init__()
        self.target = target
        self.current = current
        self.score = score
        self.queue = queue
        self.rnd = random.Random()

    def run(self):
        energy = self.compute_energy()
        
        # Put the result in the queue
        self.queue.put(energy)

    def compute_energy(self):
        # Compute the energy or score based on the current state
        return