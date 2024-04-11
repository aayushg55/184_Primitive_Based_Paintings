import multiprocessing
import random

import numpy as np
from worker import Worker
from primitives import Primitives

class Model:
    def __init__(self, source_img, output_h, output_w, num_workers):
        #TODO: decide format of inp & target img (np, etc.)
        self.source_img = source_img
        self.source_h = source_img.shape[0]
        self.source_w = source_img.shape[1]
        self.output_h = output_h
        self.output_w = output_w
        self.background_color = np.mean(source_img, axis=(0, 1))
        
        self.current_img = np.zeros((output_h, output_w, 3), dtype=np.s)
        self.scores = []
        self.primitives = []
        self.colors = []
        self.workers = []
        self.queue = multiprocessing.Queue()
        
        for _ in range(num_workers):
            worker = Worker(self.target, self.current, self.score, self.queue)
            self.workers.append(worker)

    def find_primitive_init_config(self, primitive_type, alpha, num_trials):
        # Start all the workers
        for worker in self.workers:
            worker.start()

        # Wait for all the workers to finish
        for worker in self.workers:
            worker.join()

        # Collect the results from the queue
        results = []
        while not self.queue.empty():
            result = self.queue.get()
            results.append(result)

        # Process the results
        # ...
        
    def step(self, primitive_type, alpha, num_opt_iter, num_init_iter):
        self.find_primitive_init_config(primitive_type, alpha, num_init_iter)
        
        # model.add(primitive, alpha)
        # for i := 0; i < num_opt_iter; i++ {
		# state.Worker.Init(model.Current, model.Score)
        #     a := state.Energy()
        #     state = HillClimb(state, 100).(*State)
        #     b := state.Energy()
        #     if a == b {
        #         break
        #     }
		# model.Add(state.Shape, state.Alpha)
	
        
    def add(self, prim, alpha):
        prev_img = self.current_img.copy()
        
       	# before := copyRGBA(model.Current)
        # lines := shape.Rasterize()
        # color := computeColor(model.Target, model.Current, lines, alpha)
        # drawLines(model.Current, color, lines)
        # score := differencePartial(model.Target, before, model.Current, model.Score, lines)

        # model.Score = score
        # model.Shapes = append(model.Shapes, shape)
        # model.Colors = append(model.Colors, color)
        # model.Scores = append(model.Scores, score)

        # model.Context.SetRGBA255(color.R, color.G, color.B, color.A)
        # shape.Draw(model.Context, model.Scale) 