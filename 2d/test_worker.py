import multiprocessing

class Worker(multiprocessing.Process):
    def __init__(self, name, queue):
        super(Worker, self).__init__()
        self.name = name
        self.queue = queue

    def run(self):
        print(f'Worker {self.name} is doing some work')
        self.queue.put(self.name)

if __name__ == '__main__':
    workers = []
    queue = multiprocessing.Queue()
    for i in range(5):
        worker = Worker(str(i), queue)
        workers.append(worker)

    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()
    
    res = []
    while not queue.empty():
        result = queue.get()
        res.append(result)
    
    print(res)
    print("All workers have finished their work")