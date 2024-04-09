from abc import ABC, abstractmethod
class Primitives:
    def __init__(self, shapeType, colour, alpha, isBrushed):
        self.colour = colour
        self.alpha = alpha
        self.isBrushed = isBrushed
        self.shapeType = shapeType
    @abstractmethod    
    def info(self):
        pass
    
    def fill(self):
        pass
    

    
    
    
    