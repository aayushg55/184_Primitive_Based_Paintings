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
    
    @abstractmethod
    def setColour(self, colour):
        pass
    
    @abstractmethod    
    def fill(self, image):
        pass
    
    @abstractmethod
    def svg(self):
        pass
    
    
    
    