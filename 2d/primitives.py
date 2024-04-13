from abc import ABC, abstractmethod
class Primitives:
    def __init__(self):
        # self.colour = colour
        # self.isBrushed = isBrushed
        # self.shapeType = shapeType
        pass
        
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
    
    def optimal_color_full(self, targetImage, currentCanvas):
        pass
    
    
    
    
    