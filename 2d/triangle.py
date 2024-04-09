from primitives import Primitives
import numpy as np
import cv2

class Triangle(Primitives):
    def __init__(self, colour, alpha, isBrushed, vertA, vertB, vertC):
        super().__init__("Triangle", colour, alpha, isBrushed)
        self.vertA=vertA
        self.vertB=vertB
        self.vertC=vertC
   
    def info(self):
        return f"Triangle with vertices {self.vertA}, {self.vertB}, {self.vertC}; colour {self.colour}; alpha {self.alpha}; isBrushed {self.isBrushed}."
    
    def fill(self, image):
        if not(self.isBrushed):
            points = np.array([self.vertA, self.vertB, self.vertC])
            if self.alpha == 1:
                cv2.fillPoly(image, [points], color=self.color)
            else:
                overlay = np.zeros_like(image)
                cv2.fillPoly(overlay, [points], color=self.color)
                cv2.addWeighted(overlay, self.alpha, image, 1 - self.alpha, 0, image)
        else:
            pass
        

    
    def area(self):
        ab = np.array(self.vertB) - np.array(self.vertA)
        ac = np.array(self.vertC) - np.array(self.vertA)
        return 0.5 * np.linalg.norm(np.cross(ab, ac))
    
