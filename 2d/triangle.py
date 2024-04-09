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
        overlay = image.copy()
        output = image.copy()
        points = np.array([self.vertA, self.vertB, self.vertC])
        cv2.fillPoly(image, [points], color=self.colour)
        cv2.addWeighted(overlay, self.alpha, output, 1 - self.alpha, 0, output)
        return output
    
    def area(self):
        ab = np.array(self.vertB) - np.array(self.vertA)
        ac = np.array(self.vertC) - np.array(self.vertA)
        return 0.5 * np.linalg.norm(np.cross(ab, ac))
    
