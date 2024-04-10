from primitives import Primitives
import numpy as np
import cv2

class Triangle(Primitives):
    def __init__(self, colour, alpha, isBrushed, vertA, vertB, vertC, W, H):
        super().__init__("Triangle", colour, alpha, isBrushed)
        self.vertA = (random.uniform(0, W), random.uniform(0, H))
        self.vertB = (random.uniform(0, W), random.uniform(0, H))
        self.vertC = (random.uniform(0, W), random.uniform(0, H))
   
    def info(self):
        return f"Triangle with vertices {self.vertA}, {self.vertB}, {self.vertC}; colour {self.colour}; alpha {self.alpha}; isBrushed {self.isBrushed}."
    
    def setColour(self, colour):
        self.colour=colour
    
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
        
    def svg(self):
        rgb_color = f"rgb{self.color}"

        opacity = self.alpha
        

        points_str = f"M {self.vertA[0]},{self.vertA[1]} L {self.vertB[0]},{self.vertB[1]} L {self.vertC[0]},{self.vertC[1]} Z"
        
        svg_str = f'<svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" version="1.1">\
<path d="{points_str}" fill="{rgb_color}" fill-opacity="{opacity}" /></svg>'
        
        return svg_str
        

    
    def area(self):
        ab = np.array(self.vertB) - np.array(self.vertA)
        ac = np.array(self.vertC) - np.array(self.vertA)
        return 0.5 * np.linalg.norm(np.cross(ab, ac))
    
