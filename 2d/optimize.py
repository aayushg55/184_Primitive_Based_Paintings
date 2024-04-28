import numpy as np
from state import State
import logging

def hill_climb(state: State, num_iter):
    state = state.copy()
    best_state = state.copy()
    best_energy = state.energy()
    
    for i in range(num_iter):
        # print(f"On iteration {i}")
        logging.debug(f"hill climb iteration {i}")
        old_state = state.do_move()
        # print(f"old state is {old_state.primitive.t} and  {old_state.primitive.theta}")
        # print(f"the new state is {state.primitive.t} and  {state.primitive.theta}")
        new_energy = state.energy()
        
        logging.debug(f"new energy: {new_energy}, old energy: {best_energy}")
        if new_energy < best_energy:
            # print("Move accepted")
            logging.debug(f"accepted move")
            logging.debug(f"new prim (t,theta,color): {state.primitive.t}, {state.primitive.theta}, {state.primitive.color}")
            best_energy = new_energy
            best_state = state
        else:
            # print((f"rejected move"))
            logging.debug(f"rejected move")
            state.undo_move(old_state)
            
    logging.debug(f"at end of hill climb, best energy var: {best_energy}, best energy in best_state: {best_state.energy()}") 
    return best_state.copy()