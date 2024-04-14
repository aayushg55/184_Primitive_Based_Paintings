import numpy as np
from state import State

def hill_climb(state: State, num_iter):
    state = state.copy()
    best_state = state.copy()
    best_energy = state.energy()
    
    for i in range(num_iter):
        old_state = state.do_move()
        new_energy = state.energy()
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_state = state
        else:
            state.undo_move(old_state)
            
    return best_state.copy()