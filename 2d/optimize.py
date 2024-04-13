import numpy as np

def hill_climb(state, num_iter):
    best_state = state.copy()
    best_energy = state.energy()
    
    for i in range(num_iter):
        old_state = state.do_move()
        new_energy = state.energy()
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_state = state.copy()
        else:
            state.undo_move(old_state)
            
    return best_state