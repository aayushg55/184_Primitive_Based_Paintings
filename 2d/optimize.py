import numpy as np
from state import State

def hill_climb(state: State, num_iter):
    state = state.copy()
    best_state = state.copy()
    best_energy = state.energy()
    
    for i in range(num_iter):
        print(f"hill climb iteration {i}")
        old_state = state.do_move()
        new_energy = state.energy()
        print(f"new energy: {new_energy}, old energy: {best_energy}")
        if new_energy < best_energy:
            print(f"accepted move")
            print(f"new prim (t,theta,color): {state.primitive.t}, {state.primitive.theta}, {state.primitive.color}")
            best_energy = new_energy
            best_state = state
        else:
            print(f"rejected move")
            state.undo_move(old_state)
            
    print(f"at end of hill climb, best energy var: {best_energy}, best energy in best_state: {best_state.energy()}")    
    return best_state.copy()