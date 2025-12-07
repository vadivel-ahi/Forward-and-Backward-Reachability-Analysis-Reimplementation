"""
Forward Reachability Analysis
==============================
Computes one-step forward reachable sets for NNCS.

Given:
    - Current reachable set R_t (list of boxes)
    - Neural network controller
    - Plant dynamics

Computes:
    - Next reachable set R_{t+1}
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from fbra.nn_bounds import nn_forward_box
from fbra.boxes import Box


def forward_reach_one_step(R_t, model, plant):
    """
    Compute one-step forward reachable set (Paper Equation 8)
    
    For each box X_t in R_t:
        1. Compute control bounds: U_t = κ(n(X_t))
        2. Propagate through dynamics: X_{t+1} = τ(f(X_t, U_t))
    
    Args:
        R_t: List of boxes at time t
        model: Neural network controller
        plant: Dynamics function (box, u_box) -> next_box
        
    Returns:
        R_{t+1}: List of boxes at time t+1
    """
    
    R_next = []
    
    for box in R_t:
        # Step 1: Overapproximate neural network output
        # U_t = κ(n(X_t)) where κ is overapproximation operator
        u_box = nn_forward_box(box, model)
        
        # Step 2: Propagate through plant dynamics
        # X_{t+1} = τ(f(X_t, U_t)) where τ is overapproximation operator
        next_box = plant(box, u_box)
        
        R_next.append(next_box)
    
    return R_next


def forward_reach(X0, model, plant, T):
    """
    Compute forward reachable sets for T timesteps
    
    Legacy function for compatibility
    
    Args:
        X0: Initial set (Box or list of Box)
        model: Neural network
        plant: Dynamics
        T: Time horizon
        
    Returns:
        Dictionary {t: [boxes]} for t=0 to T
    """
    
    # Ensure X0 is a list
    if isinstance(X0, Box):
        R = {0: [X0]}
    else:
        R = {0: X0}
    
    # Forward propagation
    for t in range(T):
        R[t+1] = forward_reach_one_step(R[t], model, plant)
    
    return R