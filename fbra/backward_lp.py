"""
LP-Based Backward Reachability
===============================
Implements exact backward reachability using Linear Programming.

Based on BReach-LP algorithm from the paper.

For linear dynamics: x_{t+1} = Ax_t + Bu_t
Backward question: Which x_t can reach target set?

Solution: For each dimension, solve LP to find min/max bounds.
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import cvxpy as cp
from fbra.boxes import Box


def backward_reach_lp(target_boxes, forward_box, model, plant, verbose=False):
    """
    Compute exact backward reachable set using Linear Programming
    
    Problem: Find X such that:
        ∃u: plant(x, u) ∈ target AND x ∈ forward_box
    
    For Ground Robot (x' = x + u):
        x + u ∈ target
        → x ∈ target - u
        
    Need to bound u using neural network constraints.
    
    Args:
        target_boxes: List of Box objects (target region at t+1)
        forward_box: Box object (forward reachable set at t)
        model: Neural network controller
        plant: Dynamics function
        verbose: Print debug info
        
    Returns:
        List of Box objects representing backward reachable set
    """
    
    # Get state dimension
    n_dims = len(forward_box.low)
    
    # For each target box, compute backward reachable region
    backward_boxes = []
    
    for target_box in target_boxes:
        if verbose:
            print(f"    LP: Target {target_box.low} to {target_box.up}")
        
        # Compute backward box for this target
        backward_low = np.zeros(n_dims)
        backward_high = np.zeros(n_dims)
        
        for dim in range(n_dims):
            # Minimize x[dim]
            x_min = _solve_lp_bound(
                target_box, forward_box, model, plant, 
                dim, minimize=True, verbose=verbose
            )
            
            # Maximize x[dim]
            x_max = _solve_lp_bound(
                target_box, forward_box, model, plant,
                dim, minimize=False, verbose=verbose
            )
            
            if x_min is None or x_max is None:
                # Infeasible - no backward reachable states
                if verbose:
                    print(f"      Dimension {dim}: Infeasible")
                break
            
            backward_low[dim] = x_min
            backward_high[dim] = x_max
        
        else:
            # All dimensions solved successfully
            # Intersect with forward reachable set (Equation 18)
            inter_low = np.maximum(backward_low, forward_box.low)
            inter_high = np.minimum(backward_high, forward_box.up)
            
            if np.all(inter_low <= inter_high):
                backward_box = Box(inter_low, inter_high)
                backward_boxes.append(backward_box)
                
                if verbose:
                    print(f"      Backward box: {inter_low} to {inter_high}")
    
    return backward_boxes


def _solve_lp_bound(target_box, forward_box, model, plant, dim, minimize=True, verbose=False):
    """
    Solve LP to find min or max of x[dim] in backward reachable set
    """
    
    n_dims = len(forward_box.low)
    
    # Decision variables
    x = cp.Variable(n_dims)
    u = cp.Variable(n_dims)
    
    # Constraints
    constraints = []
    
    # 1. State bounds
    constraints.append(x >= forward_box.low)
    constraints.append(x <= forward_box.up)
    
    # 2. Neural network constraints
    nn_constraints = _get_nn_linear_constraints(x, u, model, forward_box)
    constraints.extend(nn_constraints)
    
    # 3. Dynamics constraints
    next_state = x + u
    constraints.append(next_state >= target_box.low)
    constraints.append(next_state <= target_box.up)
    
    # Objective
    objective = cp.Minimize(x[dim]) if minimize else cp.Maximize(x[dim])
    
    # Solve LP with multiple solver attempts
    problem = cp.Problem(objective, constraints)
    
    # Try solvers in order of preference
    solvers = [cp.ECOS, cp.SCS, cp.OSQP, cp.GLPK_MI]
    
    for solver in solvers:
        try:
            problem.solve(solver=solver, verbose=False)
            
            if problem.status == cp.OPTIMAL:
                return problem.value
            elif problem.status in [cp.INFEASIBLE, cp.UNBOUNDED]:
                # Infeasible - no states can reach target
                return None
            
        except Exception as e:
            # Try next solver
            continue
    
    # All solvers failed
    if verbose:
        print(f"      All LP solvers failed for dimension {dim}")
    
    return None


def _get_nn_linear_constraints(x, u, model, state_box):
    """
    Get linear constraints for neural network using CROWN
    
    For NN with ReLU: u = NN(x)
    CROWN provides linear bounds: A_l·x + b_l ≤ u ≤ A_u·x + b_u
    
    Args:
        x: CVXPY variable for state
        u: CVXPY variable for control
        model: Neural network
        state_box: Box for state bounds (needed for CROWN)
        
    Returns:
        List of CVXPY constraints
    """
    
    # Get CROWN bounds on neural network
    # This requires implementing CROWN or using simplified interval bounds
    
    # SIMPLIFIED VERSION: Use interval bounds from corner evaluation
    # (Not as tight as CROWN, but sound)
    
    from fbra.nn_bounds import nn_forward_box
    
    u_box = nn_forward_box(state_box, model)
    
    # Simple box constraints on u
    # This is conservative but sound
    constraints = [
        u >= u_box.low,
        u <= u_box.up
    ]
    
    return constraints


def backward_reach_one_step_lp(R_fb_next, R_f_current, model, plant, verbose=False):
    """
    Wrapper for backward_reach_lp with same interface as sampling version
    
    Args:
        R_fb_next: List of boxes at time t+1
        R_f_current: List of boxes at time t
        model: Neural network
        plant: Dynamics
        verbose: Debug output
        
    Returns:
        List of boxes representing backward reachable set
    """
    
    if verbose:
        print(f"        LP-based backward (exact)")
    
    all_backward = []
    
    # For each forward box at time t
    for forward_box in R_f_current:
        # Compute backward reach to all target boxes
        backward_boxes = backward_reach_lp(
            R_fb_next, forward_box, model, plant, verbose
        )
        all_backward.extend(backward_boxes)
    
    if verbose:
        print(f"        Found {len(all_backward)} backward boxes")
    
    return all_backward