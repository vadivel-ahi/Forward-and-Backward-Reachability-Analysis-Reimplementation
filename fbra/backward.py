"""
Backward Reachability Analysis - Hybrid Implementation
=======================================================
Supports both sampling-based and LP-based backward reach.

Usage:
    backward_reach_one_step(..., method="sampling")  # Fast, approximate
    backward_reach_one_step(..., method="lp")        # Slow, exact
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from fbra.boxes import Box


def backward_reach_one_step(R_fb_next, R_f_current, model, plant, 
                           method="lp", samples=500, verbose=False):
    """
    Compute one-step backward reachable set
    
    Supports multiple methods:
        - "lp": Linear Programming (exact, paper-accurate)
        - "sampling": Monte Carlo sampling (approximate, faster)
    
    Args:
        R_fb_next: Target boxes at time t+1
        R_f_current: Forward reachable boxes at time t
        model: Neural network controller
        plant: System dynamics
        method: "lp" or "sampling"
        samples: Number of samples (for sampling method)
        verbose: Print debug info
        
    Returns:
        List of boxes representing backward reachable set
    """
    
    if method == "lp":
        from fbra.backward_lp import backward_reach_one_step_lp
        return backward_reach_one_step_lp(
            R_fb_next, R_f_current, model, plant, verbose
        )
    
    elif method == "sampling":
        return _backward_reach_sampling(
            R_fb_next, R_f_current, model, plant, samples, verbose
        )
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'lp' or 'sampling'")


def _backward_reach_sampling(R_fb_next, R_f_current, model, plant, samples, verbose):
    """
    Sampling-based backward reachability (legacy)
    """
    
    from utils.sampling import sample_box
    from utils.merge import merge_box_list
    import torch
    
    if verbose:
        print(f"        Sampling-based backward ({samples} samples)")
    
    result_boxes = []
    
    # Merge forward boxes for sampling
    if len(R_f_current) == 1:
        merged_box = R_f_current[0]
    else:
        merged_box = merge_box_list(R_f_current)
    
    # For each target box
    for target_box in R_fb_next:
        samples_array = sample_box(merged_box, samples)
        reaching_points = []
        
        for x in samples_array:
            x_tensor = torch.tensor(x, dtype=torch.float32)
            u = model(x_tensor).detach().numpy()
            
            x_box = Box(x, x)
            u_box = Box(u, u)
            next_box = plant(x_box, u_box)
            
            if target_box.intersects(next_box):
                reaching_points.append(x)
        
        if len(reaching_points) > 0:
            reaching_points = np.array(reaching_points)
            backward_box = Box(
                low=reaching_points.min(axis=0),
                up=reaching_points.max(axis=0)
            )
            result_boxes.append(backward_box)
    
    if verbose:
        print(f"        Found {len(result_boxes)} backward boxes")
    
    return result_boxes


# Backward compatibility
def backward_step(R_fb_t, R_f_tminus1, model, plant, samples=500):
    """Legacy function"""
    return backward_reach_one_step(
        R_fb_t, R_f_tminus1, model, plant, 
        method="sampling", samples=samples
    )