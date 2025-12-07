"""
FBRA Verifier - Complete Implementation with Partitioning
==========================================================
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from fbra.boxes import Box
from fbra.forward import forward_reach_one_step
from fbra.backward import backward_reach_one_step


class VerificationResult:
    """Store verification results"""
    def __init__(self):
        self.status = None
        self.R_forward = {}
        self.R_backward = {}
        self.partitions = []
        self.pruning = []
        self.time_taken = 0.0
        self.iterations = 0


def verify_fbra(X0, model, plant, unsafe, T, max_iterations=10, verbose=True):
    """Main FBRA verification entry point"""
    
    result = VerificationResult()
    X0_partitions = [X0] if isinstance(X0, Box) else X0
    
    if verbose:
        print("\n" + "="*70)
        print("FBRA VERIFICATION - Paper-Accurate Implementation")
        print("="*70)
        print(f"Time horizon: T = {T}")
        print(f"Initial partitions: {len(X0_partitions)}")
        print("="*70 + "\n")
    
    all_safe = True
    any_unsafe = False
    any_unknown = False
    
    for partition_idx, X0_sub in enumerate(X0_partitions):
        if verbose and len(X0_partitions) > 1:
            print(f"\n--- Verifying Partition {partition_idx + 1}/{len(X0_partitions)} ---")
        
        status = verify_partition(
            X0_sub, model, plant, unsafe, T, 
            result, max_iterations, verbose, depth=0
        )
        
        if status == "Unsafe":
            any_unsafe = True
            all_safe = False
            result.status = "Unsafe"
            if verbose:
                print(f"\n✗ Partition {partition_idx + 1} is UNSAFE")
            return result
        
        elif status == "Unknown":
            any_unknown = True
            all_safe = False
            if verbose:
                print(f"\n? Partition {partition_idx + 1} is UNKNOWN")
        
        elif status == "Safe":
            if verbose:
                print(f"\n✓ Partition {partition_idx + 1} is SAFE")
    
    if any_unsafe:
        result.status = "Unsafe"
    elif any_unknown:
        result.status = "Unknown"
    elif all_safe:
        result.status = "Safe"
    else:
        result.status = "Unknown"
    
    if verbose:
        print("\n" + "="*70)
        print(f"✓ VERIFICATION COMPLETE: {result.status}")
        print(f"Total iterations: {result.iterations}")
        if result.status == "Unknown":
            print("Note: Unknown results require deeper partitioning or longer analysis")
        print("="*70 + "\n")
    
    return result


def verify_partition(X0, model, plant, unsafe, T, result, max_iterations, verbose, depth):
    """Verify a single partition with recursive partitioning support"""
    
    MAX_DEPTH = 3
    iteration = 0
    ever_detected_unknown = False
    
    while iteration < max_iterations:
        iteration += 1
        result.iterations += 1
        
        if verbose:
            indent = "  " * depth
            print(f"\n{indent}{'─'*70}")
            print(f"{indent}Iteration {iteration} (Depth {depth})")
            print(f"{indent}{'─'*70}")
        
        R_f = {0: [X0]}
        completed_all_safe = True
        
        for t in range(T):
            if verbose:
                indent = "  " * depth
                print(f"\n{indent}  Forward Step t={t} → t={t+1}")
            
            R_f[t+1] = forward_reach_one_step(R_f[t], model, plant)
            
            if verbose:
                print(f"{indent}    Boxes at t={t+1}: {len(R_f[t+1])}")
            
            status = safety_check(R_f[t+1], unsafe, verbose, depth)
            
            if verbose:
                print(f"{indent}    Status: {status}")
            
            if status == "Unsafe":
                if verbose:
                    print(f"{indent}    ✗ UNSAFE detected")
                return "Unsafe"
            
            elif status == "Safe":
                if verbose:
                    print(f"{indent}    ✓ Safe")
                continue
            
            elif status == "Unknown":
                ever_detected_unknown = True
                completed_all_safe = False
                
                if verbose:
                    print(f"{indent}    ? Unknown - refinement needed")
                
                R_fb = extract_intersection(R_f[t+1], unsafe)
                
                if verbose:
                    print(f"{indent}      Intersection: {len(R_fb)} boxes")
                
                refine_result = backward_refinement(
                    R_f, R_fb, t+1, model, plant, X0, unsafe, verbose, depth
                )
                
                if verbose:
                    print(f"{indent}      Refinement: {refine_result}")
                
                if refine_result == "restart":
                    if depth >= MAX_DEPTH:
                        if verbose:
                            print(f"{indent}      Max depth reached")
                        return "Unknown"
                    
                    if verbose:
                        print(f"{indent}      → Partitioning X0")
                    
                    from fbra.partition import partition_initial_set
                    
                    R_b_0 = compute_backward_to_zero(R_f, R_fb, t+1, model, plant)
                    
                    X0_parts = partition_initial_set(X0, R_b_0, unsafe, "uniform", 2)
                    
                    if verbose:
                        print(f"{indent}      Created {len(X0_parts)} partitions")
                    
                    all_safe = True
                    
                    for i, X0_sub in enumerate(X0_parts):
                        if verbose:
                            print(f"\n{indent}      === Part {i+1}/{len(X0_parts)} ===")
                        
                        sub_status = verify_partition(
                            X0_sub, model, plant, unsafe, T,
                            result, max_iterations, verbose, depth + 1
                        )
                        
                        if sub_status == "Unsafe":
                            if verbose:
                                print(f"{indent}      Part {i+1}: UNSAFE")
                            return "Unsafe"
                        elif sub_status != "Safe":
                            all_safe = False
                    
                    return "Safe" if all_safe else "Unknown"
                
                elif refine_result == "pruned":
                    if verbose:
                        print(f"{indent}      → Pruned")
                    continue
                
                elif refine_result == "unsafe":
                    return "Unsafe"
                
                break
        
        if completed_all_safe:
            if verbose:
                indent = "  " * depth
                print(f"\n{indent}  ✓ All timesteps safe")
            result.R_forward = R_f
            return "Safe"
    
    if verbose:
        indent = "  " * depth
        print(f"\n{indent}  ⚠ Max iterations")
    
    return "Unknown" if ever_detected_unknown else "Safe"


def safety_check(boxes, unsafe, verbose, depth):
    """Check safety status"""
    indent = "  " * depth
    
    has_intersection = False
    has_fully_unsafe = False
    
    for box in boxes:
        if box.intersects(unsafe):
            has_intersection = True
            if unsafe.contains(box):
                has_fully_unsafe = True
    
    if not has_intersection:
        return "Safe"
    elif has_fully_unsafe:
        return "Unsafe"
    else:
        if verbose:
            print(f"{indent}      Partial intersection (Unknown)")
        return "Unknown"


def extract_intersection(boxes, unsafe):
    """Extract boxes intersecting unsafe"""
    result = []
    for box in boxes:
        inter = box.intersect(unsafe)
        if inter is not None:
            result.append(inter)
    return result


def backward_refinement(R_f, R_fb, t_current, model, plant, X0, unsafe, verbose, depth):
    """Backward reachability refinement"""
    indent = "  " * depth
    R_fb_dict = {t_current: R_fb}
    
    if verbose:
        print(f"\n{indent}    Backward from t={t_current}:")
    
    for tb in range(t_current - 1, -1, -1):
        if verbose:
            print(f"{indent}      t={tb+1} → t={tb}")
        
        R_b = backward_reach_one_step(
                R_fb_dict[tb + 1], R_f[tb], model, plant, 
                method="lp",  # ← Use LP-based backward
                verbose=False
                )
        
        if verbose:
            print(f"{indent}        Back boxes: {len(R_b)}")
        
        R_fb_dict[tb] = intersect_lists(R_b, R_f[tb])
        
        if verbose:
            print(f"{indent}        Intersection: {len(R_fb_dict[tb])}")
        
        if len(R_fb_dict[tb]) == 0:
            if verbose:
                print(f"{indent}        ✓ Empty - spurious")
            return "pruned"
        
        if tb == 0:
            if verbose:
                print(f"{indent}        ! Reached t=0")
            
            hits_X0 = any(b.intersects(X0) for b in R_fb_dict[0])
            
            if hits_X0:
                if verbose:
                    print(f"{indent}        ! Hits X0 - need partition")
                return "restart"
            else:
                if verbose:
                    print(f"{indent}        ✓ Misses X0")
                return "pruned"
    
    return "pruned"


def compute_backward_to_zero(R_f, R_fb, t_current, model, plant):
    """Compute backward reach to t=0"""
    R_fb_dict = {t_current: R_fb}
    
    for tb in range(t_current - 1, -1, -1):
        R_b = backward_reach_one_step(
                R_fb_dict[tb + 1], R_f[tb], model, plant, 
                method="lp",  # ← Use LP-based backward
                verbose=False
                )
        R_fb_dict[tb] = intersect_lists(R_b, R_f[tb])
        if len(R_fb_dict[tb]) == 0:
            return []
    
    return R_fb_dict[0]


def intersect_lists(boxes1, boxes2):
    """Intersect two lists of boxes"""
    result = []
    for b1 in boxes1:
        for b2 in boxes2:
            inter = b1.intersect(b2)
            if inter is not None:
                result.append(inter)
    return result


def prune_boxes(boxes, to_remove):
    """Remove boxes that intersect with to_remove"""
    pruned = []
    for box in boxes:
        keep = True
        for rem in to_remove:
            if box.intersect(rem) is not None:
                keep = False
                break
        if keep:
            pruned.append(box)
    return pruned


def verify(X0, model, plant, unsafe, T, return_all=False):
    """Backward compatibility wrapper"""
    result = verify_fbra(X0, model, plant, unsafe, T, verbose=True)
    
    if return_all:
        return (result.status, result.R_forward, result.R_backward, result.partitions)
    else:
        return result.status