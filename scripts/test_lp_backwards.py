"""
Test LP-Based Backward Reachability
====================================
Compares sampling vs LP-based methods.
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import time
import numpy as np
from fbra.boxes import Box
from fbra.backward import backward_reach_one_step
from experiments.controller import ground_robot_controller
from experiments.dynamics import ground_robot


def test_backward_methods():
    """Compare sampling vs LP backward reach"""
    
    print("\n" + "="*70)
    print("COMPARING BACKWARD REACHABILITY METHODS")
    print("="*70)
    
    # Setup test case
    target_box = Box([-1.0, -0.5], [0.5, 0.5])
    forward_box = Box([-3.0, -0.8], [-2.0, 0.8])
    
    print("\nTest Configuration:")
    print(f"  Target:  {target_box.low} to {target_box.up}")
    print(f"  Forward: {forward_box.low} to {forward_box.up}")
    
    # ====================================
    # Method 1: Sampling-Based
    # ====================================
    print("\n" + "-"*70)
    print("Method 1: Sampling-Based (500 samples)")
    print("-"*70)
    
    start = time.time()
    result_sampling = backward_reach_one_step(
        [target_box], [forward_box],
        ground_robot_controller, ground_robot,
        method="sampling", samples=500, verbose=True
    )
    time_sampling = time.time() - start
    
    print(f"\nResults:")
    print(f"  Boxes found: {len(result_sampling)}")
    if result_sampling:
        for i, box in enumerate(result_sampling):
            print(f"  Box {i+1}: {box.low} to {box.up}")
            volume = np.prod(box.up - box.low)
            print(f"         Volume: {volume:.6f}")
    print(f"  Time: {time_sampling:.4f}s")
    
    # ====================================
    # Method 2: LP-Based
    # ====================================
    print("\n" + "-"*70)
    print("Method 2: LP-Based (Exact)")
    print("-"*70)
    
    start = time.time()
    result_lp = backward_reach_one_step(
        [target_box], [forward_box],
        ground_robot_controller, ground_robot,
        method="lp", verbose=True
    )
    time_lp = time.time() - start
    
    print(f"\nResults:")
    print(f"  Boxes found: {len(result_lp)}")
    if result_lp:
        for i, box in enumerate(result_lp):
            print(f"  Box {i+1}: {box.low} to {box.up}")
            volume = np.prod(box.up - box.low)
            print(f"         Volume: {volume:.6f}")
    print(f"  Time: {time_lp:.4f}s")
    
    # ====================================
    # Comparison
    # ====================================
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    if result_sampling and result_lp:
        vol_sampling = sum(np.prod(b.up - b.low) for b in result_sampling)
        vol_lp = sum(np.prod(b.up - b.low) for b in result_lp)
        
        print(f"\nTotal Volume:")
        print(f"  Sampling: {vol_sampling:.6f}")
        print(f"  LP:       {vol_lp:.6f}")
        print(f"  Difference: {abs(vol_sampling - vol_lp):.6f}")
        
        print(f"\nSpeed:")
        print(f"  Sampling: {time_sampling:.4f}s")
        print(f"  LP:       {time_lp:.4f}s")
        print(f"  Speedup:  {time_sampling/time_lp:.2f}x" if time_lp > 0 else "")
        
        print(f"\nAccuracy:")
        print(f"  LP is exact (guaranteed sound)")
        print(f"  Sampling is approximate (may miss states)")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    test_backward_methods()