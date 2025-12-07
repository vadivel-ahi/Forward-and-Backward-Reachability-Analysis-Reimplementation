"""
Test LP Backward with Reachable Target
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import time
import numpy as np
from fbra.boxes import Box
from fbra.backward import backward_reach_one_step
from experiments.controller import BuggyGroundRobotController
from experiments.dynamics import ground_robot


def test_reachable_backward():
    """Test backward reach with buggy controller (large control)"""
    
    print("\n" + "="*70)
    print("LP BACKWARD TEST - Reachable Target")
    print("="*70)
    
    # Use buggy controller (has large control ~2.0)
    buggy = BuggyGroundRobotController()
    
    # Setup: Target that IS reachable
    target_box = Box([0.0, -0.5], [1.0, 0.5])  # Near unsafe region
    forward_box = Box([-3.0, -0.5], [-1.5, 0.5])  # One step before
    
    print("\nConfiguration:")
    print(f"  Forward (t=1): {forward_box.low} to {forward_box.up}")
    print(f"  Target (t=2):  {target_box.low} to {target_box.up}")
    print(f"  Controller:    Buggy (strong control ~2.0)")
    
    print("\nExpected: Backward set should be NON-EMPTY")
    print("  (Buggy controller can push from [-3,-1.5] to [0,1])")
    
    # ====================================
    # Test Sampling
    # ====================================
    print("\n" + "-"*70)
    print("Sampling-Based Backward")
    print("-"*70)
    
    start = time.time()
    result_sampling = backward_reach_one_step(
        [target_box], [forward_box], buggy, ground_robot,
        method="sampling", samples=1000, verbose=True
    )
    time_sampling = time.time() - start
    
    print(f"\n  Boxes: {len(result_sampling)}")
    if result_sampling:
        for box in result_sampling:
            print(f"    {box.low} to {box.up}")
            print(f"    Volume: {np.prod(box.up - box.low):.6f}")
    print(f"  Time: {time_sampling:.4f}s")
    
    # ====================================
    # Test LP
    # ====================================
    print("\n" + "-"*70)
    print("LP-Based Backward (Exact)")
    print("-"*70)
    
    start = time.time()
    result_lp = backward_reach_one_step(
        [target_box], [forward_box], buggy, ground_robot,
        method="lp", verbose=True
    )
    time_lp = time.time() - start
    
    print(f"\n  Boxes: {len(result_lp)}")
    if result_lp:
        for box in result_lp:
            print(f"    {box.low} to {box.up}")
            print(f"    Volume: {np.prod(box.up - box.low):.6f}")
    print(f"  Time: {time_lp:.4f}s")
    
    # ====================================
    # Comparison
    # ====================================
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    if result_sampling and result_lp:
        print("✓ Both methods found backward reachable states")
        
        vol_sampling = sum(np.prod(b.up - b.low) for b in result_sampling)
        vol_lp = sum(np.prod(b.up - b.low) for b in result_lp)
        
        print(f"\nVolumes:")
        print(f"  Sampling: {vol_sampling:.6f}")
        print(f"  LP:       {vol_lp:.6f}")
        print(f"  Ratio:    {vol_sampling/vol_lp:.3f}" if vol_lp > 0 else "")
        
        if vol_lp > vol_sampling:
            print("\n→ LP found MORE states (more accurate)")
        elif vol_lp < vol_sampling:
            print("\n→ Sampling overestimated (conservative)")
        else:
            print("\n→ Both methods agree")
    
    elif result_lp and not result_sampling:
        print("⚠ LP found states but sampling missed them")
        print("  → Sampling can be incomplete!")
    
    elif result_sampling and not result_lp:
        print("⚠ Sampling found states but LP didn't")
        print("  → Check LP implementation")
    
    else:
        print("✓ Both correctly identified: No backward reachable states")
    
    print("="*70)


if __name__ == "__main__":
    test_reachable_backward()