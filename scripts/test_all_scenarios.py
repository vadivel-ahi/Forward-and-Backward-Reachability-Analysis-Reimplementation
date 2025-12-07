"""
Comprehensive Test Suite for FBRA
==================================
Tests all three verification outcomes:
    1. SAFE: Controller avoids obstacle
    2. UNSAFE: Buggy controller crashes
    3. UNKNOWN: Needs refinement (tighter unsafe region)
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import time
from fbra.verifier import verify_fbra
from fbra.boxes import Box
from experiments.controller import (
    ground_robot_controller, 
    BuggyGroundRobotController
)
from experiments.dynamics import ground_robot
from experiments.sets import X0_ground_robot, Unsafe_ground_robot


def print_scenario_header(scenario_name):
    """Print formatted scenario header"""
    print("\n" + "="*70)
    print(f"SCENARIO: {scenario_name}")
    print("="*70)


def test_scenario_1_safe():
    """
    Scenario 1: SAFE
    ----------------
    - Safe controller
    - Original unsafe region
    - Expected: Safe (controller avoids obstacle)
    """
    
    print_scenario_header("SAFE - Controller Successfully Avoids Obstacle")
    
    print("\nConfiguration:")
    print(f"  Initial Set: {X0_ground_robot.low} to {X0_ground_robot.up}")
    print(f"  Unsafe Set:  {Unsafe_ground_robot.low} to {Unsafe_ground_robot.up}")
    print(f"  Controller:  Safe (trained to avoid obstacles)")
    print(f"  Time Horizon: T = 9")
    
    start = time.time()
    result = verify_fbra(
        X0=X0_ground_robot,
        model=ground_robot_controller,
        plant=ground_robot,
        unsafe=Unsafe_ground_robot,
        T=9,
        verbose=False  # Less verbose for overview
    )
    end = time.time()
    
    print("\n" + "-"*70)
    print(f"Result: {result.status}")
    print(f"Time: {end - start:.3f}s")
    print(f"Iterations: {result.iterations}")
    print("-"*70)
    
    return result


def test_scenario_2_unsafe():
    """
    Scenario 2: UNSAFE
    ------------------
    - Buggy controller (pushes toward unsafe region)
    - Original unsafe region
    - Expected: Unsafe (controller deliberately crashes)
    """
    
    print_scenario_header("UNSAFE - Buggy Controller Crashes Into Obstacle")
    
    buggy_controller = BuggyGroundRobotController()
    
    print("\nConfiguration:")
    print(f"  Initial Set: {X0_ground_robot.low} to {X0_ground_robot.up}")
    print(f"  Unsafe Set:  {Unsafe_ground_robot.low} to {Unsafe_ground_robot.up}")
    print(f"  Controller:  BUGGY (biased toward unsafe region)")
    print(f"  Time Horizon: T = 9")
    
    start = time.time()
    result = verify_fbra(
        X0=X0_ground_robot,
        model=buggy_controller,
        plant=ground_robot,
        unsafe=Unsafe_ground_robot,
        T=9,
        verbose=False
    )
    end = time.time()
    
    print("\n" + "-"*70)
    print(f"Result: {result.status}")
    print(f"Time: {end - start:.3f}s")
    print(f"Iterations: {result.iterations}")
    print("-"*70)
    
    return result


def test_scenario_3_unknown_needs_refinement():
    """
    Scenario 3: UNKNOWN → SAFE (after refinement)
    ---------------------------------------------
    - Safe controller
    - Larger unsafe region (causes overapproximation overlap)
    - Expected: Unknown initially, then Safe after refinement
    """
    
    print_scenario_header("UNKNOWN - Needs Refinement (Large Unsafe Region)")
    
    # Create a LARGER unsafe region that causes overapproximation issues
    # This makes the forward reachable set overlap due to conservativeness
    larger_unsafe = Box(
        low=[-2.0, -2.0],  # Expanded from [-1, -1]
        up=[2.0, 2.0]       # Expanded from [1, 1]
    )
    
    print("\nConfiguration:")
    print(f"  Initial Set: {X0_ground_robot.low} to {X0_ground_robot.up}")
    print(f"  Unsafe Set:  {larger_unsafe.low} to {larger_unsafe.up} (LARGER)")
    print(f"  Controller:  Safe (but larger unsafe region)")
    print(f"  Time Horizon: T = 9")
    print("\nExpected Behavior:")
    print("  1. Forward reachability will show partial overlap (Unknown)")
    print("  2. Backward analysis checks if overlap is real")
    print("  3. Refinement (partitioning/pruning) resolves Unknown")
    
    start = time.time()
    result = verify_fbra(
        X0=X0_ground_robot,
        model=ground_robot_controller,
        plant=ground_robot,
        unsafe=larger_unsafe,
        T=9,
        verbose=True  # Show refinement process
    )
    end = time.time()
    
    print("\n" + "-"*70)
    print(f"Result: {result.status}")
    print(f"Time: {end - start:.3f}s")
    print(f"Iterations: {result.iterations}")
    print("-"*70)
    
    return result


def test_scenario_4_unknown_near_boundary():
    """
    Scenario 4: UNKNOWN - Very Close to Unsafe Region
    --------------------------------------------------
    - Safe controller
    - Unsafe region positioned where overapproximation causes overlap
    - Expected: Unknown, then Safe after refinement
    """
    
    print_scenario_header("UNKNOWN - Near Boundary Case")
    
    # Position unsafe region where the reachable set comes close
    # Based on forward propagation, the robot moves from left to right
    near_boundary_unsafe = Box(
        low=[0.5, -1.5],   # Positioned in path
        up=[2.5, 1.5]
    )
    
    print("\nConfiguration:")
    print(f"  Initial Set: {X0_ground_robot.low} to {X0_ground_robot.up}")
    print(f"  Unsafe Set:  {near_boundary_unsafe.low} to {near_boundary_unsafe.up}")
    print(f"  Controller:  Safe")
    print(f"  Time Horizon: T = 9")
    print("\nThis tests boundary cases where overapproximation is critical")
    
    start = time.time()
    result = verify_fbra(
        X0=X0_ground_robot,
        model=ground_robot_controller,
        plant=ground_robot,
        unsafe=near_boundary_unsafe,
        T=9,
        verbose=True
    )
    end = time.time()
    
    print("\n" + "-"*70)
    print(f"Result: {result.status}")
    print(f"Time: {end - start:.3f}s")
    print(f"Iterations: {result.iterations}")
    print("-"*70)
    
    return result


def test_scenario_5_guaranteed_unsafe():
    """
    Scenario 5: GUARANTEED UNSAFE
    -----------------------------
    - Initial set INSIDE unsafe region
    - Expected: Immediate Unsafe detection
    """
    
    print_scenario_header("GUARANTEED UNSAFE - Starting Inside Unsafe Region")
    
    # Initial set completely inside unsafe region
    unsafe_start = Box(
        low=[-0.5, -0.5],  # Inside [-1, 1] × [-1, 1]
        up=[0.5, 0.5]
    )
    
    print("\nConfiguration:")
    print(f"  Initial Set: {unsafe_start.low} to {unsafe_start.up}")
    print(f"  Unsafe Set:  {Unsafe_ground_robot.low} to {Unsafe_ground_robot.up}")
    print(f"  Controller:  Safe")
    print(f"  Time Horizon: T = 9")
    print("\nExpected: Immediate Unsafe (already in unsafe region)")
    
    start = time.time()
    result = verify_fbra(
        X0=unsafe_start,
        model=ground_robot_controller,
        plant=ground_robot,
        unsafe=Unsafe_ground_robot,
        T=9,
        verbose=False
    )
    end = time.time()
    
    print("\n" + "-"*70)
    print(f"Result: {result.status}")
    print(f"Time: {end - start:.3f}s")
    print(f"Iterations: {result.iterations}")
    print("-"*70)
    
    return result


def main():
    """Run all test scenarios"""
    
    print("\n" + "="*70)
    print("FBRA COMPREHENSIVE TEST SUITE")
    print("="*70)
    print("\nTesting all verification outcomes:")
    print("  1. SAFE - Successful avoidance")
    print("  2. UNSAFE - Buggy controller")
    print("  3. UNKNOWN → SAFE - Refinement needed (large unsafe)")
    print("  4. UNKNOWN → SAFE - Near boundary case")
    print("  5. GUARANTEED UNSAFE - Starting inside unsafe")
    print("\n" + "="*70)
    
    results = {}
    
    # Test each scenario
    try:
        print("\n" + "🔵 "*25)
        results['safe'] = test_scenario_1_safe()
        
        print("\n" + "🔴 "*25)
        results['unsafe'] = test_scenario_2_unsafe()
        
        print("\n" + "🟡 "*25)
        results['unknown_large'] = test_scenario_3_unknown_needs_refinement()
        
        print("\n" + "🟠 "*25)
        results['unknown_boundary'] = test_scenario_4_unknown_near_boundary()
        
        print("\n" + "⚫ "*25)
        results['guaranteed_unsafe'] = test_scenario_5_guaranteed_unsafe()
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    print(f"\n{'Scenario':<30} {'Result':<15} {'Time':<10} {'Iterations'}")
    print("-"*70)
    
    scenario_names = {
        'safe': 'Safe Controller',
        'unsafe': 'Buggy Controller',
        'unknown_large': 'Unknown (Large Unsafe)',
        'unknown_boundary': 'Unknown (Boundary)',
        'guaranteed_unsafe': 'Guaranteed Unsafe'
    }
    
    for key, result in results.items():
        name = scenario_names[key]
        status = result.status
        time_str = f"{result.time_taken:.3f}s" if hasattr(result, 'time_taken') else "N/A"
        iterations = result.iterations
        
        # Color-code status
        if status == "Safe":
            status_display = f"✓ {status}"
        elif status == "Unsafe":
            status_display = f"✗ {status}"
        else:
            status_display = f"? {status}"
        
        print(f"{name:<30} {status_display:<15} {time_str:<10} {iterations}")
    
    print("\n" + "="*70)
    print("EXPECTED RESULTS:")
    print("  1. Safe Controller:        ✓ Safe")
    print("  2. Buggy Controller:       ✗ Unsafe")
    print("  3. Unknown (Large):        ✓ Safe (after refinement)")
    print("  4. Unknown (Boundary):     ✓ Safe (after refinement)")
    print("  5. Guaranteed Unsafe:      ✗ Unsafe (immediate)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()