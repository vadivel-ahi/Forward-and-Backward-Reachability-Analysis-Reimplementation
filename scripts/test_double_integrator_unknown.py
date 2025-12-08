"""
Force Unknown Status in Double Integrator
==========================================
Creates scenarios that trigger Unknown → partitioning
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from fbra.boxes import Box
from fbra.verifier import verify_fbra
from fbra.forward import forward_reach
from experiments.controller import double_integrator_controller, DoubleIntegratorPDController
from experiments.dynamics import double_integrator


def scenario_unknown_grazing():
    """
    Scenario designed to graze the unsafe region
    Should trigger Unknown → partitioning
    """
    
    print("\n" + "="*70)
    print("SCENARIO: Grazing Unsafe Region (Designed for Unknown)")
    print("="*70)
    
    # Position initial set so reachable set GRAZES unsafe boundary
    X0 = Box([3.5, 0.2], [4.0, 0.6])  # Start very close
    unsafe = Box([4.8, -0.3], [5.5, 0.3])  # Unsafe just ahead
    
    print("\nConfiguration:")
    print(f"  Initial: pos ∈ [3.5, 4.0], vel ∈ [0.2, 0.6]")
    print(f"  Unsafe:  pos ∈ [4.8, 5.5], vel ∈ [-0.3, 0.3]")
    print(f"  Gap: {4.8 - 4.0} meters (very small!)")
    print(f"  Strategy: Positive velocity will push toward unsafe")
    
    # Quick forward check
    print("\nForward reachability preview:")
    R = forward_reach(X0, double_integrator_controller, double_integrator, T=3)
    
    for t in range(4):
        boxes = R[t]
        if boxes:
            box = boxes[0]
            print(f"  t={t}: pos ∈ [{box.low[0]:.3f}, {box.up[0]:.3f}], "
                  f"vel ∈ [{box.low[1]:.3f}, {box.up[1]:.3f}]")
            
            # Check overlap
            if box.intersects(unsafe):
                print(f"        → Intersects unsafe!")
                inter = box.intersect(unsafe)
                if inter:
                    print(f"        → Intersection: pos ∈ [{inter.low[0]:.3f}, {inter.up[0]:.3f}]")
    
    # Run FBRA
    print("\n" + "-"*70)
    print("Running FBRA Verification...")
    print("-"*70)
    
    result = verify_fbra(
        X0, double_integrator_controller, double_integrator,
        unsafe, T=3, max_iterations=10, verbose=True
    )
    
    print("\n" + "="*70)
    print(f"RESULT: {result.status}")
    print(f"Iterations: {result.iterations}")
    print("="*70)
    
    if result.status == "Unknown":
        print("\n✓ SUCCESS: Triggered Unknown status!")
        print("  This demonstrates FBRA's refinement capability")
    elif result.status == "Safe":
        print("\n→ Resolved to Safe (FBRA successfully pruned false positive)")
    elif result.status == "Unsafe":
        print("\n→ Detected Unsafe (found real collision trajectory)")
    
    return result


def scenario_unknown_large_box():
    """
    Use large initial box to force overapproximation overlap
    """
    
    print("\n" + "="*70)
    print("SCENARIO: Large Initial Box (Overapproximation Overlap)")
    print("="*70)
    
    # Very wide initial box
    X0 = Box([1.0, -0.5], [3.0, 1.5])  # Wide: 2m × 2m/s
    unsafe = Box([4.5, -0.3], [5.5, 0.3])
    
    print("\nConfiguration:")
    print(f"  Initial: pos ∈ [1.0, 3.0], vel ∈ [-0.5, 1.5] (WIDE)")
    print(f"  Width: 2.0m × 2.0m/s")
    print(f"  Unsafe:  pos ∈ [4.5, 5.5], vel ∈ [-0.3, 0.3]")
    print(f"  Strategy: Wide box → large overapproximation → overlap")
    
    result = verify_fbra(
        X0, double_integrator_controller, double_integrator,
        unsafe, T=3, max_iterations=10, verbose=True
    )
    
    print("\n" + "="*70)
    print(f"RESULT: {result.status}")
    print("="*70)
    
    return result


def scenario_unknown_with_pd():
    """
    PD controller with carefully chosen parameters
    """
    
    print("\n" + "="*70)
    print("SCENARIO: PD Controller Boundary Case")
    print("="*70)
    
    pd = DoubleIntegratorPDController(Kp=0.3, Kd=0.5)  # Weak control
    
    X0 = Box([3.0, 0.4], [3.5, 0.8])  # Moving toward unsafe
    unsafe = Box([4.5, -0.4], [5.5, 0.4])
    
    print("\nConfiguration:")
    print(f"  Initial: pos ∈ [3.0, 3.5], vel ∈ [0.4, 0.8] (moving right)")
    print(f"  Unsafe:  pos ∈ [4.5, 5.5], vel ∈ [-0.4, 0.4]")
    print(f"  Controller: Weak PD (may not brake fast enough)")
    
    result = verify_fbra(
        X0, pd, double_integrator,
        unsafe, T=4, max_iterations=10, verbose=True
    )
    
    print("\n" + "="*70)
    print(f"RESULT: {result.status}")
    print("="*70)
    
    return result


def main():
    """Test all scenarios designed to trigger Unknown"""
    
    print("\n" + "❓"*35)
    print("TESTING FOR UNKNOWN STATUS")
    print("❓"*35)
    
    print("\nGoal: Create scenarios that trigger 'Unknown' status")
    print("This demonstrates FBRA's refinement capabilities\n")
    
    results = {}
    
    print("\n" + "🔵"*35)
    results['grazing'] = scenario_unknown_grazing()
    
    print("\n" + "🟡"*35)
    results['large_box'] = scenario_unknown_large_box()
    
    print("\n" + "🟠"*35)
    results['pd_boundary'] = scenario_unknown_with_pd()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Unknown Status Tests")
    print("="*70)
    
    for name, result in results.items():
        status_symbol = "?" if result.status == "Unknown" else ("✓" if result.status == "Safe" else "✗")
        print(f"  {name:<20}: {status_symbol} {result.status} (iter={result.iterations})")
    
    unknown_count = sum(1 for r in results.values() if r.status == "Unknown")
    
    print("\n" + "="*70)
    
    if unknown_count > 0:
        print(f"✓ Successfully triggered Unknown in {unknown_count}/3 scenarios!")
        print("  This proves FBRA can handle uncertain cases")
    else:
        print("→ All scenarios resolved without Unknown")
        print("  Possible reasons:")
        print("    1. Boxes cleanly avoid or hit unsafe (no grazing)")
        print("    2. Backward analysis successfully pruned all overlaps")
        print("    3. Need more extreme configurations to force Unknown")
    
    print("="*70)


if __name__ == "__main__":
    main()