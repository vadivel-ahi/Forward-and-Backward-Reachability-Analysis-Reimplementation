"""
Detailed diagnostic for buggy controller
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from fbra.boxes import Box
from fbra.forward import forward_reach_one_step
from experiments.controller import BuggyGroundRobotController
from experiments.dynamics import ground_robot
from experiments.sets import X0_ground_robot, Unsafe_ground_robot

print("\n" + "="*70)
print("DETAILED BUGGY CONTROLLER DIAGNOSTIC")
print("="*70)

controller = BuggyGroundRobotController()

# Manual forward propagation
R = {0: [X0_ground_robot]}

for t in range(3):  # Just first 3 steps
    print(f"\n{'─'*70}")
    print(f"Timestep {t} → {t+1}")
    print(f"{'─'*70}")
    
    R[t+1] = forward_reach_one_step(R[t], controller, ground_robot)
    
    for i, box in enumerate(R[t+1]):
        print(f"\nBox {i+1}:")
        print(f"  Lower: {box.low}")
        print(f"  Upper: {box.up}")
        
        # Detailed intersection check
        print(f"\nUnsafe region:")
        print(f"  Lower: {Unsafe_ground_robot.low}")
        print(f"  Upper: {Unsafe_ground_robot.up}")
        
        # Check intersection manually
        intersects = box.intersects(Unsafe_ground_robot)
        print(f"\nIntersection check:")
        print(f"  box.intersects(unsafe): {intersects}")
        
        if intersects:
            intersection = box.intersect(Unsafe_ground_robot)
            print(f"  Intersection box: {intersection.low} to {intersection.up}")
            
            # Check containment
            unsafe_contains_box = Unsafe_ground_robot.contains(box)
            box_contains_unsafe = box.contains(Unsafe_ground_robot)
            
            print(f"  Unsafe contains box: {unsafe_contains_box}")
            print(f"  Box contains unsafe: {box_contains_unsafe}")
            
            # What should the status be?
            if unsafe_contains_box:
                expected = "Unsafe"
            elif box_contains_unsafe:
                expected = "Unknown (box contains unsafe)"
            else:
                expected = "Unknown (partial overlap)"
            
            print(f"  Expected status: {expected}")