"""
Trace complete verification flow for buggy controller
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from fbra.verifier import verify_fbra
from experiments.controller import BuggyGroundRobotController
from experiments.dynamics import ground_robot
from experiments.sets import X0_ground_robot, Unsafe_ground_robot

print("\n" + "="*70)
print("FULL VERIFICATION FLOW - BUGGY CONTROLLER")
print("="*70)

controller = BuggyGroundRobotController()

# Run with verbose=True and max_iterations=3 to see what happens
result = verify_fbra(
    X0=X0_ground_robot,
    model=controller,
    plant=ground_robot,
    unsafe=Unsafe_ground_robot,
    T=5,  # Just 5 timesteps to see the issue faster
    max_iterations=3,  # Only 3 iterations
    verbose=True
)

print("\n" + "="*70)
print(f"FINAL RESULT: {result.status}")
print(f"Iterations: {result.iterations}")
print("="*70)