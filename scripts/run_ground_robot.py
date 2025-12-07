"""
Test FBRA on Ground Robot Benchmark
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import time
from fbra.verifier import verify_fbra
from experiments.controller import ground_robot_controller
from experiments.dynamics import ground_robot
from experiments.sets import X0_ground_robot, Unsafe_ground_robot

print("\n" + "="*70)
print("FBRA VERIFICATION - Ground Robot Benchmark")
print("="*70)

T = 9  # Time horizon from paper

start_time = time.time()

result = verify_fbra(
    X0=X0_ground_robot,
    model=ground_robot_controller,
    plant=ground_robot,
    unsafe=Unsafe_ground_robot,
    T=T,
    max_iterations=10,
    verbose=True
)

end_time = time.time()

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Status: {result.status}")
print(f"Time: {end_time - start_time:.3f} seconds")
print(f"Iterations: {result.iterations}")
print("="*70 + "\n")