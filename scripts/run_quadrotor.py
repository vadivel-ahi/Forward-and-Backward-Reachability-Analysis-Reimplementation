# scripts/run_quadrotor.py

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import time
from fbra.verifier import verify
from experiments.controller import quadrotor_controller
from experiments.dynamics import quadrotor_dynamics
from experiments.sets import Quad_X0, Quad_Unsafe

print("\nRunning FBRA on Quadrotor (simplified 6D model)...\n")

T = 30  # small horizon; higher values will cause big over-approx

start = time.time()
result = verify(Quad_X0, quadrotor_controller, quadrotor_dynamics, Quad_Unsafe, T)
end = time.time()

print("Result:", result)
print("Time:", round(end - start, 3), "seconds\n")
