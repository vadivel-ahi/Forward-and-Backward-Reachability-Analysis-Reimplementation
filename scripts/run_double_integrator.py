# scripts/run_double_integrator.py

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import time
from fbra.verifier import verify
from experiments.controller import double_integrator_controller
from experiments.dynamics import double_integrator
from experiments.sets import X0_double_integrator, Unsafe_double_integrator

print("\nRunning FBRA on Double Integrator...\n")

T = 5  # typical setting

start = time.time()
result = verify(X0_double_integrator, double_integrator_controller, double_integrator, Unsafe_double_integrator, T)
end = time.time()

print("Result:", result)
print("Time:", round(end - start, 3), "seconds\n")
