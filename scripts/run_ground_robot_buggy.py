# scripts/run_ground_robot_buggy.py

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import time
from fbra.verifier import verify
from experiments.controller import BuggyGroundRobotController
from experiments.dynamics import ground_robot
from experiments.sets import X0_ground_robot, Unsafe_ground_robot

print("\nRunning FBRA on Ground Robot (BUGGY controller)...\n")

buggy = BuggyGroundRobotController()
T = 6  # shorter horizon, like in the paper buggy case

start = time.time()
result = verify(X0_ground_robot, buggy, ground_robot, Unsafe_ground_robot, T)
end = time.time()

print("Result:", result)
print("Time:", round(end - start, 3), "seconds\n")
