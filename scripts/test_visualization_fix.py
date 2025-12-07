"""
Test the fixed visualization module
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from fbra.boxes import Box
from fbra.forward import forward_reach
from experiments.controller import ground_robot_controller
from experiments.dynamics import ground_robot
from experiments.sets import X0_ground_robot, Unsafe_ground_robot
from utils.visualization import (
    plot_reachable_sets_evolution,
    plot_initial_and_unsafe
)

print("="*70)
print("TESTING FIXED VISUALIZATION")
print("="*70)

# Test 1: Simple initial + unsafe plot
print("\nTest 1: Plotting initial and unsafe sets...")
fig, ax = plot_initial_and_unsafe(
    X0_ground_robot,
    Unsafe_ground_robot,
    save_path="results/test_initial_unsafe.png"
)
plt.show()

# Test 2: Forward reachability
print("\nTest 2: Forward reachability visualization...")
R_forward = forward_reach(X0_ground_robot, ground_robot_controller, ground_robot, T=9)
reachable_sets = [R_forward[t] for t in range(10)]

fig, ax = plot_reachable_sets_evolution(
    reachable_sets,
    Unsafe_ground_robot,
    X0_ground_robot,
    title="Safe Controller - Forward Reachability (FIXED)",
    save_path="results/test_forward_reach_fixed.png"
)
plt.show()

print("\n" + "="*70)
print("✓ Tests complete! Check the results/ folder for output.")
print("="*70)