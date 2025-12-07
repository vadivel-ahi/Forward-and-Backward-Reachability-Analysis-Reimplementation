import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fbra.verifier import verify
from experiments.controller import ground_robot_controller
from experiments.dynamics import ground_robot
from experiments.sets import X0_ground_robot, Unsafe_ground_robot
from utils.visualization import plot_three_stages

print("\nGenerating visualization for Ground Robot...\n")

T = 9

result, R_f_initial, R_f_refined, R_b = verify(
    X0_ground_robot,
    ground_robot_controller,
    ground_robot,
    Unsafe_ground_robot,
    T,
    return_all=True
)

print("Verification result:", result)

plot_three_stages(R_f_initial, R_f_refined, R_b, Unsafe_ground_robot)
