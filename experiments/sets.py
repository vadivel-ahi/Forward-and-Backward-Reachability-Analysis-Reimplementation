# experiments/sets.py

from fbra.boxes import Box

# ----------------------------------------------------
# Ground Robot (safe & buggy share X0 / Unsafe)
# ----------------------------------------------------
X0_ground_robot = Box(
    low=[-5.5, -0.5],
    up=[-4.5,  0.5]
)

Unsafe_ground_robot = Box(
    low=[-1.0, -1.0],
    up=[ 1.0,  1.0]
)


# ----------------------------------------------------
# Double Integrator benchmark
# From typical settings: x0 in [-4, -2.4] × [-0.5, 0.5]
# Unsafe region near target: [4.5, 5] × [-0.25, 0.25]
# ----------------------------------------------------
X0_double_integrator = Box(
    low=[-4.0, -0.5],
    up=[-2.4,  0.5]
)

Unsafe_double_integrator = Box(
    low=[4.5, -0.25],
    up=[5.0,  0.25]
)


# ----------------------------------------------------
# Quadrotor benchmark (simplified)
# Hovering near (0,0,1) with small velocities
# Unsafe if far away in x,y and low in z
# ----------------------------------------------------
Quad_X0 = Box(
    low=[-0.1, -0.1, 0.9, -0.05, -0.05, -0.05],
    up=[ 0.1,  0.1, 1.1,  0.05,  0.05,  0.05]
)

Quad_Unsafe = Box(
    low=[2.0, 2.0, 0.0, -10.0, -10.0, -10.0],
    up=[5.0, 5.0, 2.0,  10.0,  10.0,  10.0]
)
