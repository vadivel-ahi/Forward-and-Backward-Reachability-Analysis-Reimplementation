# experiments/dynamics.py

import numpy as np
from fbra.boxes import Box


# ----------------------------------------------------
# Ground Robot dynamics (2D)
# state x = [px, py]
# control u = [vx_cmd, vy_cmd]
# simple integrator: p' = p + dt * u
# ----------------------------------------------------
def ground_robot(box: Box, u_box: Box, dt: float = 1.0) -> Box:
    xl, xu = box.low, box.up
    ul, uu = u_box.low, u_box.up

    # px' = px + dt * vx_cmd
    px_low  = xl[0] + dt * ul[0]
    px_high = xu[0] + dt * uu[0]

    # py' = py + dt * vy_cmd
    py_low  = xl[1] + dt * ul[1]
    py_high = xu[1] + dt * uu[1]

    return Box([px_low, py_low], [px_high, py_high])


# ----------------------------------------------------
# Double Integrator dynamics (2D state, 1D control)
# x = [pos, vel]
# pos' = pos + dt * vel
# vel' = vel + dt * u
# ----------------------------------------------------
def double_integrator(box: Box, u_box: Box, dt: float = 0.1) -> Box:
    xl, xu = box.low, box.up
    ul, uu = u_box.low, u_box.up  # scalar control in a 1D box

    # pos' bounds
    pos_low  = xl[0] + dt * xl[1]
    pos_high = xu[0] + dt * xu[1]

    # vel' bounds
    vel_low  = xl[1] + dt * ul[0]
    vel_high = xu[1] + dt * uu[0]

    return Box([pos_low, vel_low], [pos_high, vel_high])


# ----------------------------------------------------
# Quadrotor simplified linearized hover dynamics (6D)
# x = [px, py, pz, vx, vy, vz]
# u = [ax, ay, az]
# p' = p + dt * v
# v' = v + dt * u
# ----------------------------------------------------
def quadrotor_dynamics(box: Box, u_box: Box, dt: float = 0.02) -> Box:
    xl, xu = box.low, box.up       # state low/high
    ul, uu = u_box.low, u_box.up   # control low/high (3D)

    # Positions
    p_low  = xl[0:3] + dt * xl[3:6]
    p_high = xu[0:3] + dt * xu[3:6]

    # Velocities
    v_low  = xl[3:6] + dt * ul[0:3]
    v_high = xu[3:6] + dt * uu[0:3]

    next_low  = np.concatenate([p_low, v_low])
    next_high = np.concatenate([p_high, v_high])

    return Box(next_low, next_high)
