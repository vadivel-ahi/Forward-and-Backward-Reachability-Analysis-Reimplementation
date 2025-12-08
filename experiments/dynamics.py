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
def double_integrator(x_box: Box, u_box: Box, dt=1.0):
    """
    Double integrator dynamics using matrix interval multiplication
    
    x' = A·x + B·u where:
      A = [[1, dt],    B = [[0.5·dt²],
           [0,  1]]         [dt    ]]
    """
    xl, xu = x_box.low, x_box.up
    ul, uu = u_box.low, u_box.up
    
    # Define system matrices
    A = np.array([[1, dt],
                  [0, 1]])
    
    B = np.array([[0.5 * dt**2],
                  [dt]])
    
    # Interval matrix multiplication: [A] * [x]
    # For row i: sum_j A[i,j] * x_interval[j]
    
    # Position row: [1, dt] * [pos, vel]
    # = 1·pos + dt·vel
    pos_from_A_l = xl[0] + dt * xl[1]  # Minimum
    pos_from_A_u = xu[0] + dt * xu[1]  # Maximum
    
    # Velocity row: [0, 1] * [pos, vel]
    # = 0·pos + 1·vel = vel
    vel_from_A_l = xl[1]
    vel_from_A_u = xu[1]
    
    # Interval matrix multiplication: [B] * [u]
    # Position: 0.5·dt²·accel
    pos_from_B_l = 0.5 * dt**2 * ul[0]
    pos_from_B_u = 0.5 * dt**2 * uu[0]
    
    # Velocity: dt·accel
    vel_from_B_l = dt * ul[0]
    vel_from_B_u = dt * uu[0]
    
    # Combine: x' = A·x + B·u
    pos_l = pos_from_A_l + pos_from_B_l
    pos_u = pos_from_A_u + pos_from_B_u
    
    vel_l = vel_from_A_l + vel_from_B_l
    vel_u = vel_from_A_u + vel_from_B_u
    
    return Box([pos_l, vel_l], [pos_u, vel_u])


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
