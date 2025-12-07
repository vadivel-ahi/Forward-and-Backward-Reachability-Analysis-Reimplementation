# fbra/plant.py
from fbra.boxes import Box

def double_integrator_forward(x_box: Box, u_box: Box, dt=0.1):
    xl, xu = x_box.low, x_box.up
    ul, uu = u_box.low, u_box.up

    pos_l = xl[0] + xl[1] * dt
    pos_u = xu[0] + xu[1] * dt

    vel_l = xl[1] + ul[0] * dt
    vel_u = xu[1] + uu[0] * dt

    return Box([pos_l, vel_l], [pos_u, vel_u])
