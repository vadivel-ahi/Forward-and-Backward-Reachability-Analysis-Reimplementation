import numpy as np

class GroundRobot:
    """
    2D Ground Robot System
    State: x = [px, py]^T (position in x-y plane)
    Control: u = [vx, vy]^T (velocity)
    Dynamics: x(t+1) = A*x(t) + B*u(t)
    """
    
    def __init__(self, dt=1.0):
        self.dt = dt
        self.nx = 2  # state dimension
        self.nu = 2  # control dimension
        
        # System matrices
        self.A = np.array([[1, 0],
                          [0, 1]])
        
        self.B = np.array([[1, 0],
                          [0, 1]])
    
    def step(self, x, u):
        """Single step dynamics"""
        return self.A @ x + self.B @ u
    
    def get_initial_set(self):
        """Return initial state bounds"""
        # [-5.5, -4.5] × [-0.5, 0.5]
        return np.array([[-5.5, -4.5],
                        [-0.5, 0.5]])
    
    def get_unsafe_set(self):
        """Return unsafe region bounds"""
        # [-1, 1] × [-1, 1]
        return np.array([[-1, 1],
                        [-1, 1]])