"""
Neural Network Controllers for FBRA Benchmarks
==============================================
Implements controllers for:
    1. Ground Robot (Safe & Buggy versions)
    2. Double Integrator
    3. Quadrotor
"""

import torch
import torch.nn as nn


# ----------------------------------------------------
# Ground Robot SAFE controller (2D → 2D)
# ----------------------------------------------------
class GroundRobotController(nn.Module):
    """
    Safe controller for ground robot
    Trained to avoid obstacles
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2)   # control = (ux, uy)
        )

        # Small weights for stability
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.05, 0.05)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        return self.net(x)


# Default instance
ground_robot_controller = GroundRobotController()


# ----------------------------------------------------
# Ground Robot BUGGY controller (2D → 2D)
# Same architecture but with strong bias toward unsafe
# ----------------------------------------------------
class BuggyGroundRobotController(nn.Module):
    """
    Buggy controller that pushes robot toward unsafe region
    
    Starts at x ≈ -5 (left), unsafe at x ≈ 0 (center)
    Adds positive x-bias to push right toward obstacle
    """
    def __init__(self):
        super().__init__()
        
        # Same architecture as safe controller
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        
        # Initialize with small weights
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.05, 0.05)
                nn.init.zeros_(m.bias)
        
        # BUG: Add strong bias to final layer
        # This overrides the learned safe behavior
        with torch.no_grad():
            # Push RIGHT toward obstacle (x-direction)
            self.net[-1].bias[0] += 2.0  # Strong positive x-bias
            self.net[-1].bias[1] += 0.0  # No y-bias
    
    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        return self.net(x)


# ----------------------------------------------------
# Double Integrator SAFE controller (2D → 1D)
# state = [pos, vel], control = [accel]
# ----------------------------------------------------
class DoubleIntegratorController(nn.Module):
    """Safe controller for double integrator"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)   # scalar control
        )

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.05, 0.05)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        return self.net(x)


double_integrator_controller = DoubleIntegratorController()


# ----------------------------------------------------
# Quadrotor controller (6D → 3D)
# state = [px, py, pz, vx, vy, vz]
# control = [ax, ay, az]
# ----------------------------------------------------
class QuadrotorController(nn.Module):
    """Safe controller for quadrotor"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3)   # accelerations
        )

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.05, 0.05)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        return self.net(x)


quadrotor_controller = QuadrotorController()