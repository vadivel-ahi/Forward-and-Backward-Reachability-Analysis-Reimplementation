"""
Debug Reachability - Visualize What's Actually Happening
=========================================================
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from fbra.boxes import Box
from fbra.forward import forward_reach
from experiments.controller import ground_robot_controller, BuggyGroundRobotController
from experiments.dynamics import ground_robot
from experiments.sets import X0_ground_robot, Unsafe_ground_robot


def print_box_stats(box, name="Box"):
    """Print detailed box statistics"""
    print(f"\n{name}:")
    print(f"  Lower: {box.low}")
    print(f"  Upper: {box.up}")
    print(f"  Width: {box.width()}")
    print(f"  Volume: {np.prod(box.width()):.6f}")
    print(f"  Center: {(box.low + box.up) / 2}")


def visualize_reachability(controller_name, controller, T=9):
    """Visualize forward reachability step by step"""
    
    print("\n" + "="*70)
    print(f"REACHABILITY ANALYSIS: {controller_name}")
    print("="*70)
    
    print_box_stats(X0_ground_robot, "Initial Set")
    print_box_stats(Unsafe_ground_robot, "Unsafe Region")
    
    # Manual forward propagation with detailed output
    from fbra.nn_bounds import nn_forward_box
    
    current_boxes = [X0_ground_robot]
    
    for t in range(T):
        print(f"\n{'-'*70}")
        print(f"Timestep {t} → {t+1}")
        print(f"{'-'*70}")
        
        next_boxes = []
        
        for i, x_box in enumerate(current_boxes):
            print(f"\n  Box {i+1}/{len(current_boxes)}:")
            print_box_stats(x_box, "    State")
            
            # Compute control bounds
            u_box = nn_forward_box(x_box, controller)
            print_box_stats(u_box, "    Control")
            
            # Propagate dynamics
            next_box = ground_robot(x_box, u_box)
            print_box_stats(next_box, "    Next State")
            
            # Check intersection
            intersects = next_box.intersects(Unsafe_ground_robot)
            contained = Unsafe_ground_robot.contains(next_box)
            contains = next_box.contains(Unsafe_ground_robot)
            
            print(f"\n    Safety Check:")
            print(f"      Intersects unsafe? {intersects}")
            print(f"      Contained in unsafe? {contained}")
            print(f"      Contains unsafe? {contains}")
            
            if intersects:
                intersection = next_box.intersect(Unsafe_ground_robot)
                if intersection:
                    print(f"      Intersection volume: {np.prod(intersection.width()):.6f}")
            
            next_boxes.append(next_box)
        
        current_boxes = next_boxes
    
    print("\n" + "="*70)


def main():
    """Run diagnostics on both controllers"""
    
    print("\n" + "🔵"*35)
    visualize_reachability("SAFE Controller", ground_robot_controller, T=9)
    
    print("\n" + "🔴"*35)
    buggy = BuggyGroundRobotController()
    visualize_reachability("BUGGY Controller", buggy, T=9)


if __name__ == "__main__":
    main()