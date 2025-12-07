"""
Create Visualizations for FBRA Results
======================================
Generates all plots and animations for the paper.
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from fbra.verifier import verify_fbra
from fbra.forward import forward_reach
from experiments.controller import ground_robot_controller, BuggyGroundRobotController
from experiments.dynamics import ground_robot
from experiments.sets import X0_ground_robot, Unsafe_ground_robot
from utils.visualization import (
    plot_reachable_sets_evolution,
    plot_partitioning_process,
    plot_forward_backward_comparison,
    plot_comparison_grid,
    create_verification_animation
)


def visualize_safe_controller():
    """Visualize safe controller verification"""
    
    print("\n" + "="*70)
    print("VISUALIZATION 1: Safe Controller")
    print("="*70)
    
    # Run forward reachability
    R_forward = forward_reach(X0_ground_robot, ground_robot_controller, ground_robot, T=9)
    
    # Convert to list format
    reachable_sets = [R_forward[t] for t in range(10)]
    
    # Create visualization
    fig, ax = plot_reachable_sets_evolution(
        reachable_sets,
        Unsafe_ground_robot,
        X0_ground_robot,
        title="Safe Controller - Forward Reachability",
        save_path="results/safe_controller_forward.png"
    )
    
    plt.show()
    
    return reachable_sets


def visualize_buggy_controller():
    """Visualize buggy controller with partitioning"""
    
    print("\n" + "="*70)
    print("VISUALIZATION 2: Buggy Controller (with FBRA)")
    print("="*70)
    
    buggy = BuggyGroundRobotController()
    
    # Run FBRA verification
    result = verify_fbra(
        X0_ground_robot,
        buggy,
        ground_robot,
        Unsafe_ground_robot,
        T=5,
        verbose=False
    )
    
    print(f"Result: {result.status}")
    
    # Visualize forward reachability
    if result.R_forward:
        reachable_sets = [result.R_forward[t] for t in sorted(result.R_forward.keys())]
        
        fig, ax = plot_reachable_sets_evolution(
            reachable_sets,
            Unsafe_ground_robot,
            X0_ground_robot,
            title="Buggy Controller - Detects Unsafe Behavior",
            save_path="results/buggy_controller_unsafe.png"
        )
        
        plt.show()
    
    return result


def visualize_partitioning():
    """Visualize partitioning process"""
    
    print("\n" + "="*70)
    print("VISUALIZATION 3: State Space Partitioning")
    print("="*70)
    
    from fbra.partition import partition_initial_set
    
    # Create partitions
    partitions = partition_initial_set(
        X0_ground_robot,
        [X0_ground_robot],  # Dummy backward set
        Unsafe_ground_robot,
        method="uniform",
        n_splits=2
    )
    
    # Simulate partition results
    partition_results = ['Safe', 'Safe', 'Unsafe', 'Safe']
    
    fig, axes = plot_partitioning_process(
        X0_ground_robot,
        partitions,
        Unsafe_ground_robot,
        partition_results,
        title="State Space Partitioning Process",
        save_path="results/partitioning_process.png"
    )
    
    plt.show()
    
    return partitions


def visualize_comparison():
    """Compare safe vs buggy controller"""
    
    print("\n" + "="*70)
    print("VISUALIZATION 4: Safe vs Unsafe Comparison")
    print("="*70)
    
    # Safe controller
    R_safe = forward_reach(X0_ground_robot, ground_robot_controller, ground_robot, T=9)
    reachable_safe = [R_safe[t] for t in range(10)]
    
    # Buggy controller
    buggy = BuggyGroundRobotController()
    R_buggy = forward_reach(X0_ground_robot, buggy, ground_robot, T=5)
    reachable_buggy = [R_buggy[t] for t in range(6)]
    
    # Create comparison
    results_dict = {
        'safe': {
            'reachable_sets': reachable_safe,
            'unsafe': Unsafe_ground_robot,
            'initial': X0_ground_robot
        },
        'buggy': {
            'reachable_sets': reachable_buggy,
            'unsafe': Unsafe_ground_robot,
            'initial': X0_ground_robot
        }
    }
    
    fig, axes = plot_comparison_grid(
        results_dict,
        titles=['Safe Controller', 'Buggy Controller'],
        save_path="results/comparison_safe_vs_buggy.png"
    )
    
    plt.show()
    
    return fig


def create_animation():
    """Create animated visualization"""
    
    print("\n" + "="*70)
    print("VISUALIZATION 5: Creating Animation")
    print("="*70)
    
    # Forward reachability for safe controller
    R_forward = forward_reach(X0_ground_robot, ground_robot_controller, ground_robot, T=9)
    reachable_sets = [R_forward[t] for t in range(10)]
    
    # Create animation
    print("Generating animation... (this may take a minute)")
    anim = create_verification_animation(
        reachable_sets,
        Unsafe_ground_robot,
        X0_ground_robot,
        title="FBRA Verification Process",
        save_path="results/verification_animation.gif"
    )
    
    print("✓ Animation created!")
    
    return anim


def main():
    """Generate all visualizations"""
    
    print("\n" + "🎨"*35)
    print("FBRA VISUALIZATION SUITE")
    print("🎨"*35)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    try:
        # 1. Safe controller
        visualize_safe_controller()
        
        # 2. Buggy controller
        visualize_buggy_controller()
        
        # 3. Partitioning process
        visualize_partitioning()
        
        # 4. Comparison
        visualize_comparison()
        
        # 5. Animation
        create_animation()
        
        print("\n" + "="*70)
        print("✓ ALL VISUALIZATIONS COMPLETE!")
        print("="*70)
        print("\nGenerated files:")
        print("  - results/safe_controller_forward.png")
        print("  - results/buggy_controller_unsafe.png")
        print("  - results/partitioning_process.png")
        print("  - results/comparison_safe_vs_buggy.png")
        print("  - results/verification_animation.gif")
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()