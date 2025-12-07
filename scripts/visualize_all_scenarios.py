"""
Comprehensive Visualization Suite
==================================
Creates all publication-quality plots for your project.
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from fbra.boxes import Box
from fbra.forward import forward_reach
from fbra.verifier import verify_fbra
from fbra.partition import partition_initial_set
from experiments.controller import ground_robot_controller, BuggyGroundRobotController
from experiments.dynamics import ground_robot
from experiments.sets import X0_ground_robot, Unsafe_ground_robot
from utils.visualization import plot_box_on_axis, set_axis_limits


def plot_forward_reach_detailed(reachable_sets, unsafe_set, initial_set,
                                title="Forward Reachability - Detailed View",
                                save_path=None):
    """
    Enhanced visualization showing ALL timesteps clearly
    Uses offset visualization to show overlapping boxes
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # ====================================
    # LEFT: Standard view
    # ====================================
    ax1.set_title('Standard View', fontsize=13, fontweight='bold')
    
    # Unsafe region
    plot_box_on_axis(ax1, unsafe_set,
                    facecolor='#ff6b6b', edgecolor='#c92a2a',
                    alpha=0.4, linewidth=2.5, label='Unsafe', zorder=1)
    
    # Initial set
    plot_box_on_axis(ax1, initial_set,
                    facecolor='#4dabf7', edgecolor='#1971c2',
                    alpha=0.7, linewidth=2.5, label='Initial (t=0)', zorder=10)
    
    # All reachable sets
    n_steps = len(reachable_sets)
    colors = plt.cm.viridis(np.linspace(0, 1, n_steps))
    
    for t, boxes in enumerate(reachable_sets[1:], 1):
        for box in boxes:
            plot_box_on_axis(ax1, box,
                           facecolor='none',
                           edgecolor=colors[t],
                           alpha=0.9,
                           linewidth=2,
                           label=f't={t}' if t in [1, n_steps-1] else '',
                           zorder=2+t)
    
    # Collect all boxes for limits
    all_boxes = [unsafe_set, initial_set]
    for boxes in reachable_sets:
        all_boxes.extend(boxes)
    
    set_axis_limits(ax1, all_boxes, padding=0.15)
    ax1.set_xlabel('$x_1$', fontsize=12)
    ax1.set_ylabel('$x_2$', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # ====================================
    # RIGHT: Exploded view (shows growth)
    # ====================================
    ax2.set_title('Box Sizes Over Time (Exploded View)', fontsize=13, fontweight='bold')
    
    # Plot box widths as bars
    timesteps = list(range(n_steps))
    widths_x = []
    widths_y = []
    
    for t, boxes in enumerate(reachable_sets):
        if boxes:
            box = boxes[0]
            if isinstance(box, Box):
                width = box.up - box.low
                widths_x.append(width[0])
                widths_y.append(width[1])
            else:
                widths_x.append(0)
                widths_y.append(0)
    
    x_pos = np.arange(n_steps)
    width_bar = 0.35
    
    bars1 = ax2.bar(x_pos - width_bar/2, widths_x, width_bar, 
                    label='Width in $x_1$', alpha=0.8, color='#4dabf7')
    bars2 = ax2.bar(x_pos + width_bar/2, widths_y, width_bar,
                    label='Width in $x_2$', alpha=0.8, color='#ff6b6b')
    
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Box Width', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f't={t}' for t in timesteps])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add values on top of bars
    for i, (w_x, w_y) in enumerate(zip(widths_x, widths_y)):
        ax2.text(i - width_bar/2, w_x, f'{w_x:.3f}', 
                ha='center', va='bottom', fontsize=8)
        ax2.text(i + width_bar/2, w_y, f'{w_y:.3f}',
                ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(title, fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig, (ax1, ax2)


def plot_buggy_vs_safe_comparison(save_path=None):
    """
    Side-by-side comparison of safe vs buggy controller
    Shows the crash trajectory
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # ====================================
    # LEFT: Safe Controller
    # ====================================
    print("\nComputing safe controller reachability...")
    R_safe = forward_reach(X0_ground_robot, ground_robot_controller, ground_robot, T=9)
    safe_sets = [R_safe[t] for t in range(10)]
    
    ax1.set_title('Safe Controller - Avoids Obstacle', fontsize=14, fontweight='bold', color='green')
    
    plot_box_on_axis(ax1, Unsafe_ground_robot,
                    facecolor='#ff6b6b', edgecolor='#c92a2a',
                    alpha=0.4, linewidth=2.5, label='Unsafe')
    
    plot_box_on_axis(ax1, X0_ground_robot,
                    facecolor='#4dabf7', edgecolor='#1971c2',
                    alpha=0.7, linewidth=2.5, label='Initial')
    
    colors = plt.cm.winter(np.linspace(0, 1, 10))
    for t, boxes in enumerate(safe_sets[1:], 1):
        for box in boxes:
            plot_box_on_axis(ax1, box,
                           facecolor='none', edgecolor=colors[t],
                           alpha=0.9, linewidth=1.8)
    
    all_boxes_safe = [Unsafe_ground_robot, X0_ground_robot] + [b for boxes in safe_sets for b in boxes]
    set_axis_limits(ax1, all_boxes_safe, padding=0.15)
    
    ax1.set_xlabel('Position $x_1$', fontsize=12)
    ax1.set_ylabel('Position $x_2$', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # ====================================
    # RIGHT: Buggy Controller
    # ====================================
    print("Computing buggy controller reachability...")
    buggy = BuggyGroundRobotController()
    R_buggy = forward_reach(X0_ground_robot, buggy, ground_robot, T=5)
    buggy_sets = [R_buggy[t] for t in range(6)]
    
    ax2.set_title('Buggy Controller - Crashes Into Obstacle', fontsize=14, fontweight='bold', color='red')
    
    plot_box_on_axis(ax2, Unsafe_ground_robot,
                    facecolor='#ff6b6b', edgecolor='#c92a2a',
                    alpha=0.4, linewidth=2.5, label='Unsafe')
    
    plot_box_on_axis(ax2, X0_ground_robot,
                    facecolor='#4dabf7', edgecolor='#1971c2',
                    alpha=0.7, linewidth=2.5, label='Initial')
    
    colors_buggy = plt.cm.autumn(np.linspace(0, 1, 6))
    
    for t, boxes in enumerate(buggy_sets[1:], 1):
        for box in boxes:
            # Check if intersects unsafe
            intersects = box.intersects(Unsafe_ground_robot)
            
            plot_box_on_axis(ax2, box,
                           facecolor='none',
                           edgecolor=colors_buggy[t],
                           alpha=0.9,
                           linewidth=2.5 if intersects else 1.8,
                           linestyle='-' if not intersects else '--',
                           label=f't={t}' if t == 2 else '')
            
            # Mark intersection
            if intersects:
                center = (box.low + box.up) / 2
                ax2.scatter(center[0], center[1], 
                          s=200, c='red', marker='X', 
                          edgecolors='darkred', linewidths=2,
                          label='Crash!' if t == 2 else '', zorder=10)
    
    all_boxes_buggy = [Unsafe_ground_robot, X0_ground_robot] + [b for boxes in buggy_sets for b in boxes]
    set_axis_limits(ax2, all_boxes_buggy, padding=0.15)
    
    ax2.set_xlabel('Position $x_1$', fontsize=12)
    ax2.set_ylabel('Position $x_2$', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.suptitle('FBRA: Safe vs Unsafe Controller Comparison', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    
    return fig, (ax1, ax2)


def plot_partitioning_detailed(title="FBRA Partitioning Process",save_path=None):
    """
    Show the partitioning process in detail
    Demonstrates how FBRA refines Unknown cases
    """
    
    print("\nCreating partitioning visualization...")
    
    fig = plt.figure(figsize=(18, 6))
    
    # Create 3 subplots
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)
    
    # ====================================
    # STEP 1: Original box
    # ====================================
    ax1.set_title('Step 1: Original Initial Set', fontsize=12, fontweight='bold')
    
    plot_box_on_axis(ax1, Unsafe_ground_robot,
                    facecolor='#ff6b6b', alpha=0.4, linewidth=2, label='Unsafe')
    plot_box_on_axis(ax1, X0_ground_robot,
                    facecolor='#4dabf7', alpha=0.6, linewidth=2.5, label='Initial')
    
    all_boxes = [Unsafe_ground_robot, X0_ground_robot]
    set_axis_limits(ax1, all_boxes, padding=0.2)
    ax1.set_xlabel('$x_1$', fontsize=11)
    ax1.set_ylabel('$x_2$', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # ====================================
    # STEP 2: First partition (2x2 = 4 boxes)
    # ====================================
    ax2.set_title('Step 2: First Partition (4 regions)', fontsize=12, fontweight='bold')
    
    plot_box_on_axis(ax2, Unsafe_ground_robot,
                    facecolor='#ff6b6b', alpha=0.4, linewidth=2)
    
    partitions_1 = partition_initial_set(X0_ground_robot, [X0_ground_robot], 
                                        Unsafe_ground_robot, "uniform", 2)
    
    partition_colors = ['#51cf66', '#51cf66', '#ffd43b', '#51cf66']  # Assume 1 unknown
    
    for i, (part, color) in enumerate(zip(partitions_1, partition_colors)):
        plot_box_on_axis(ax2, part,
                        facecolor=color, edgecolor='black',
                        alpha=0.5, linewidth=1.5)
        
        center = (part.low + part.up) / 2
        ax2.text(center[0], center[1], f'{i+1}',
                fontsize=12, ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', 
                         edgecolor='black', linewidth=1.5))
    
    all_boxes.extend(partitions_1)
    set_axis_limits(ax2, all_boxes, padding=0.2)
    ax2.set_xlabel('$x_1$', fontsize=11)
    ax2.set_ylabel('$x_2$', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # ====================================
    # STEP 3: Second partition (further refined)
    # ====================================
    ax3.set_title('Step 3: Recursive Partition (16 regions)', fontsize=12, fontweight='bold')
    
    plot_box_on_axis(ax3, Unsafe_ground_robot,
                    facecolor='#ff6b6b', alpha=0.4, linewidth=2)
    
    # Partition the "unknown" region further
    if len(partitions_1) >= 3:
        unknown_partition = partitions_1[2]  # Assume partition 3 was unknown
        sub_partitions = partition_initial_set(unknown_partition, [unknown_partition],
                                              Unsafe_ground_robot, "uniform", 2)
        
        # Plot safe partitions
        for i in [0, 1, 3]:
            if i < len(partitions_1):
                plot_box_on_axis(ax3, partitions_1[i],
                                facecolor='#51cf66', edgecolor='#2f9e44',
                                alpha=0.5, linewidth=1.5)
        
        # Plot sub-partitions with different results
        sub_colors = ['#51cf66', '#ff6b6b', '#51cf66', '#51cf66']  # One unsafe
        for i, (sub_part, color) in enumerate(zip(sub_partitions, sub_colors)):
            plot_box_on_axis(ax3, sub_part,
                            facecolor=color, edgecolor='black',
                            alpha=0.6, linewidth=1.2)
            
            center = (sub_part.low + sub_part.up) / 2
            status = 'U' if color == '#ff6b6b' else 'S'
            ax3.text(center[0], center[1], status,
                    fontsize=9, ha='center', va='center', fontweight='bold',
                    color='white' if status == 'U' else 'black',
                    bbox=dict(boxstyle='circle,pad=0.2', 
                             facecolor=color, edgecolor='black', linewidth=1))
    
    set_axis_limits(ax3, all_boxes, padding=0.2)
    ax3.set_xlabel('$x_1$', fontsize=11)
    ax3.set_ylabel('$x_2$', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#51cf66', edgecolor='#2f9e44', alpha=0.5, label='Safe (S)'),
        Patch(facecolor='#ff6b6b', edgecolor='#c92a2a', alpha=0.5, label='Unsafe (U)'),
        Patch(facecolor='#ffd43b', edgecolor='#f59f00', alpha=0.5, label='Unknown (?)')
    ]
    ax3.legend(handles=legend_elements, loc='best', fontsize=10)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    
    return fig


def plot_verification_summary(save_path=None):
    """
    Create comprehensive summary plot showing:
    1. Safe controller trajectory
    2. Buggy controller crash
    3. FBRA partitioning resolving Unknown
    """
    
    fig = plt.figure(figsize=(20, 6))
    
    # ====================================
    # Panel 1: Safe Controller
    # ====================================
    ax1 = plt.subplot(131)
    ax1.set_title('✓ Safe Controller', fontsize=13, fontweight='bold', color='green')
    
    R_safe = forward_reach(X0_ground_robot, ground_robot_controller, ground_robot, T=9)
    safe_sets = [R_safe[t] for t in range(10)]
    
    plot_box_on_axis(ax1, Unsafe_ground_robot,
                    facecolor='#ff6b6b', alpha=0.3, linewidth=2)
    plot_box_on_axis(ax1, X0_ground_robot,
                    facecolor='#4dabf7', alpha=0.6, linewidth=2)
    
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, 10))
    for t, boxes in enumerate(safe_sets[1:], 1):
        for box in boxes:
            plot_box_on_axis(ax1, box, facecolor='none', 
                           edgecolor=colors[t], linewidth=1.5, alpha=0.8)
    
    all_safe = [Unsafe_ground_robot, X0_ground_robot] + [b for boxes in safe_sets for b in boxes]
    set_axis_limits(ax1, all_safe, padding=0.15)
    ax1.set_xlabel('$x_1$', fontsize=11)
    ax1.set_ylabel('$x_2$', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # ====================================
    # Panel 2: Buggy Controller
    # ====================================
    ax2 = plt.subplot(132)
    ax2.set_title('✗ Buggy Controller', fontsize=13, fontweight='bold', color='red')
    
    buggy = BuggyGroundRobotController()
    R_buggy = forward_reach(X0_ground_robot, buggy, ground_robot, T=5)
    buggy_sets = [R_buggy[t] for t in range(6)]
    
    plot_box_on_axis(ax2, Unsafe_ground_robot,
                    facecolor='#ff6b6b', alpha=0.3, linewidth=2)
    plot_box_on_axis(ax2, X0_ground_robot,
                    facecolor='#4dabf7', alpha=0.6, linewidth=2)
    
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, 6))
    for t, boxes in enumerate(buggy_sets[1:], 1):
        for box in boxes:
            intersects = box.intersects(Unsafe_ground_robot)
            
            plot_box_on_axis(ax2, box,
                           facecolor='none',
                           edgecolor=colors[t],
                           linewidth=2.5 if intersects else 1.5,
                           alpha=0.9)
            
            if intersects:
                center = (box.low + box.up) / 2
                ax2.scatter(center[0], center[1], s=300, c='red', 
                          marker='X', edgecolors='darkred', linewidths=3, zorder=10)
                ax2.annotate('CRASH!', xy=center, xytext=(center[0], center[1]+0.8),
                           fontsize=11, fontweight='bold', color='red',
                           ha='center',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                                    edgecolor='red', linewidth=2))
    
    all_buggy = [Unsafe_ground_robot, X0_ground_robot] + [b for boxes in buggy_sets for b in boxes]
    set_axis_limits(ax2, all_buggy, padding=0.15)
    ax2.set_xlabel('$x_1$', fontsize=11)
    ax2.set_ylabel('$x_2$', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # ====================================
    # Panel 3: FBRA Partitioning
    # ====================================
    ax3 = plt.subplot(133)
    ax3.set_title('? FBRA Partitioning', fontsize=13, fontweight='bold', color='orange')
    
    plot_box_on_axis(ax3, Unsafe_ground_robot,
                    facecolor='#ff6b6b', alpha=0.3, linewidth=2, label='Unsafe')
    
    # Show partitions
    partitions = partition_initial_set(X0_ground_robot, [X0_ground_robot],
                                      Unsafe_ground_robot, "uniform", 2)
    
    partition_colors = ['#51cf66', '#51cf66', '#ff6b6b', '#51cf66']
    partition_labels = ['Safe', 'Safe', 'Unsafe', 'Safe']
    
    for i, (part, color, label) in enumerate(zip(partitions, partition_colors, partition_labels)):
        plot_box_on_axis(ax3, part,
                        facecolor=color, edgecolor='black',
                        alpha=0.5, linewidth=1.8)
        
        center = (part.low + part.up) / 2
        ax3.text(center[0], center[1], f'{i+1}\n{label[0]}',
                fontsize=10, ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='circle,pad=0.4', facecolor='white',
                         edgecolor='black', linewidth=1.5))
    
    all_part = [Unsafe_ground_robot] + partitions
    set_axis_limits(ax3, all_part, padding=0.2)
    ax3.set_xlabel('$x_1$', fontsize=11)
    ax3.set_ylabel('$x_2$', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    plt.suptitle('FBRA Verification: Complete Process', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    
    return fig


def main():
    """Generate all publication-quality visualizations"""
    
    print("\n" + "🎨"*35)
    print("GENERATING PUBLICATION-QUALITY VISUALIZATIONS")
    print("🎨"*35)
    
    os.makedirs("results", exist_ok=True)
    
    # 1. Safe controller detailed view
    print("\n1. Safe Controller - Detailed Analysis")
    R_safe = forward_reach(X0_ground_robot, ground_robot_controller, ground_robot, T=9)
    safe_sets = [R_safe[t] for t in range(10)]
    
    plot_forward_reach_detailed(
        safe_sets,
        Unsafe_ground_robot,
        X0_ground_robot,
        title="Safe Controller - Forward Reachability Analysis",
        save_path="results/1_safe_controller_detailed.png"
    )
    plt.show()
    
    # 2. Safe vs Buggy comparison
    print("\n2. Safe vs Buggy Controller Comparison")
    plot_buggy_vs_safe_comparison(
        save_path="results/2_safe_vs_buggy.png"
    )
    
    # 3. Partitioning process
    print("\n3. FBRA Partitioning Process")
    plot_partitioning_detailed(
        save_path="results/3_partitioning_process.png"
    )
    
    print("\n" + "="*70)
    print("✓ ALL VISUALIZATIONS COMPLETE!")
    print("="*70)
    print("\nGenerated files in results/:")
    print("  1. 1_safe_controller_detailed.png - Safe controller analysis")
    print("  2. 2_safe_vs_buggy.png - Comparison plot")
    print("  3. 3_partitioning_process.png - Partitioning demonstration")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()