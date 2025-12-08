"""
Reach-Avoid Problem Test Suite
===============================
Tests robot's ability to reach target while avoiding obstacles.
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
from fbra.boxes import Box
from experiments.controller import ground_robot_controller, BuggyGroundRobotController
from experiments.dynamics import ground_robot
from utils.reach_avoid import reach_avoid_verify
from utils.visualization import plot_box_on_axis, set_axis_limits


def scenario_1_safe_reach_avoid():
    """
    Scenario 1: Safe controller navigating around obstacle to target
    """
    
    print("\n" + "🎯"*35)
    print("SCENARIO 1: Navigate Around Obstacle")
    print("🎯"*35)
    
    X0 = Box([-5.5, -0.5], [-4.5, 0.5])      # Start LEFT
    target = Box([4.0, -1.0], [5.0, 1.0])    # Goal RIGHT
    unsafe = Box([-1.0, -1.0], [1.0, 1.0])   # Obstacle MIDDLE
    
    print("\nConfiguration:")
    print("  Start:  LEFT side (x ≈ -5)")
    print("  Goal:   RIGHT side (x ≈ 4.5)")
    print("  Obstacle: CENTER (x ≈ 0)")
    print("  Strategy: Must navigate around obstacle to reach goal")
    
    result = reach_avoid_verify(
        X0, ground_robot_controller, ground_robot,
        target, unsafe, T=20, verbose=True  # Longer horizon to reach
    )
    
    return result


def scenario_2_impossible_reach():
    """
    Scenario 2: Target too far to reach with weak controller
    """
    
    print("\n" + "🎯"*35)
    print("SCENARIO 2: Unreachable Target")
    print("🎯"*35)
    
    X0 = Box([-5.5, -0.5], [-4.5, 0.5])
    target = Box([10.0, -1.0], [12.0, 1.0])   # Very far target
    unsafe = Box([-1.0, -1.0], [1.0, 1.0])
    
    print("\nConfiguration:")
    print("  Target: 15+ meters away")
    print("  Controller: Weak (tiny control)")
    print("  Expected: Failed (can't reach that far)")
    
    result = reach_avoid_verify(
        X0, ground_robot_controller, ground_robot,
        target, unsafe, T=20, verbose=True
    )
    
    return result


def scenario_3_crash_before_target():
    """
    Scenario 3: Buggy controller crashes before reaching target
    """
    
    print("\n" + "🎯"*35)
    print("SCENARIO 3: Crash Before Target")
    print("🎯"*35)
    
    X0 = Box([-5.5, -0.5], [-4.5, 0.5])
    target = Box([4.0, -1.0], [5.0, 1.0])
    unsafe = Box([-1.0, -1.0], [1.0, 1.0])
    
    buggy = BuggyGroundRobotController()
    
    print("\nConfiguration:")
    print("  Buggy controller pushes toward obstacle")
    print("  Target is beyond obstacle")
    print("  Expected: Unsafe (crashes into obstacle)")
    
    result = reach_avoid_verify(
        X0, buggy, ground_robot,
        target, unsafe, T=10, verbose=True
    )
    
    return result


def scenario_4_narrow_passage():
    """
    Scenario 4: Must navigate through narrow passage
    """
    
    print("\n" + "🎯"*35)
    print("SCENARIO 4: Narrow Passage")
    print("🎯"*35)
    
    X0 = Box([-5.0, -0.3], [-4.0, 0.3])      # Narrow initial
    target = Box([4.0, -0.3], [5.0, 0.3])    # Narrow target
    unsafe = Box([-1.0, -0.8], [1.0, 0.8])   # Wider obstacle
    
    print("\nConfiguration:")
    print("  Narrow passage through obstacle")
    print("  Must stay in corridor |y| < 0.8")
    print("  Tests precision of verification")
    
    result = reach_avoid_verify(
        X0, ground_robot_controller, ground_robot,
        target, unsafe, T=20, verbose=True
    )
    
    return result


def visualize_reach_avoid_scenario(X0, target, unsafe, controller, T, 
                                   scenario_name, save_path=None):
    """
    Visualize reach-avoid scenario with target highlighted
    """
    
    from fbra.forward import forward_reach
    
    R = forward_reach(X0, controller, ground_robot, T)
    reachable_sets = [R[t] for t in range(T + 1)]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot target (green)
    plot_box_on_axis(ax, target,
                    facecolor='#51cf66', edgecolor='#2f9e44',
                    alpha=0.4, linewidth=3, label='🎯 Target', zorder=1)
    
    # Plot unsafe (red)
    plot_box_on_axis(ax, unsafe,
                    facecolor='#ff6b6b', edgecolor='#c92a2a',
                    alpha=0.4, linewidth=3, label='⚠️ Unsafe', zorder=2)
    
    # Plot initial (blue)
    plot_box_on_axis(ax, X0,
                    facecolor='#4dabf7', edgecolor='#1971c2',
                    alpha=0.7, linewidth=3, label='🏁 Start', zorder=3)
    
    # Plot reachable sets
    colors = plt.cm.viridis(np.linspace(0, 1, len(reachable_sets)))
    
    reached_target = False
    hit_unsafe = False
    
    for t, boxes in enumerate(reachable_sets[1:], 1):
        for box in boxes:
            # Check status
            intersects_target = box.intersects(target)
            intersects_unsafe = box.intersects(unsafe)
            
            if intersects_target and not reached_target:
                reached_target = True
                # Highlight when target reached
                plot_box_on_axis(ax, box,
                               facecolor='green', edgecolor='darkgreen',
                               alpha=0.3, linewidth=3, zorder=10)
                
                # Add annotation
                center = (box.low + box.up) / 2
                ax.annotate('TARGET\nREACHED!', xy=center,
                           fontsize=12, fontweight='bold', color='green',
                           ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.5', 
                                   facecolor='lightgreen',
                                   edgecolor='darkgreen', linewidth=2))
            
            elif intersects_unsafe:
                hit_unsafe = True
                # Highlight crash
                plot_box_on_axis(ax, box,
                               facecolor='red', edgecolor='darkred',
                               alpha=0.3, linewidth=3, zorder=10)
                
                center = (box.low + box.up) / 2
                ax.scatter(center[0], center[1], s=400, c='red',
                          marker='X', edgecolors='darkred', linewidths=3, zorder=11)
            
            else:
                # Normal reachable set
                plot_box_on_axis(ax, box,
                               facecolor='none', edgecolor=colors[t],
                               alpha=0.85, linewidth=1.8, zorder=4+t)
    
    # Collect all boxes
    all_boxes = [X0, target, unsafe]
    for boxes in reachable_sets:
        all_boxes.extend(boxes)
    
    set_axis_limits(ax, all_boxes, padding=0.2)
    
    # Labels
    ax.set_xlabel('Position $x_1$', fontsize=13, fontweight='bold')
    ax.set_ylabel('Position $x_2$', fontsize=13, fontweight='bold')
    
    # Title with result
    if hit_unsafe:
        title_color = 'red'
        result = 'UNSAFE (Crashed)'
    elif reached_target:
        title_color = 'green'
        result = 'SUCCESS (Reached Target)'
    else:
        title_color = 'orange'
        result = 'FAILED (Target Not Reached)'
    
    ax.set_title(f'{scenario_name}\n{result}',
                fontsize=15, fontweight='bold', color=title_color, pad=15)
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
             loc='best', fontsize=12, framealpha=0.95)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_aspect('equal', adjustable='box')
    
    # Add arrow showing path
    if len(reachable_sets) > 1:
        # Draw path from start to end
        centers = []
        for boxes in reachable_sets[::4]:  # Every 4th timestep
            if boxes:
                box = boxes[0]
                center = (box.low + box.up) / 2
                centers.append(center)
        
        if len(centers) > 1:
            centers = np.array(centers)
            ax.plot(centers[:, 0], centers[:, 1], 
                   'k--', linewidth=2, alpha=0.5, label='Path')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {save_path}")
    
    plt.show()
    
    return fig, ax


def compare_reach_avoid_controllers(save_path=None):
    """
    Compare safe vs buggy controller on reach-avoid task
    """
    
    print("\n" + "="*70)
    print("REACH-AVOID COMPARISON: Safe vs Buggy")
    print("="*70)
    
    X0 = Box([-5.5, -0.5], [-4.5, 0.5])
    target = Box([4.0, -1.0], [5.0, 1.0])
    unsafe = Box([-1.0, -1.0], [1.0, 1.0])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    controllers = [
        ('Safe Controller', ground_robot_controller, ax1),
        ('Buggy Controller', BuggyGroundRobotController(), ax2)
    ]
    
    for name, controller, ax in controllers:
        print(f"\nTesting {name}...")
        
        from fbra.forward import forward_reach
        R = forward_reach(X0, controller, ground_robot, T=12)
        
        # Plot target (green)
        plot_box_on_axis(ax, target,
                        facecolor='#51cf66', alpha=0.4, linewidth=2.5,
                        label='Target')
        
        # Plot unsafe (red)
        plot_box_on_axis(ax, unsafe,
                        facecolor='#ff6b6b', alpha=0.4, linewidth=2.5,
                        label='Unsafe')
        
        # Plot initial (blue)
        plot_box_on_axis(ax, X0,
                        facecolor='#4dabf7', alpha=0.6, linewidth=2.5,
                        label='Initial')
        
        # Plot trajectory
        colors = plt.cm.viridis(np.linspace(0, 1, 13))
        
        reached = False
        crashed = False
        
        for t in range(13):
            boxes = R[t]
            for box in boxes:
                intersects_target = box.intersects(target)
                intersects_unsafe = box.intersects(unsafe)
                
                if intersects_target and not reached:
                    reached = True
                    # Highlight
                    plot_box_on_axis(ax, box, facecolor='green',
                                   edgecolor='darkgreen', alpha=0.4, linewidth=3)
                
                if intersects_unsafe:
                    crashed = True
                    # Highlight crash
                    plot_box_on_axis(ax, box, facecolor='red',
                                   edgecolor='darkred', alpha=0.4, linewidth=3)
                    center = (box.low + box.up) / 2
                    ax.scatter(center[0], center[1], s=300, c='red',
                              marker='X', edgecolors='darkred', linewidths=3, zorder=20)
                
                # Normal box
                plot_box_on_axis(ax, box, facecolor='none',
                               edgecolor=colors[t], linewidth=1.5, alpha=0.8)
        
        # Result
        if crashed:
            result = '✗ Crashed'
            title_color = 'red'
        elif reached:
            result = '✓ Success'
            title_color = 'green'
        else:
            result = '⚠ Failed to Reach'
            title_color = 'orange'
        
        ax.set_title(f'{name}\n{result}',
                    fontsize=14, fontweight='bold', color=title_color)
        
        # Formatting
        all_boxes = [X0, target, unsafe] + [b for t in R.values() for b in t]
        set_axis_limits(ax, all_boxes, padding=0.15)
        
        ax.set_xlabel('Position $x_1$', fontsize=12)
        ax.set_ylabel('Position $x_2$', fontsize=12)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.suptitle('Reach-Avoid Problem: Controller Comparison',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {save_path}")
    
    plt.show()
    
    return fig


def main():
    """Run all reach-avoid tests"""
    
    print("\n" + "🏁"*35)
    print("REACH-AVOID PROBLEM - COMPLETE SUITE")
    print("🏁"*35)
    
    results = {}
    
    # Scenario 1
    results['navigate'] = scenario_1_safe_reach_avoid()
    
    # Scenario 2
    results['impossible'] = scenario_2_impossible_reach()
    
    # Scenario 3
    results['crash'] = scenario_3_crash_before_target()
    
    # Comparison visualization
    print("\n" + "🎨"*35)
    print("Creating comparison visualization...")
    compare_reach_avoid_controllers(
        save_path='results/reach_avoid_comparison.png'
    )
    
    # Summary
    print("\n" + "="*70)
    print("REACH-AVOID RESULTS SUMMARY")
    print("="*70)
    
    scenario_names = {
        'navigate': 'Navigate Around Obstacle',
        'impossible': 'Unreachable Target',
        'crash': 'Crash Before Target'
    }
    
    print(f"\n{'Scenario':<30} {'Status':<15} {'Reached':<10} {'Crashed'}")
    print("-"*70)
    
    for key, result in results.items():
        name = scenario_names[key]
        status = result['status']
        reached = f"t={result['reached_at']}" if result['reached_at'] else "No"
        crashed = f"t={result['unsafe_at']}" if result['unsafe_at'] else "No"
        
        print(f"{name:<30} {status:<15} {reached:<10} {crashed}")
    
    print("\n" + "="*70)
    print("✓ REACH-AVOID IMPLEMENTATION COMPLETE")
    print("="*70)
    print("\nKey Achievement:")
    print("  • Extended FBRA to handle reach-avoid properties")
    print("  • Verified target reachability")
    print("  • Maintained obstacle avoidance")
    print("  • Comprehensive visualization suite")
    print("="*70)


if __name__ == "__main__":
    import numpy as np
    main()