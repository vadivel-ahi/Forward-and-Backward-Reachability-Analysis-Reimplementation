"""
Test FBRA on Double Integrator Benchmark
=========================================
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import time
import matplotlib.pyplot as plt
from fbra.verifier import verify_fbra
from fbra.forward import forward_reach
from experiments.controller import double_integrator_controller
from experiments.dynamics import double_integrator
from experiments.sets import X0_double_integrator, Unsafe_double_integrator
from utils.visualization import plot_reachable_sets_evolution, plot_box_on_axis, set_axis_limits


def test_double_integrator_basic():
    """Basic forward reachability test with unsafe region visible"""
    
    print("\n" + "="*70)
    print("DOUBLE INTEGRATOR - Basic Forward Reachability")
    print("="*70)
    
    print("\nConfiguration:")
    print(f"  State: [position, velocity]")
    print(f"  Control: [acceleration]")
    print(f"  Initial:  pos ∈ {X0_double_integrator.low} to {X0_double_integrator.up}")
    print(f"  Unsafe:   pos ∈ {Unsafe_double_integrator.low} to {Unsafe_double_integrator.up}")
    print(f"  Dynamics: dt = 1.0s")
    print(f"  Horizon:  T = 5")
    
    # Forward reachability
    print("\nComputing forward reachability...")
    R_forward = forward_reach(
        X0_double_integrator,
        double_integrator_controller,
        double_integrator,
        T=5
    )
    
    # Check each timestep
    print("\nReachable sets:")
    for t in range(6):
        boxes = R_forward[t]
        print(f"\nt={t}:")
        for box in boxes:
            print(f"  pos ∈ [{box.low[0]:.3f}, {box.up[0]:.3f}]")
            print(f"  vel ∈ [{box.low[1]:.3f}, {box.up[1]:.3f}]")
            
            # Check safety
            if box.intersects(Unsafe_double_integrator):
                print(f"  ⚠ Intersects unsafe!")
            elif Unsafe_double_integrator.contains(box):
                print(f"  ✗ Inside unsafe!")
            else:
                print(f"  ✓ Safe")
    
    # Visualize with FIXED axis limits
    print("\nCreating visualization...")
    reachable_sets = [R_forward[t] for t in range(6)]
    
    fig, ax = plot_reachable_sets_evolution(
        reachable_sets,
        Unsafe_double_integrator,
        X0_double_integrator,
        title="Double Integrator - Forward Reachability",
        save_path="results/double_integrator_forward.png"
    )
    
    # FORCE axis limits to show both reachable sets AND unsafe region
    ax.set_xlim(-8, 6)   # Show from -8 to +6 (includes both ends)
    ax.set_ylim(-1.5, 1.5)  # Reasonable velocity range
    
    # Adjust axis labels for double integrator
    ax.set_xlabel('Position', fontsize=13, fontweight='bold')
    ax.set_ylabel('Velocity', fontsize=13, fontweight='bold')
    
    # Add annotation showing distance
    import numpy as np
    
    # Final reachable position
    final_box = reachable_sets[-1][0]
    final_pos_max = final_box.up[0]
    
    # Unsafe position
    unsafe_pos_min = Unsafe_double_integrator.low[0]
    
    # Draw arrow showing gap
    arrow_y = -1.2
    ax.annotate('', xy=(unsafe_pos_min, arrow_y), 
               xytext=(final_pos_max, arrow_y),
               arrowprops=dict(arrowstyle='<->', lw=2, color='green'))
    
    distance = unsafe_pos_min - final_pos_max
    ax.text((final_pos_max + unsafe_pos_min)/2, arrow_y - 0.15,
           f'Safe gap: {distance:.1f}m',
           ha='center', fontsize=11, fontweight='bold', color='green',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', 
                    edgecolor='green', linewidth=2))
    
    plt.tight_layout()
    plt.savefig("results/double_integrator_forward_complete.png", dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print("\n✓ Visualization saved to results/double_integrator_forward_complete.png")
    print(f"✓ Final position: {final_pos_max:.2f}")
    print(f"✓ Unsafe starts at: {unsafe_pos_min:.2f}")
    print(f"✓ Safety margin: {distance:.2f} meters")


def test_double_integrator_fbra():
    """Full FBRA verification"""
    
    print("\n" + "="*70)
    print("DOUBLE INTEGRATOR - FBRA Verification")
    print("="*70)
    
    start = time.time()
    
    result = verify_fbra(
        X0=X0_double_integrator,
        model=double_integrator_controller,
        plant=double_integrator,
        unsafe=Unsafe_double_integrator,
        T=5,
        max_iterations=10,
        verbose=True
    )
    
    elapsed = time.time() - start
    
    print("\n" + "="*70)
    print("DOUBLE INTEGRATOR RESULTS")
    print("="*70)
    print(f"Status: {result.status}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Iterations: {result.iterations}")
    print("="*70)
    
    return result


def test_double_integrator_dual_view():
    """
    Create dual-view visualization:
    1. Zoomed to data (current view)
    2. Full view showing unsafe region
    """
    
    print("\n" + "="*70)
    print("DOUBLE INTEGRATOR - Dual View Visualization")
    print("="*70)
    
    # Compute reachability
    R_forward = forward_reach(
        X0_double_integrator,
        double_integrator_controller,
        double_integrator,
        T=5
    )
    
    reachable_sets = [R_forward[t] for t in range(6)]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # ====================================
    # LEFT: Zoomed to reachable sets
    # ====================================
    ax1.set_title('Zoomed View (Reachable Region)', fontsize=13, fontweight='bold')
    
    # Plot unsafe (even if off-screen, for legend)
    plot_box_on_axis(ax1, Unsafe_double_integrator,
                    facecolor='#ff6b6b', edgecolor='#c92a2a',
                    alpha=0.4, linewidth=2, label='Unsafe (off-screen)', zorder=1)
    
    # Plot initial
    plot_box_on_axis(ax1, X0_double_integrator,
                    facecolor='#4dabf7', edgecolor='#1971c2',
                    alpha=0.7, linewidth=2.5, label='Initial', zorder=2)
    
    # Plot reachable sets
    colors = plt.cm.viridis(np.linspace(0, 1, 6))
    for t, boxes in enumerate(reachable_sets[1:], 1):
        for box in boxes:
            plot_box_on_axis(ax1, box, facecolor='none',
                           edgecolor=colors[t], linewidth=2, alpha=0.9,
                           label=f't={t}' if t in [1, 5] else '', zorder=3+t)
    
    # Auto-scale to reachable sets only
    all_reach = [X0_double_integrator] + [b for boxes in reachable_sets for b in boxes]
    set_axis_limits(ax1, all_reach, padding=0.15)
    
    ax1.set_xlabel('Position', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Velocity', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # ====================================
    # RIGHT: Full view including unsafe
    # ====================================
    ax2.set_title('Full View (Including Unsafe Region)', fontsize=13, fontweight='bold')
    
    # Plot unsafe
    plot_box_on_axis(ax2, Unsafe_double_integrator,
                    facecolor='#ff6b6b', edgecolor='#c92a2a',
                    alpha=0.5, linewidth=3, label='Unsafe Region', zorder=1)
    
    # Plot initial
    plot_box_on_axis(ax2, X0_double_integrator,
                    facecolor='#4dabf7', edgecolor='#1971c2',
                    alpha=0.8, linewidth=3, label='Initial', zorder=2)
    
    # Plot final reachable set (t=5)
    final_box = reachable_sets[-1][0]
    plot_box_on_axis(ax2, final_box,
                    facecolor='#ffd43b', edgecolor='#f59f00',
                    alpha=0.6, linewidth=3, label=f'Final (t=5)', zorder=10)
    
    # Draw arrow showing trajectory direction
    initial_center = (X0_double_integrator.low + X0_double_integrator.up) / 2
    final_center = (final_box.low + final_box.up) / 2
    
    ax2.annotate('', xy=final_center, xytext=initial_center,
                arrowprops=dict(arrowstyle='->', lw=3, color='blue', alpha=0.7))
    
    ax2.text((initial_center[0] + final_center[0])/2, 
            (initial_center[1] + final_center[1])/2 + 0.3,
            'System\nTrajectory', fontsize=11, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor='blue', linewidth=2))
    
    # Draw distance annotation
    final_pos = final_center[0]
    unsafe_pos = Unsafe_double_integrator.low[0]
    distance = unsafe_pos - final_pos
    
    ax2.annotate('', xy=(unsafe_pos, -1.0), xytext=(final_pos, -1.0),
                arrowprops=dict(arrowstyle='<->', lw=2.5, color='green'))
    
    ax2.text((final_pos + unsafe_pos)/2, -1.3,
            f'Safety Margin\n{distance:.1f} meters',
            ha='center', fontsize=11, fontweight='bold', color='green',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen',
                     edgecolor='darkgreen', linewidth=2))
    
    # Set full view limits
    ax2.set_xlim(-8, 6)
    ax2.set_ylim(-1.5, 1.5)
    
    ax2.set_xlabel('Position', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Velocity', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='k', linewidth=0.5, alpha=0.3)
    ax2.axvline(0, color='k', linewidth=0.5, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.suptitle('Double Integrator - Dual View Analysis',
                fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("results/double_integrator_dual_view.png", dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print("\n✓ Dual view saved to results/double_integrator_dual_view.png")


def main():
    """Run all Double Integrator tests"""
    
    print("\n" + "🔵"*35)
    print("DOUBLE INTEGRATOR BENCHMARK SUITE")
    print("🔵"*35)
    
    # Test 1: Basic forward reach
    test_double_integrator_basic()
    
    # Test 2: Dual view (zoomed + full)
    test_double_integrator_dual_view()
    
    # Test 3: Full FBRA verification
    test_double_integrator_fbra()
    
    print("\n" + "="*70)
    print("✓ DOUBLE INTEGRATOR TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()