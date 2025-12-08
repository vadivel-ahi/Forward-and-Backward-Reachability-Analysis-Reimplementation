"""
Double Integrator - Complete Test & Visualization Suite
========================================================
Tests all scenarios with enhanced visualizations.
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import time
import numpy as np
import matplotlib.pyplot as plt
from fbra.boxes import Box
from fbra.verifier import verify_fbra
from fbra.forward import forward_reach
from experiments.controller import (
    double_integrator_controller,
    DoubleIntegratorPDController,
    DoubleIntegratorAggressiveController
)
from experiments.dynamics import double_integrator
from utils.visualization import plot_box_on_axis, set_axis_limits


def plot_position_velocity_evolution(reachable_sets, unsafe, initial, 
                                     title="Double Integrator",
                                     save_path=None):
    """
    Create 2x2 subplot showing:
    - Phase space (position vs velocity)
    - Position over time
    - Velocity over time
    - State space volume over time
    """
    
    fig = plt.figure(figsize=(16, 12))
    
    # Extract data
    timesteps = list(range(len(reachable_sets)))
    pos_min, pos_max = [], []
    vel_min, vel_max = [], []
    volumes = []
    
    for boxes in reachable_sets:
        if boxes:
            box = boxes[0]
            pos_min.append(box.low[0])
            pos_max.append(box.up[0])
            vel_min.append(box.low[1])
            vel_max.append(box.up[1])
            volumes.append(np.prod(box.up - box.low))
    
    # ====================================
    # Subplot 1: Phase Space (top-left)
    # ====================================
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title('Phase Space (Position vs Velocity)', fontsize=12, fontweight='bold')
    
    plot_box_on_axis(ax1, unsafe, facecolor='#ff6b6b', alpha=0.4, 
                    linewidth=2, label='Unsafe')
    plot_box_on_axis(ax1, initial, facecolor='#4dabf7', alpha=0.6,
                    linewidth=2, label='Initial')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(reachable_sets)))
    for t, boxes in enumerate(reachable_sets[1:], 1):
        for box in boxes:
            plot_box_on_axis(ax1, box, facecolor='none', 
                           edgecolor=colors[t], linewidth=1.8, alpha=0.9)
    
    all_boxes = [unsafe, initial] + [b for boxes in reachable_sets for b in boxes]
    set_axis_limits(ax1, all_boxes, padding=0.15)
    
    ax1.set_xlabel('Position', fontsize=11)
    ax1.set_ylabel('Velocity', fontsize=11)
    ax1.axhline(0, color='k', linewidth=0.5, alpha=0.3)
    ax1.axvline(0, color='k', linewidth=0.5, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ====================================
    # Subplot 2: Position vs Time (top-right)
    # ====================================
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title('Position Bounds Over Time', fontsize=12, fontweight='bold')
    
    ax2.fill_between(timesteps, pos_min, pos_max, 
                     alpha=0.4, color='#4dabf7', label='Position range')
    ax2.plot(timesteps, pos_min, 'o-', color='#1971c2', 
            linewidth=2, markersize=6, label='Min position')
    ax2.plot(timesteps, pos_max, 's-', color='#1971c2',
            linewidth=2, markersize=6, label='Max position')
    
    # Mark unsafe region
    ax2.axhspan(unsafe.low[0], unsafe.up[0], alpha=0.3, 
               color='red', label='Unsafe region', zorder=0)
    
    ax2.set_xlabel('Time Step', fontsize=11)
    ax2.set_ylabel('Position', fontsize=11)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # ====================================
    # Subplot 3: Velocity vs Time (bottom-left)
    # ====================================
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_title('Velocity Bounds Over Time', fontsize=12, fontweight='bold')
    
    ax3.fill_between(timesteps, vel_min, vel_max,
                     alpha=0.4, color='#ff6b6b', label='Velocity range')
    ax3.plot(timesteps, vel_min, 'o-', color='#c92a2a',
            linewidth=2, markersize=6, label='Min velocity')
    ax3.plot(timesteps, vel_max, 's-', color='#c92a2a',
            linewidth=2, markersize=6, label='Max velocity')
    
    # Mark unsafe velocity region
    ax3.axhspan(unsafe.low[1], unsafe.up[1], alpha=0.3,
               color='red', label='Unsafe region', zorder=0)
    
    ax3.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time Step', fontsize=11)
    ax3.set_ylabel('Velocity', fontsize=11)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # ====================================
    # Subplot 4: State Space Volume (bottom-right)
    # ====================================
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_title('Reachable Set Volume Growth', fontsize=12, fontweight='bold')
    
    ax4.plot(timesteps, volumes, 'o-', color='#7950f2', 
            linewidth=2.5, markersize=8, label='Volume')
    ax4.fill_between(timesteps, 0, volumes, alpha=0.3, color='#7950f2')
    
    # Add growth rate annotation
    if len(volumes) > 1:
        growth_rate = ((volumes[-1] / volumes[0]) ** (1/(len(volumes)-1)) - 1) * 100
        ax4.text(0.98, 0.98, f'Avg growth: {growth_rate:.1f}%/step',
                transform=ax4.transAxes, fontsize=10,
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax4.set_xlabel('Time Step', fontsize=11)
    ax4.set_ylabel('Volume (Area)', fontsize=11)
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    
    return fig


def compare_controllers(save_path=None):
    """
    Compare three different controllers:
    1. Default (tiny random weights)
    2. PD controller (classic control)
    3. Aggressive (pushes toward unsafe)
    """
    
    print("\n" + "="*70)
    print("COMPARING DOUBLE INTEGRATOR CONTROLLERS")
    print("="*70)
    
    controllers = {
        'Default': double_integrator_controller,
        'PD': DoubleIntegratorPDController(),
        'Aggressive': DoubleIntegratorAggressiveController()
    }
    
    X0 = Box([0.0, 0.5], [1.0, 1.0])  # Start with positive velocity
    unsafe = Box([4.0, -0.5], [5.0, 0.5])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (name, controller) in enumerate(controllers.items()):
        ax = axes[idx]
        
        print(f"\nTesting {name} controller...")
        
        # Forward reach
        R = forward_reach(X0, controller, double_integrator, T=5)
        reachable_sets = [R[t] for t in range(6)]
        
        # Plot
        plot_box_on_axis(ax, unsafe, facecolor='#ff6b6b', alpha=0.4, linewidth=2)
        plot_box_on_axis(ax, X0, facecolor='#4dabf7', alpha=0.6, linewidth=2)
        
        colors = plt.cm.viridis(np.linspace(0, 1, 6))
        
        any_unsafe = False
        for t, boxes in enumerate(reachable_sets[1:], 1):
            for box in boxes:
                intersects = box.intersects(unsafe)
                if intersects:
                    any_unsafe = True
                
                plot_box_on_axis(ax, box, facecolor='none',
                               edgecolor=colors[t], linewidth=2,
                               linestyle='--' if intersects else '-',
                               alpha=0.9)
        
        # Title with result
        if any_unsafe:
            title_color = 'red'
            result = 'UNSAFE'
        else:
            title_color = 'green'
            result = 'SAFE'
        
        ax.set_title(f'{name} Controller - {result}', 
                    fontsize=13, fontweight='bold', color=title_color)
        
        all_boxes = [unsafe, X0] + [b for boxes in reachable_sets for b in boxes]
        set_axis_limits(ax, all_boxes, padding=0.15)
        
        ax.set_xlabel('Position', fontsize=11)
        ax.set_ylabel('Velocity', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
    
    plt.suptitle('Double Integrator Controller Comparison',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {save_path}")
    
    plt.show()
    
    return fig


def main():
    """Run complete Double Integrator suite"""
    
    print("\n" + "🚀"*35)
    print("DOUBLE INTEGRATOR - COMPLETE SUITE")
    print("🚀"*35)
    
    # 1. Test with PD controller
    print("\n1. Testing PD Controller...")
    pd_controller = DoubleIntegratorPDController()
    X0 = Box([-4.0, -0.5], [-2.4, 0.5])
    unsafe = Box([4.5, -0.25], [5.0, 0.25])
    
    R = forward_reach(X0, pd_controller, double_integrator, T=5)
    reachable_sets = [R[t] for t in range(6)]
    
    plot_position_velocity_evolution(
        reachable_sets, unsafe, X0,
        title="Double Integrator with PD Controller",
        save_path="results/double_integrator_pd_complete.png"
    )
    
    # 2. Compare controllers
    print("\n2. Comparing Different Controllers...")
    compare_controllers(
        save_path="results/double_integrator_comparison.png"
    )
    
    # 3. FBRA verification on aggressive controller
    print("\n3. FBRA Verification - Aggressive Controller...")
    aggressive = DoubleIntegratorAggressiveController()
    X0_test = Box([0.0, 0.5], [1.0, 1.0])
    unsafe_test = Box([4.0, -0.5], [5.0, 0.5])
    
    result = verify_fbra(
        X0_test, aggressive, double_integrator,
        unsafe_test, T=5, verbose=True
    )
    
    print(f"\n  Aggressive Controller Result: {result.status}")
    
    print("\n" + "="*70)
    print("✓ DOUBLE INTEGRATOR SUITE COMPLETE!")
    print("="*70)
    print("\nGenerated:")
    print("  - results/double_integrator_pd_complete.png")
    print("  - results/double_integrator_comparison.png")
    print("="*70)


if __name__ == "__main__":
    main()