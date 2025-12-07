"""
Advanced Visualization for FBRA - FIXED VERSION
================================================
Creates publication-quality plots with proper coordinate scaling.

Key Fix: Uses actual Box coordinates instead of normalizing to [0,1]
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.collections import PatchCollection
import numpy as np
from fbra.boxes import Box


# Publication-quality settings
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def get_bounds_from_boxes(box_list):
    """
    Compute min/max coordinates from list of boxes
    
    Args:
        box_list: List of Box objects or tuples of (low, high)
        
    Returns:
        (x_min, x_max, y_min, y_max)
    """
    all_lows = []
    all_highs = []
    
    for item in box_list:
        if isinstance(item, Box):
            all_lows.append(item.low)
            all_highs.append(item.up)
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            # Assume (low, high) tuple
            all_lows.append(item[0] if hasattr(item[0], '__iter__') else [item[0]])
            all_highs.append(item[1] if hasattr(item[1], '__iter__') else [item[1]])
    
    if not all_lows:
        return -6, 2, -2, 2  # Default
    
    all_lows = np.array(all_lows)
    all_highs = np.array(all_highs)
    
    x_min = all_lows[:, 0].min()
    x_max = all_highs[:, 0].max()
    y_min = all_lows[:, 1].min()
    y_max = all_highs[:, 1].max()
    
    return x_min, x_max, y_min, y_max


def set_axis_limits(ax, box_list, padding=0.1):
    """
    Set axis limits based on data with padding
    
    Args:
        ax: Matplotlib axis
        box_list: List of boxes to consider
        padding: Fraction of range to add as padding
    """
    x_min, x_max, y_min, y_max = get_bounds_from_boxes(box_list)
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    x_pad = max(x_range * padding, 0.5)
    y_pad = max(y_range * padding, 0.5)
    
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)


def plot_box_on_axis(ax, box, **kwargs):
    """
    Plot a single box on given axis using actual coordinates
    
    Args:
        ax: Matplotlib axis
        box: Box object or (low, high) tuple
        **kwargs: Passed to Rectangle (facecolor, edgecolor, alpha, etc.)
    """
    if isinstance(box, Box):
        low, high = box.low, box.up
    elif isinstance(box, (tuple, list)) and len(box) == 2:
        low, high = box[0], box[1]
    else:
        raise ValueError(f"Cannot plot box of type {type(box)}")
    
    # Ensure we have numpy arrays
    low = np.array(low)
    high = np.array(high)
    
    # Create rectangle using ACTUAL coordinates
    width = high[0] - low[0]
    height = high[1] - low[1]
    
    rect = patches.Rectangle(
        (low[0], low[1]),  # Bottom-left corner
        width,
        height,
        **kwargs
    )
    ax.add_patch(rect)
    
    return rect


def plot_reachable_sets_evolution(reachable_sets, unsafe_set, initial_set, 
                                   title="Forward Reachability Analysis",
                                   save_path=None):
    """
    Plot evolution of reachable sets over time - FIXED VERSION
    
    Args:
        reachable_sets: List where reachable_sets[t] = list of Box objects at time t
        unsafe_set: Box object for unsafe region
        initial_set: Box object for initial set
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        fig, ax
    """
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Collect all boxes for axis scaling
    all_boxes = [unsafe_set, initial_set]
    for boxes_at_t in reachable_sets:
        all_boxes.extend(boxes_at_t)
    
    # Plot unsafe region (red, semi-transparent)
    plot_box_on_axis(ax, unsafe_set,
                    facecolor='#ff6b6b', edgecolor='#c92a2a',
                    alpha=0.4, linewidth=2.5, label='Unsafe Region', zorder=1)
    
    # Plot initial set (blue, filled)
    plot_box_on_axis(ax, initial_set,
                    facecolor='#4dabf7', edgecolor='#1971c2',
                    alpha=0.7, linewidth=2.5, label='Initial Set (t=0)', zorder=2)
    
    # Plot reachable sets with color gradient
    n_steps = len(reachable_sets)
    colors = plt.cm.viridis(np.linspace(0, 1, n_steps))
    
    # Track which timesteps to label
    label_timesteps = set([1, 2, 3, n_steps-1]) if n_steps > 3 else set(range(n_steps))
    
    for t, boxes_at_t in enumerate(reachable_sets[1:], 1):  # Skip t=0 (initial)
        for box in boxes_at_t:
            plot_box_on_axis(ax, box,
                           facecolor='none', 
                           edgecolor=colors[t],
                           alpha=0.85, 
                           linewidth=1.8,
                           label=f't={t}' if t in label_timesteps else '',
                           zorder=3+t)
    
    # Add timestep annotations at centers
    annotation_timesteps = list(range(0, n_steps, max(1, n_steps//5)))  # Every ~5th timestep
    
    for t in annotation_timesteps:
        if t < len(reachable_sets):
            boxes_at_t = reachable_sets[t]
            if boxes_at_t:
                box = boxes_at_t[0]
                if isinstance(box, Box):
                    center = (box.low + box.up) / 2
                else:
                    center = (np.array(box[0]) + np.array(box[1])) / 2
                
                ax.annotate(f't={t}', 
                           xy=center, 
                           fontsize=9, 
                           ha='center', 
                           va='center',
                           bbox=dict(boxstyle='round,pad=0.4', 
                                   facecolor='white', 
                                   edgecolor='gray',
                                   alpha=0.8))
    
    # Set proper axis limits
    set_axis_limits(ax, all_boxes, padding=0.15)
    
    # Labels and formatting
    ax.set_xlabel('Position $x_1$', fontsize=13, fontweight='bold')
    ax.set_ylabel('Position $x_2$', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicates
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), 
             loc='best', fontsize=11, framealpha=0.95)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_aspect('equal', adjustable='box')
    
    # Add colorbar for time
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                               norm=plt.Normalize(vmin=0, vmax=n_steps-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Time Step', pad=0.02, fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig, ax


def plot_partitioning_process(initial_set, partitions, unsafe_set, 
                              partition_results=None,
                              title="State Space Partitioning",
                              save_path=None):
    """
    Visualize partitioning process - FIXED VERSION
    
    Args:
        initial_set: Box object for initial region
        partitions: List of Box objects (partitioned regions)
        unsafe_set: Box object for unsafe region
        partition_results: List of status strings ('Safe', 'Unsafe', 'Unknown')
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        fig, (ax1, ax2)
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Collect all boxes
    all_boxes = [unsafe_set, initial_set] + partitions
    
    # ====================================
    # LEFT: Before Partitioning
    # ====================================
    ax1.set_title('Before Partitioning', fontsize=14, fontweight='bold')
    
    plot_box_on_axis(ax1, unsafe_set,
                    facecolor='#ff6b6b', edgecolor='#c92a2a',
                    alpha=0.4, linewidth=2, label='Unsafe Region')
    
    plot_box_on_axis(ax1, initial_set,
                    facecolor='#4dabf7', edgecolor='#1971c2',
                    alpha=0.5, linewidth=2.5, label='Initial Set')
    
    set_axis_limits(ax1, all_boxes, padding=0.15)
    ax1.set_xlabel('$x_1$', fontsize=12, fontweight='bold')
    ax1.set_ylabel('$x_2$', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_aspect('equal', adjustable='box')
    
    # ====================================
    # RIGHT: After Partitioning
    # ====================================
    ax2.set_title(f'After Partitioning ({len(partitions)} regions)', 
                 fontsize=14, fontweight='bold')
    
    plot_box_on_axis(ax2, unsafe_set,
                    facecolor='#ff6b6b', edgecolor='#c92a2a',
                    alpha=0.4, linewidth=2, label='Unsafe Region')
    
    # Default results if not provided
    if partition_results is None:
        partition_results = ['Unknown'] * len(partitions)
    
    # Color map for results
    color_map = {
        'Safe': '#51cf66',      # Green
        'Unsafe': '#ff6b6b',    # Red
        'Unknown': '#ffd43b'    # Yellow
    }
    
    edge_map = {
        'Safe': '#2f9e44',
        'Unsafe': '#c92a2a',
        'Unknown': '#f59f00'
    }
    
    # Plot partitions
    for i, (partition, result) in enumerate(zip(partitions, partition_results)):
        facecolor = color_map.get(result, '#868e96')
        edgecolor = edge_map.get(result, '#495057')
        
        plot_box_on_axis(ax2, partition,
                        facecolor=facecolor, edgecolor=edgecolor,
                        alpha=0.5, linewidth=1.8)
        
        # Add partition label
        if isinstance(partition, Box):
            center = (partition.low + partition.up) / 2
        else:
            center = (np.array(partition[0]) + np.array(partition[1])) / 2
        
        # Status letter: S=Safe, U=Unsafe, ?=Unknown
        status_letter = result[0]
        
        ax2.text(center[0], center[1], 
                f'{i+1}\n{status_letter}',
                fontsize=10, 
                ha='center', 
                va='center',
                fontweight='bold',
                bbox=dict(boxstyle='circle,pad=0.4', 
                         facecolor='white', 
                         edgecolor='black',
                         linewidth=1.5,
                         alpha=0.9))
    
    # Create custom legend for partition colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff6b6b', edgecolor='#c92a2a', alpha=0.4, label='Unsafe Region'),
        Patch(facecolor='#51cf66', edgecolor='#2f9e44', alpha=0.5, label='Safe Partition'),
        Patch(facecolor='#ff6b6b', edgecolor='#c92a2a', alpha=0.5, label='Unsafe Partition'),
        Patch(facecolor='#ffd43b', edgecolor='#f59f00', alpha=0.5, label='Unknown Partition')
    ]
    
    ax2.legend(handles=legend_elements, fontsize=11, loc='best')
    
    set_axis_limits(ax2, all_boxes, padding=0.15)
    ax2.set_xlabel('$x_1$', fontsize=12, fontweight='bold')
    ax2.set_ylabel('$x_2$', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_aspect('equal', adjustable='box')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig, (ax1, ax2)


def plot_forward_backward_comparison(forward_sets, backward_sets, 
                                     unsafe_set, initial_set,
                                     intersection_set=None,
                                     title="Forward vs Backward Reachability",
                                     save_path=None):
    """
    Compare forward and backward reachable sets - FIXED VERSION
    
    Args:
        forward_sets: List where forward_sets[t] = list of Box at time t
        backward_sets: List where backward_sets[t] = list of Box at time t
        unsafe_set: Box for unsafe region
        initial_set: Box for initial set
        intersection_set: Optional list of intersection boxes
        title: Plot title
        save_path: Path to save
        
    Returns:
        fig, ax
    """
    
    fig, ax = plt.subplots(figsize=(13, 9))
    
    # Collect all boxes
    all_boxes = [unsafe_set, initial_set]
    for boxes in forward_sets:
        all_boxes.extend(boxes)
    if backward_sets:
        for boxes in backward_sets:
            all_boxes.extend(boxes)
    if intersection_set:
        all_boxes.extend(intersection_set)
    
    # Plot unsafe and initial
    plot_box_on_axis(ax, unsafe_set,
                    facecolor='#ff6b6b', edgecolor='#c92a2a',
                    alpha=0.35, linewidth=2.5, label='Unsafe Region', zorder=1)
    
    plot_box_on_axis(ax, initial_set,
                    facecolor='#4dabf7', edgecolor='#1971c2',
                    alpha=0.6, linewidth=2.5, label='Initial Set', zorder=2)
    
    # Plot forward sets (blue gradient, solid)
    n_forward = len(forward_sets)
    forward_colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_forward))
    
    for t, boxes in enumerate(forward_sets[1:], 1):  # Skip initial
        for box in boxes:
            plot_box_on_axis(ax, box,
                           facecolor='none',
                           edgecolor=forward_colors[t],
                           alpha=0.8,
                           linewidth=2,
                           linestyle='-',
                           label='Forward Sets' if t == 1 else '',
                           zorder=3)
    
    # Plot backward sets (red gradient, dashed)
    if backward_sets:
        n_backward = len(backward_sets)
        backward_colors = plt.cm.Reds(np.linspace(0.4, 0.9, n_backward))
        
        for t, boxes in enumerate(backward_sets):
            for box in boxes:
                plot_box_on_axis(ax, box,
                               facecolor='none',
                               edgecolor=backward_colors[t],
                               alpha=0.8,
                               linewidth=2,
                               linestyle='--',
                               label='Backward Sets' if t == 0 else '',
                               zorder=4)
    
    # Plot intersection if provided
    if intersection_set:
        for box in intersection_set:
            plot_box_on_axis(ax, box,
                           facecolor='#51cf66', edgecolor='#2f9e44',
                           alpha=0.5, linewidth=2.5, 
                           label='Intersection', zorder=5)
    
    set_axis_limits(ax, all_boxes, padding=0.15)
    
    ax.set_xlabel('$x_1$', fontsize=13, fontweight='bold')
    ax.set_ylabel('$x_2$', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), 
             loc='best', fontsize=11, framealpha=0.95)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig, ax


def plot_comparison_grid(results_dict, titles=None, save_path=None):
    """
    Create grid comparing multiple scenarios - FIXED VERSION
    
    Args:
        results_dict: Dict mapping scenario_name -> {
            'reachable_sets': list of boxes at each timestep,
            'unsafe': Box for unsafe region,
            'initial': Box for initial set
        }
        titles: Optional list of titles for each subplot
        save_path: Path to save
        
    Returns:
        fig, axes
    """
    
    n_scenarios = len(results_dict)
    ncols = min(3, n_scenarios)
    nrows = (n_scenarios + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    axes = np.atleast_1d(axes).flatten()
    
    if titles is None:
        titles = [name.replace('_', ' ').title() for name in results_dict.keys()]
    
    for idx, (scenario, data) in enumerate(results_dict.items()):
        ax = axes[idx]
        
        reachable_sets = data['reachable_sets']
        unsafe_set = data['unsafe']
        initial_set = data['initial']
        
        # Collect boxes for this subplot
        all_boxes = [unsafe_set, initial_set]
        for boxes in reachable_sets:
            all_boxes.extend(boxes)
        
        # Plot unsafe and initial
        plot_box_on_axis(ax, unsafe_set,
                        facecolor='#ff6b6b', alpha=0.4, linewidth=2)
        plot_box_on_axis(ax, initial_set,
                        facecolor='#4dabf7', alpha=0.6, linewidth=2)
        
        # Plot reachable sets
        n_steps = len(reachable_sets)
        colors = plt.cm.viridis(np.linspace(0, 1, n_steps))
        
        for t, boxes in enumerate(reachable_sets):
            for box in boxes:
                plot_box_on_axis(ax, box,
                               facecolor='none',
                               edgecolor=colors[t],
                               alpha=0.8,
                               linewidth=1.5)
        
        set_axis_limits(ax, all_boxes, padding=0.15)
        
        ax.set_title(titles[idx], fontsize=13, fontweight='bold')
        ax.set_xlabel('$x_1$', fontsize=11)
        ax.set_ylabel('$x_2$', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
    
    # Hide extra subplots
    for idx in range(len(results_dict), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig, axes


def create_verification_animation(reachable_sets, unsafe_set, initial_set,
                                  title="FBRA Verification Process",
                                  save_path=None):
    """
    Create animated visualization - FIXED VERSION
    
    Args:
        reachable_sets: List of boxes at each timestep
        unsafe_set: Box for unsafe region
        initial_set: Box for initial set
        title: Animation title
        save_path: Path to save GIF
        
    Returns:
        animation object
    """
    
    # Collect all boxes for axis limits
    all_boxes = [unsafe_set, initial_set]
    for boxes in reachable_sets:
        all_boxes.extend(boxes)
    
    x_min, x_max, y_min, y_max = get_bounds_from_boxes(all_boxes)
    x_pad = (x_max - x_min) * 0.15
    y_pad = (y_max - y_min) * 0.15
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(reachable_sets)))
    
    def init():
        ax.clear()
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('$x_1$', fontsize=12)
        ax.set_ylabel('$x_2$', fontsize=12)
        return []
    
    def animate(frame):
        ax.clear()
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('$x_1$', fontsize=12, fontweight='bold')
        ax.set_ylabel('$x_2$', fontsize=12, fontweight='bold')
        
        # Plot unsafe region
        plot_box_on_axis(ax, unsafe_set,
                        facecolor='#ff6b6b', edgecolor='#c92a2a',
                        alpha=0.4, linewidth=2, label='Unsafe')
        
        # Plot initial set
        plot_box_on_axis(ax, initial_set,
                        facecolor='#4dabf7', edgecolor='#1971c2',
                        alpha=0.6, linewidth=2, label='Initial')
        
        # Plot reachable sets up to current frame
        for t in range(min(frame + 1, len(reachable_sets))):
            for box in reachable_sets[t]:
                plot_box_on_axis(ax, box,
                               facecolor='none',
                               edgecolor=colors[t],
                               alpha=0.8,
                               linewidth=1.8)
        
        ax.set_title(f'{title} - Timestep {min(frame, len(reachable_sets)-1)}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        
        return []
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=len(reachable_sets) + 5,
                                  interval=500, blit=True, repeat=True)
    
    if save_path:
        try:
            anim.save(save_path, writer='pillow', fps=2)
            print(f"✓ Saved animation: {save_path}")
        except Exception as e:
            print(f"⚠ Could not save animation: {e}")
            print("  (pillow writer required: pip install pillow)")
    
    return anim


# Legacy function names for backward compatibility
plot_boxes = plot_box_on_axis


def plot_initial_and_unsafe(X0, Unsafe, save_path=None):
    """Simple plot of initial and unsafe sets"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    plot_box_on_axis(ax, Unsafe,
                    facecolor='#ff6b6b', edgecolor='#c92a2a',
                    alpha=0.4, linewidth=2, label='Unsafe Region')
    
    plot_box_on_axis(ax, X0,
                    facecolor='#4dabf7', edgecolor='#1971c2',
                    alpha=0.6, linewidth=2, label='Initial Set')
    
    set_axis_limits(ax, [X0, Unsafe], padding=0.2)
    
    ax.set_xlabel('State 1', fontsize=12)
    ax.set_ylabel('State 2', fontsize=12)
    ax.set_title('Initial and Unsafe Sets', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax