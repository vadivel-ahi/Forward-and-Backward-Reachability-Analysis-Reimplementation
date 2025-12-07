"""
State Space Partitioning for FBRA
==================================
Implements Algorithm 2 and 3 from the paper.

Key functions:
    - partition_initial_set: Partition X0 based on backward reach
    - adaptive_partition: Distance and time-aware partitioning
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from fbra.boxes import Box


def partition_initial_set(X0, R_backward, unsafe, method="uniform", n_splits=2):
    """
    Partition initial set X0 based on backward reachable set
    
    When backward analysis reaches t=0 with non-empty intersection,
    we need to partition X0 to refine the analysis.
    
    Args:
        X0: Initial set (Box)
        R_backward: Backward reachable set at t=0 (list of Box)
        unsafe: Unsafe region (Box)
        method: Partitioning strategy ("uniform", "adaptive", "guided")
        n_splits: Number of splits per dimension
        
    Returns:
        List of Box representing partitioned X0
    """
    
    if method == "uniform":
        return uniform_partition(X0, n_splits)
    
    elif method == "adaptive":
        return adaptive_partition(X0, R_backward, unsafe, n_splits)
    
    elif method == "guided":
        return guided_partition(X0, R_backward, unsafe)
    
    else:
        raise ValueError(f"Unknown partitioning method: {method}")


def uniform_partition(box, n_splits=2):
    """
    Uniformly partition box into n_splits^n_dims sub-boxes
    
    Example for 2D box with n_splits=2:
        ┌─────┬─────┐
        │  1  │  2  │
        ├─────┼─────┤
        │  3  │  4  │
        └─────┴─────┘
    
    Args:
        box: Box to partition
        n_splits: Number of splits per dimension
        
    Returns:
        List of sub-boxes
    """
    
    n_dims = len(box.low)
    partitions = []
    
    # Generate split points for each dimension
    split_points = []
    for dim in range(n_dims):
        points = np.linspace(box.low[dim], box.up[dim], n_splits + 1)
        split_points.append(points)
    
    # Generate all combinations of sub-boxes
    def generate_boxes(dim, current_low, current_up):
        if dim == n_dims:
            partitions.append(Box(current_low.copy(), current_up.copy()))
            return
        
        for i in range(n_splits):
            current_low[dim] = split_points[dim][i]
            current_up[dim] = split_points[dim][i + 1]
            generate_boxes(dim + 1, current_low, current_up)
    
    generate_boxes(0, np.zeros(n_dims), np.zeros(n_dims))
    
    return partitions


def adaptive_partition(X0, R_backward, unsafe, base_splits=2):
    """
    Adaptive partitioning based on intersection with backward reach
    
    Strategy:
        1. Compute intersection: I = X0 ∩ R_backward
        2. Partition based on distance to intersection
        3. More splits near intersection, fewer far away
    
    Args:
        X0: Initial set
        R_backward: Backward reachable sets at t=0
        unsafe: Unsafe region
        base_splits: Base number of splits
        
    Returns:
        List of sub-boxes with adaptive granularity
    """
    
    # Compute intersection with backward reach
    intersections = []
    for R_b in R_backward:
        intersection = X0.intersect(R_b)
        if intersection is not None:
            intersections.append(intersection)
    
    if len(intersections) == 0:
        # No intersection, just uniform partition
        return uniform_partition(X0, base_splits)
    
    # Merge all intersections into bounding box
    from utils.merge import merge_boxes
    intersection_region = merge_boxes(intersections)
    
    # Choose dimension to split based on largest width
    widths = X0.width()
    split_dim = np.argmax(widths)
    
    # Adaptive split count based on proximity to intersection
    # More splits near intersection, fewer far away
    partitions = []
    
    n_dims = len(X0.low)
    
    # Split the chosen dimension adaptively
    split_points = _adaptive_split_points(
        X0.low[split_dim], 
        X0.up[split_dim],
        intersection_region.low[split_dim],
        intersection_region.up[split_dim],
        base_splits
    )
    
    # For other dimensions, split uniformly
    other_split_points = []
    for dim in range(n_dims):
        if dim == split_dim:
            other_split_points.append(split_points)
        else:
            points = np.linspace(X0.low[dim], X0.up[dim], base_splits + 1)
            other_split_points.append(points)
    
    # Generate all sub-boxes
    def generate_boxes(dim, current_low, current_up):
        if dim == n_dims:
            partitions.append(Box(current_low.copy(), current_up.copy()))
            return
        
        n_splits_this_dim = len(other_split_points[dim]) - 1
        for i in range(n_splits_this_dim):
            current_low[dim] = other_split_points[dim][i]
            current_up[dim] = other_split_points[dim][i + 1]
            generate_boxes(dim + 1, current_low, current_up)
    
    generate_boxes(0, np.zeros(n_dims), np.zeros(n_dims))
    
    return partitions


def _adaptive_split_points(x_min, x_max, int_min, int_max, base_splits):
    """
    Generate adaptive split points
    More splits near intersection region
    """
    
    points = []
    
    # Region before intersection
    if x_min < int_min:
        n_before = max(1, base_splits // 2)
        before_points = np.linspace(x_min, int_min, n_before + 1)[:-1]
        points.extend(before_points)
    
    # Intersection region (more splits)
    n_inside = base_splits
    inside_points = np.linspace(int_min, int_max, n_inside + 1)
    points.extend(inside_points[:-1])
    
    # Region after intersection
    if x_max > int_max:
        n_after = max(1, base_splits // 2)
        after_points = np.linspace(int_max, x_max, n_after + 1)
        points.extend(after_points)
    else:
        points.append(x_max)
    
    return np.array(points)


def guided_partition(X0, R_backward, unsafe):
    """
    Guided partitioning using intersection boundaries
    
    Implements Algorithm 2 from the paper:
        - Partition along intersection boundaries
        - Separate regions that lead to unsafe vs safe
    
    Args:
        X0: Initial set
        R_backward: Backward reachable sets
        unsafe: Unsafe region
        
    Returns:
        List of partitioned boxes
    """
    
    # Start with X0
    partitions = [X0]
    
    # For each backward reachable box, split along its boundaries
    for R_b in R_backward:
        intersection = X0.intersect(R_b)
        if intersection is None:
            continue
        
        # Split partitions that intersect with R_b
        new_partitions = []
        for part in partitions:
            if part.intersects(R_b):
                # Split this partition
                # Choose dimension with largest intersection width
                int_box = part.intersect(R_b)
                if int_box is not None:
                    widths = int_box.width()
                    split_dim = np.argmax(widths)
                    
                    # Split in two along the intersection boundary
                    sub_boxes = part.split(split_dim, 2)
                    new_partitions.extend(sub_boxes)
                else:
                    new_partitions.append(part)
            else:
                new_partitions.append(part)
        
        partitions = new_partitions
    
    return partitions


def distance_to_region(box, region):
    """
    Compute minimum distance from box to region
    
    Returns:
        0 if they intersect
        Positive distance otherwise
    """
    
    if box.intersects(region):
        return 0.0
    
    # Compute distance between closest points
    dist = 0.0
    for dim in range(len(box.low)):
        if box.up[dim] < region.low[dim]:
            dist += (region.low[dim] - box.up[dim]) ** 2
        elif box.low[dim] > region.up[dim]:
            dist += (box.low[dim] - region.up[dim]) ** 2
    
    return np.sqrt(dist)