"""
Reach-Avoid Safety Verification
================================
Extends safety checking to include target reachability.

Reach-Avoid Property:
    ∃ t ∈ [0, T]: x(t) ∈ Target  (must reach target)
    ∧
    ∀ t ∈ [0, T]: x(t) ∉ Unsafe  (must avoid unsafe)
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from fbra.boxes import Box


def reach_avoid_status(reachable_sets, target, unsafe, current_time):
    """
    Check reach-avoid status at a given timestep
    
    Args:
        reachable_sets: Dictionary {t: [boxes]} of reachable sets
        target: Box representing target region
        unsafe: Box representing unsafe region
        current_time: Current timestep to check up to
        
    Returns:
        ("status", timestep) where:
            status = "Unsafe" (hit obstacle)
                   | "Reached" (reached target, avoided obstacle)  
                   | "Unknown" (partial overlap)
                   | "InProgress" (safe so far, haven't reached target)
    """
    
    reached_target = False
    target_time = None
    
    for t in range(current_time + 1):
        boxes_at_t = reachable_sets.get(t, [])
        
        # Check unsafe collision
        for box in boxes_at_t:
            # Fully inside unsafe
            if unsafe.contains(box):
                return "Unsafe", t
            
            # Partial overlap (Unknown)
            if box.intersects(unsafe) and not reached_target:
                # If we haven't reached target yet, partial overlap is bad
                return "Unknown", t
        
        # Check target reach
        for box in boxes_at_t:
            if box.intersects(target):
                reached_target = True
                if target_time is None:
                    target_time = t
    
    # Final determination
    if reached_target:
        return "Reached", target_time
    else:
        return "InProgress", None


def reach_avoid_verify(X0, model, plant, target, unsafe, T, verbose=True):
    """
    Complete reach-avoid verification
    
    Combines FBRA verification with target checking
    
    Args:
        X0: Initial set
        model: Neural network controller
        plant: System dynamics
        target: Target region (must reach)
        unsafe: Unsafe region (must avoid)
        T: Time horizon
        verbose: Print progress
        
    Returns:
        Dictionary with:
            - 'status': "Success", "Unsafe", "Failed", "Unknown"
            - 'reached_at': Timestep when target reached (or None)
            - 'unsafe_at': Timestep when unsafe hit (or None)
            - 'reachable_sets': Forward reachable sets
    """
    
    from fbra.forward import forward_reach
    
    if verbose:
        print("\n" + "="*70)
        print("REACH-AVOID VERIFICATION")
        print("="*70)
        print(f"Initial:  {X0.low} to {X0.up}")
        print(f"Target:   {target.low} to {target.up}")
        print(f"Unsafe:   {unsafe.low} to {unsafe.up}")
        print(f"Horizon:  T = {T}")
        print("="*70)
    
    # Compute forward reachability
    R = forward_reach(X0, model, plant, T)
    
    reached_target = False
    target_time = None
    unsafe_time = None
    unknown_time = None
    
    # Check each timestep
    for t in range(T + 1):
        boxes = R[t]
        
        if verbose:
            print(f"\nt={t}:")
        
        # Check unsafe (AVOID)
        has_unsafe_intersection = False
        fully_unsafe = False
        
        for box in boxes:
            if unsafe.contains(box):
                fully_unsafe = True
                unsafe_time = t
                if verbose:
                    print(f"  ✗ Fully inside unsafe region")
                break
            
            if box.intersects(unsafe):
                has_unsafe_intersection = True
                if unknown_time is None:
                    unknown_time = t
                if verbose:
                    print(f"  ⚠ Partial overlap with unsafe")
        
        if fully_unsafe:
            result = {
                'status': 'Unsafe',
                'reached_at': target_time,
                'unsafe_at': unsafe_time,
                'reachable_sets': R
            }
            
            if verbose:
                print("\n" + "="*70)
                print(f"RESULT: Unsafe (crashed at t={unsafe_time})")
                print("="*70)
            
            return result
        
        # Check target (REACH)
        for box in boxes:
            if box.intersects(target):
                if not reached_target:
                    reached_target = True
                    target_time = t
                    if verbose:
                        print(f"  ✓ Reached target region!")
    
    # Final determination
    if verbose:
        print("\n" + "="*70)
    
    if reached_target and not has_unsafe_intersection:
        status = 'Success'
        if verbose:
            print(f"RESULT: Success!")
            print(f"  ✓ Reached target at t={target_time}")
            print(f"  ✓ Avoided unsafe for all timesteps")
    
    elif reached_target and has_unsafe_intersection:
        status = 'Unknown'
        if verbose:
            print(f"RESULT: Unknown")
            print(f"  ✓ Reached target at t={target_time}")
            print(f"  ⚠ But partial overlap with unsafe at t={unknown_time}")
            print(f"  → Need FBRA refinement to resolve")
    
    elif not reached_target and not has_unsafe_intersection:
        status = 'Failed'
        if verbose:
            print(f"RESULT: Failed to Reach Target")
            print(f"  ✓ Avoided unsafe")
            print(f"  ✗ Did not reach target in {T} timesteps")
    
    else:
        status = 'Unknown'
        if verbose:
            print(f"RESULT: Unknown")
            print(f"  ✗ Did not reach target")
            print(f"  ⚠ Partial overlap with unsafe")
    
    if verbose:
        print("="*70)
    
    result = {
        'status': status,
        'reached_at': target_time,
        'unsafe_at': unsafe_time,
        'reachable_sets': R
    }
    
    return result