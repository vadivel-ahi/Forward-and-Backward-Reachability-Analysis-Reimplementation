# utils/merge.py
import numpy as np
from fbra.boxes import Box

def merge_boxes(box_list):
    """Merge multiple boxes into one big bounding box."""
    lows = [b.low for b in box_list]
    ups  = [b.up  for b in box_list]

    low = np.min(np.array(lows), axis=0)
    up  = np.max(np.array(ups),  axis=0)

    return Box(low, up)

def merge_box_list(boxes):
    if len(boxes) == 1:
        return boxes[0]
    return merge_boxes(boxes)
