# fbra/refine_forward.py
import numpy as np
from fbra.boxes import Box

def refine_forward_boxes(boxes, unsafe):
    new_boxes = []
    for b in boxes:
        if b.intersect(unsafe):
            dim = np.argmax(b.width())
            new_boxes.extend(b.split(dim, 4))
        else:
            new_boxes.append(b)
    return new_boxes
