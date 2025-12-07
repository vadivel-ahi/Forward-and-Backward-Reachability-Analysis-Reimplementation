# fbra/boxes.py
import numpy as np

class Box:
    def __init__(self, low, up):
        self.low = np.array(low, dtype=float)
        self.up  = np.array(up,  dtype=float)

    def intersect(self, other):
        low = np.maximum(self.low, other.low)
        up  = np.minimum(self.up,  other.up)
        if np.any(low > up):
            return None
        return Box(low, up)

    def intersects(self, other):
        """Return True if intersection is non-empty."""
        return self.intersect(other) is not None

    def contains(self, other):
        """
        Return True if THIS box fully contains the OTHER box.
        """
        return np.all(self.low <= other.low) and np.all(self.up >= other.up)

    def width(self):
        return self.up - self.low

    def split(self, dim, k):
        step = (self.up[dim] - self.low[dim]) / k
        boxes = []
        for i in range(k):
            l = self.low.copy()
            u = self.up.copy()
            l[dim] = self.low[dim] + i * step
            u[dim] = self.low[dim] + (i + 1) * step
            boxes.append(Box(l, u))
        return boxes
