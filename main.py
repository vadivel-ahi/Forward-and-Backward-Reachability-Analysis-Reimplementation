# main.py
from fbra.boxes import Box
from fbra.verifier import verify
from experiments.controller import controller
from experiments.dynamics import plant
from experiments.sets import X0, Unsafe

T = 10

result = verify(X0, controller, plant, Unsafe, T)
print("Verification result:", result)
