# fbra/nn_bounds.py
import torch
import torch.nn as nn
from fbra.boxes import Box

def nn_forward_box(box: Box, model: nn.Module) -> Box:
    l = torch.tensor(box.low, dtype=torch.float32)
    u = torch.tensor(box.up,  dtype=torch.float32)

    # If the model has a `.net`, use that; otherwise assume model is Sequential
    layers = model.net if hasattr(model, "net") else model

    for layer in layers:
        if isinstance(layer, nn.Linear):
            W, b = layer.weight, layer.bias

            lw = torch.minimum(W, torch.zeros_like(W))
            uw = torch.maximum(W, torch.zeros_like(W))

            l_new = lw @ u + uw @ l + b
            u_new = uw @ u + lw @ l + b
            l, u = l_new, u_new

        elif isinstance(layer, nn.ReLU):
            l = torch.relu(l)
            u = torch.relu(u)

        else:
            raise NotImplementedError("Only Linear + ReLU layers supported")

    return Box(l.detach().numpy(), u.detach().numpy())
