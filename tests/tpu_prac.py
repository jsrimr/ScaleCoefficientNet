# imports pytorch
import torch

# imports the torch_xla package
import torch_xla
import torch_xla.core.xla_model as xm


# Creates a random tensor on xla:1 (a Cloud TPU core)
dev = xm.xla_device()
t1 = torch.ones(3, 3, device = dev)
print(t1)