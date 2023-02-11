import torch

t = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=int)
print(torch.mean(t, axis=0, dtype=float))