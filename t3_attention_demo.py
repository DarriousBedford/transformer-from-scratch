import torch
from torch.nn import functional as F

torch.manual_seed(42)

T = 4   # 4 positions (like 4 characters)
C = 2   # each position is a vector of 2 numbers
x = torch.randn(T, C)
print("x:\n", x)

# 1. The triangle mask
tril = torch.tril(torch.ones(T, T))        # lower triangle of 1s
print("tril:\n", tril)

# 2. Equal scores, but block the future
wei = torch.zeros(T, T)                                     # every score starts equal (0)
wei = wei.masked_fill(tril == 0, float("-inf"))     # future positions → -infinity
print("masked scores:\n", wei)

wei = F.softmax(wei, dim=-1)              # scores → weights that add up to 1
print("weights:\n", wei)

# 3. Each position = weighted average of the past
out = wei @ x
print("out:\n", out)