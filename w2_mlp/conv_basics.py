import torch
from torch import nn

X = torch.rand(1, 1, 6, 6)
K = torch.rand(1, 1, 3, 3)

Y_api = nn.functional.conv2d(X, K)
print(Y_api.shape)

H, W = 6 - 3 + 1, 6 - 3 + 1
Y_hand = torch.zeros(1, 1, H, W)
for i in range(H):
    for j in range(W):
        Y_hand[0, 0, i, j] = (X[0, 0, i:i+3, j:j+3] * K[0, 0]).sum()
print(torch.allclose(Y_api, Y_hand, atol=1e-5))  
