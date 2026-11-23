import torch
from torch import nn

conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
X=torch.rand(1,3,8,8)
print(conv(X).shape)

conv_p=nn.Conv2d(3,16,kernel_size=3,padding=1)
print(conv_p(X).shape)

conv_s=nn.Conv2d(3,16,kernel_size=3,stride=2,padding=1)
print(conv_s(X).shape)

pool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
print(pool(X).shape)

print(sum(p.numel()for p in conv.parameters()))

bn=nn.BatchNorm2d(16)
Y=bn(torch.rand(4,16,8,8))
print(Y.mean(),Y.std())
print(len(list(bn.parameters())))