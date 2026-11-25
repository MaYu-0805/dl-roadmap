import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 数据路径锚定到本文件位置：无论从哪个目录运行，都指向 E:\dl-roadmap\data
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),   
])

train_set = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)

# num_workers=0 不涉及子进程，顶层创建安全；训练代码里 import 它们即可
train_iter = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
test_iter = DataLoader(test_set, batch_size=256, num_workers=0)

if __name__ == "__main__":
    X, y = next(iter(train_iter))
    print(X.shape, y.shape)
    print(train_set.classes)