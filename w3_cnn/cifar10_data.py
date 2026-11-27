import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 数据路径锚定到本文件位置：无论从哪个目录运行，都指向 E:\dl-roadmap\data
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

# 训练集 transform：随机增强治过拟合（每 epoch 看到的图片都不同）
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),       # 随机水平翻转：狗翻过来还是狗
    transforms.RandomCrop(32, padding=4),    # 随机裁剪：物体不一定在正中央
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# 测试集 transform：原样评测，绝不能加随机增强（评测需要确定性）
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_set = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_tf)
test_set = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_tf)

# num_workers=0 不涉及子进程，顶层创建安全；训练代码里 import 它们即可
train_iter = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
test_iter = DataLoader(test_set, batch_size=256, num_workers=0)

if __name__ == "__main__":
    X, y = next(iter(train_iter))
    print(X.shape, y.shape)
    print(train_set.classes)
