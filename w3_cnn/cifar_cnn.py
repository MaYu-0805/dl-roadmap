import torch
from torch import nn


def make_cnn():
    return nn.Sequential(
        # 卷积块 1：3→32 通道，尺寸 32→16
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2),                      # 32x32 -> 16x16
        # 卷积块 2：32→64
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2),                      # 16x16 -> 8x8
        # 卷积块 3：64→128
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128), nn.ReLU(),
        nn.MaxPool2d(2),                      # 8x8 -> 4x4
        # 分类头
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, 10),
    )


def accuracy(net, data_iter, device):
    correct, total = 0, 0
    net.eval()
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            pred = net(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total

if __name__ == "__main__":
    from cifar10_data import train_iter, test_iter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")

    net = make_cnn().to(device)         
    loss_fn = nn.CrossEntropyLoss()
    updater = torch.optim.Adam(net.parameters(), lr=1e-3)

    for epoch in range(15):
        net.train()
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)  
            l = loss_fn(net(X), y)
            updater.zero_grad()
            l.backward()
            updater.step()
        acc = accuracy(net, test_iter, device)
        print(f"epoch {epoch+1}: test acc {acc:.3f}")