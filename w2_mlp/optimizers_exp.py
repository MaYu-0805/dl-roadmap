import torch
from torch import nn
from d2l import torch as d2l


def accuracy(net, data_iter):
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            pred = net(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


def make_net():
    return nn.Sequential(nn.Flatten(),
                         nn.Linear(784, 256), nn.ReLU(),
                         nn.Linear(256, 10))


def train_with(net, updater, tag, epochs=10):
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for epoch in range(epochs):
        for X, y in train_iter:
            l = loss_fn(net(X), y)
            updater.zero_grad()
            l.backward()
            updater.step()
            losses.append(l.item())       
        print(f"[{tag}] epoch {epoch+1}: test acc {accuracy(net, test_iter):.3f}")
    return losses


if __name__ == "__main__":
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    curves = {}
    net1 = make_net()
    curves["SGD"] = train_with(net1, torch.optim.SGD(net1.parameters(), lr=0.1), "SGD")

    net2 = make_net()
    curves["动量"] = train_with(net2, torch.optim.SGD(net2.parameters(), lr=0.1, momentum=0.9), "Momentum")

    net3 = make_net()
    curves["Adam"] = train_with(net3, torch.optim.Adam(net3.parameters(), lr=0.001), "Adam")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    for name, losses in curves.items():
        plt.plot(losses, label=name, alpha=0.7)
    plt.legend(); plt.xlabel("batch"); plt.ylabel("loss")
    plt.title("Optimizer comparison")
    plt.savefig("optimizer_curves.png", dpi=150)
    plt.show()
