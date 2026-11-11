import torch
from torch import nn
from d2l import torch as d2l

def accuracy(net , data_iter):
    correct,total =0,0
    with torch.no_grad():
        for X,y in data_iter:
            pred = net(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
        return correct / total

if __name__ == "__main__":
    batch_size =256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,256),nn.ReLU(),
        nn.Linear(256,10),
    )
    loss_fn = nn.CrossEntropyLoss()
    updater = torch.optim.SGD(net.parameters(), lr=0.1)
    for epoch in range(10):
        for X,y in train_iter:
            l = loss_fn(net(X),y)
            updater.zero_grad()
            l.backward()
            updater.step()
        acc = accuracy(net, test_iter)
        print(f"epoch {epoch + 1}:test acc {acc:.3f}")
