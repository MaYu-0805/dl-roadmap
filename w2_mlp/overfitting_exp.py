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

def train_and_eval(net, train_iter,test_iter,tag,epochs=10):
    updater = torch.optim.SGD(net.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for X,y in train_iter:
            l =loss_fn(net(X),y)
            updater.zero_grad()
            l.backward()
            updater.step()
        print(f"[{tag}] epoch {epoch + 1}: tarin {accuracy(net, train_iter):.3f} "
              f"test {accuracy(net, test_iter):.3f}")

if __name__ == "__main__":
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    small_train = torch.utils.data.DataLoader(
        list(train_iter.dataset)[:6000], batch_size=batch_size, shuffle=True)

    net_over = nn.Sequential(nn.Flatten(),
                             nn.Linear(784, 512), nn.ReLU(),
                             nn.Linear(512, 10))
    train_and_eval(net_over, small_train, test_iter, "过拟合")