import torch
from torch import nn
from d2l import torch as d2l
def relu(X):
    return torch.max(X, torch.zeros_like(X))

def net(X):
    X= X.reshape((-1,num_inputs))
    H=relu(X@W1 + b1)
    return H@W2 + b2

def accuracy(net, data_iter):
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            pred = net(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total

if __name__ == "__main__":
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    num_inputs,num_hiddens, num_outputs = 784, 256,10

    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) *0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens))
    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) *0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs))
    params=[W1,b1,W2,b2]

    loss_fn=nn.CrossEntropyLoss()
    Ir,num_epochs=0.1,5

    updater=torch.optim.SGD(params, lr=Ir)

    for epoch in range(num_epochs):
        for X,y in train_iter:
            y_hat=net(X)
            l=loss_fn(y_hat,y)
            updater.zero_grad()
            l.backward()
            updater.step()

        print(f"测试集准确率: {accuracy(net, test_iter):.3f}")
