import torch
import matplotlib.pyplot as plt

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)

def plot_fn(fn, name):
    y = fn(x)
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.plot(x.detach(), y.detach()); plt.title(name)
    plt.subplot(1, 2, 2)
    y.sum().backward()          
    plt.plot(x.detach(), x.grad); plt.title(f"{name}' derivative")
    plt.show()
    x.grad.zero_()              

plot_fn(torch.relu, 'ReLU')
plot_fn(torch.sigmoid, 'Sigmoid')
plot_fn(torch.tanh, 'Tanh')
