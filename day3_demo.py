import torch
torch.manual_seed(0)

X=torch.rand(200,1)*10
y=3*X+2+torch.randn(200,1)

w=torch.zeros(1,requires_grad=True)
b=torch.zeros(1,requires_grad=True)
lr=0.01

for epoch in range(500):
    pred=X*w+b
    loss=((pred-y)**2).mean()
    loss.backward()
    with torch.no_grad():
        w-=lr*w.grad
        b-=lr*b.grad
        w.grad.zero_();b.grad.zero_()
    if epoch%100==0:
        print(f"epoch={epoch} loss={loss.item():.4f} w={w.item():.3f} b={b.item():.3f}")

print("训练结束,w应接近3,b应接近2")