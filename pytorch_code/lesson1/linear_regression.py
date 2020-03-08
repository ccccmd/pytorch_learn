#-*-coding:utf-8-*-
import torch
import matplotlib.pyplot as plt
torch.manual_seed(10)
lr = 0.05

# 创建训练数据
x = torch.rand(20, 1) * 10      # x shape=(20, 1)
y = 2*x + (5 + torch.randn(20, 1))   # y shape=(20,1)

# 构建线性回归参数
w = torch.randn((1), requires_grad=True)
b = torch.zeros((1), requires_grad=True)

for iteration in range(1000):
    # forward
    wx = torch.mul(w,x)
    y_pred = torch.add(wx,b)

    # MSE loss
    loss = (0.5 * (y - y_pred) ** 2).mean()

    # backward
    loss.backward()

    # update
    b.data.sub_(lr * b.grad)
    w.data.sub_(lr * w.grad)

    # 清零张量梯度
    w.grad.zero_()
    b.grad.zero_()

    #绘图
    if iteration % 20 == 0:
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
        plt.text(2, 20, 'Loss=%.4f'% loss.data.numpy(), fontdict={'size':20, 'color': 'red'})
        plt.xlim(1.5,10)
        plt.ylim(8,28)
        plt.title("Iteration: {}\nw: {} b: {}".format(iteration, w.data.numpy(), b.data.numpy()))
        plt.pause(0.5)
        if loss.data.numpy() < 1:
            break
