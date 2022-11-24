"""
Prediction --> Manual
Grad Compute --> Pytorch
Loss Compute --> Manual
Param Update --> Manual
"""
import numpy as np
import torch

X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# Prediction

def forward(x):
    return w*x

# Loss (Mean Squared Errror)
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()


print(f"Before Training: f(5) = {forward(5):.3f}")

# Training
lr = 0.01
n_iter = 50

for epoch in range(n_iter):
    # prediction --> Forward pass
    y_pred = forward(X)
    # loss
    l = loss(Y, y_pred)
    # grad = backward pass
    l.backward()
    # update weights
    with torch.no_grad():
        w -= lr*w.grad
    
    w.grad.zero_()


    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f} : loss = {l:.4f}')
        pass

print(f"After Training: f(5) = {forward(5):.3f}")
