"""
Prediction --> Manual
Grad Compute --> Pytorch
Loss Compute --> Pytorch
Param Update --> Pytorch
"""

"""
1. Designing the Model --> Input and Output size, Forward Pass.
2. Construct the loss and optimizer.
3. Training Loop.
    - Forward pass --> Compute the prediction
    - Backward Pass --> Gradients
    - Update weights
"""

import torch
import torch.nn as nn


X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

learning_rate = 0.01
n_iters = 50
# Prediction

def forward(x):
    return w*x

loss = nn.MSELoss()
optmizer = torch.optim.SGD([w],lr=learning_rate)

print(f"Before Training: f(5) = {forward(5):.3f}")

# Training
for epoch in range(n_iters):
    # prediction --> Forward pass
    y_pred = forward(X)
    # loss
    l = loss(Y, y_pred)
    # grad = backward pass
    l.backward()
    optmizer.step()    
    optmizer.zero_grad()


    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f} : loss = {l:.4f}')
        pass

print(f"After Training: f(5) = {forward(5):.3f}")
