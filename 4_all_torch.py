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


X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)

print(f"Before Training: f(5) = {model(X_test).item():.3f}")

learning_rate = 0.01
n_iters = 50
# Prediction

loss = nn.MSELoss()
optmizer = torch.optim.SGD(model.parameters(),lr=learning_rate)



# Training
for epoch in range(n_iters):
    # prediction --> Forward pass
    y_pred = model(X)
    # loss
    l = loss(Y, y_pred)
    # grad = backward pass
    l.backward()
    optmizer.step()    
    optmizer.zero_grad()


    if epoch % 1 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item:.3f} : loss = {l:.4f}')
        pass

print(f"After Training: f(5) = {forward(5):.3f}")
