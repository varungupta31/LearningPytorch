"""
Prediction --> Manual
Grad Compute --> Manual
Loss Compute --> Manual
Param Update --> Manual
"""
import numpy as np

X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)
# X = np.array([1,4,16,48], dtype=np.float32)
# Y = np.array([2,8,32,96], dtype=np.float32)

w = 0.0

# Prediction

def forward(x):
    return w*x

# Loss (Mean Squared Errror)
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

# Gradients
def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()


print(f"Before Training: f(5) = {forward(5):.3f}")

# Training
lr = 0.01
n_iter = 100

for epoch in range(n_iter):
    # prediction --> Forward pass
    y_pred = forward(X)
    # loss
    l = loss(Y, y_pred)
    # grad
    dw = gradient(X, Y, y_pred)
    # update weights
    w -= lr*dw

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f} : loss = {l:.4f}')
        pass

print(f"After Training: f(5) = {forward(5):.3f}")
