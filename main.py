import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import FCN
from utils import oscillator

d=2 #the damping ratio.
w0=20 #the natural frequency of the undamped system.
# get the analytical solution over the full domain
x = torch.linspace(0,1,1000).view(-1,1) #creates a 1-dimensional tensor with 500 steps linearly spaced between 0 and 1. view(-1, 1) reshapes the tensor to have 1000 rows and 1 column, which is required by the oscillator function.
y = oscillator(d, w0, x).view(-1,1)
print(x.shape, y.shape)
x_data = x[0:1000:20]
y_data = y[0:1000:20]
print(x_data.shape, y_data.shape)

plt.figure()
plt.plot(x, y, label="Exact solution")
plt.scatter(x_data, y_data, color="tab:orange", label="Training data")
plt.legend()
plt.title("Exact Solution vs Training Data")
plt.xlabel("Time (t)")
plt.ylabel("Displacement (x)")
plt.show()

# train standard neural network to fit training data
  

# Define sampling locations over the problem domain for computing the physics loss
x_physics = torch.linspace(0, 1, 30).view(-1, 1).requires_grad_(True)

# Define parameters for the damped harmonic oscillator
mu, k = 2 * d, w0 ** 2

# Set the random seed for reproducibility
torch.manual_seed(123)

# Initialize the fully connected neural network model
model = FCN(1, 1, 32, 3)

# Define the optimizer with learning rate 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Initialize a list to store file paths of generated plots (if any)
files = []

# Train the model for 20000 iterations
for i in range(20000):
    optimizer.zero_grad()  # Clear the gradients of all optimized tensors

    # Compute the "data loss" using mean squared error between model predictions and actual data
    yh = model(x_data)
    loss1 = torch.mean((yh - y_data) ** 2)

    # Compute the "physics loss" to ensure the solution satisfies the underlying differential equation
    yhp = model(x_physics)
    dx = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0]  # Compute dy/dx
    dx2 = torch.autograd.grad(dx, x_physics, torch.ones_like(dx), create_graph=True)[0]  # Compute d^2y/dx^2
    physics = dx2 + mu * dx + k * yhp  # Compute the residual of the 1D harmonic oscillator differential equation
    loss2 = (1e-4) * torch.mean(physics ** 2)  # Scale the physics loss

    # Backpropagate the joint loss (data loss + physics loss)
    loss = loss1 + loss2
    loss.backward()
    optimizer.step()  # Update model parameters
    
   