import torch
import torch.nn as nn

class FCN(nn.Module):
    "Defines a fully connected neural network"
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        
        # Activation function to be used in hidden layers
        activation = nn.Tanh
        
        # Define the first layer (input to first hidden layer)
        self.fcs = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()
        )
        
        # Define the hidden layers
        self.fch = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation()
            ) for _ in range(N_LAYERS - 1)
        ])
        
        # Define the last layer (last hidden layer to output)
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
    def forward(self, x):
        # Pass through the first layer
        x = self.fcs(x)
        
        # Pass through the hidden layers
        x = self.fch(x)
        
        # Pass through the output layer
        x = self.fce(x)
        
        return x

