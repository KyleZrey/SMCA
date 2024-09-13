import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, input_size):
        super(DenseLayer, self).__init__()
        # Define a dense layer (fully connected layer) with 2 units
        self.fc = nn.Linear(input_size, 1)  # input_size -> number of input features, 2 -> number of units

    def forward(self, x):
        # Pass the input through the dense layer
        x = self.fc(x)
        # Apply the sigmoid activation function
        x = torch.sigmoid(x)
        return x
    
class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        # Initialize BCELoss
        self.bce_loss = nn.BCELoss()

    def forward(self, output, target):
        # Compute and return the binary cross-entropy loss
        return self.bce_loss(output, target)

# # Example usage:
# input_size = 5
# model = DenseLayer(input_size)

# # Instantiate the custom BCELoss class
# criterion = BCELoss()

# # Create a random input tensor
# input_tensor = torch.randn(1, input_size)

# # Define a target tensor with binary labels
# target = torch.tensor([[1.0]])  # Target must be a float tensor in [0, 1] range

# # Forward pass
# output = model(input_tensor)

# # Compute loss using the custom BCELoss class
# loss = criterion(output, target)

# print("Output:", output)
# print("Loss:", loss.item())