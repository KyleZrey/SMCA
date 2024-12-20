import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, input_size):
        super(DenseLayer, self).__init__()
        # Define a dense layer (fully connected layer) with 2 units
        self.fc = nn.Linear(input_size, 1)  # input_size -> number of input features, 2 -> number of units

    def forward(self, x):
        # Pass the input through the dense layer
        x = self.fc(x)
        # Apply the sigmoid activation function !!!DONT USE WITH BCEWithLogits!!!
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
    
class BCEWithLogits(nn.Module):
    def __init__(self):
        super(BCEWithLogits, self).__init__()
        # Define pos_weight to balance the true (positive) class
        pos_weight = torch.tensor([2.94])
        # Initialize BCEWithLogitsLoss, which combines Sigmoid + BCELoss
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        # Compute and return the binary cross-entropy loss with logits
        return self.bce_loss(output, target)

class CustomLoss(nn.Module):
    def __init__(self, pos_weight=2.94, margin=1.0):
        super(CustomLoss, self).__init__()
        # Initialize the positive weight and margin
        self.pos_weight = pos_weight
        self.margin = margin

    def forward(self, output, target):
        # Convert target to -1 or 1
        target = target * 2 - 1  # Assuming target is [0, 1], convert to [-1, 1]
        
        # Compute hinge loss
        hinge_loss = F.relu(self.margin - target * output)
        
        # Calculate weights based on the target
        weights = torch.where(target == 1, self.pos_weight, 1.0)  # Weight for positive class
        
        # Apply weights to the hinge loss
        weighted_loss = weights * hinge_loss
        
        # Return the mean of the weighted loss
        return weighted_loss.mean()

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