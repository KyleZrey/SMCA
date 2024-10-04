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
        # Apply the sigmoid activation function !!!DONT USE WITH BCEWithLogits!!!
        # x = torch.sigmoid(x)
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
        # Initialize BCEWithLogitsLoss, which combines Sigmoid + BCELoss
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        # Compute and return the binary cross-entropy loss with logits
        return self.bce_loss(output, target)

class CustomLoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_f1=0.5):
        super(CustomLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.weight_bce = weight_bce
        self.weight_f1 = weight_f1

    def forward(self, outputs, targets):
        # Binary Cross-Entropy Loss
        bce_loss = self.bce_loss(outputs, targets)

        # Calculate precision, recall, and F1 score
        pred_labels = (outputs > 0.5).float()  # Assuming a threshold of 0.5
        tp = (pred_labels * targets).sum().to(torch.float32)
        tn = ((1 - pred_labels) * (1 - targets)).sum().to(torch.float32)
        fp = (pred_labels * (1 - targets)).sum().to(torch.float32)
        fn = ((1 - pred_labels) * targets).sum().to(torch.float32)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Loss is weighted sum of BCE loss and (1 - F1) to minimize both
        total_loss = self.weight_bce * bce_loss + self.weight_f1 * (1 - f1)

        return total_loss

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