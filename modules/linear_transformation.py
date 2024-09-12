# Linear transformation to match dimensions
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F

class LinearTransformations(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearTransformations, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)