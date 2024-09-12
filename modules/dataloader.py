import torch
import numpy as np
import os
import torch.nn as nn

def load_npy_files(directory):
    file_list = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.npy')]
    feature_vectors = [(file, torch.tensor(np.load(file))) for file in file_list]
    return feature_vectors