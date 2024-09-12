import torch
import torch.nn as nn
import os 

class CrossAttention(nn.Module):
    def __init__(self, d_in_1, d_in_2, d_out_kq, d_out_v):
        super(CrossAttention, self).__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Parameter(torch.rand(d_in_1, d_out_kq))
        self.W_key = nn.Parameter(torch.rand(d_in_2, d_out_kq))
        self.W_value = nn.Parameter(torch.rand(d_in_2, d_out_v))
    
    def forward(self, modality1, modality2):
        
        queries = modality1.matmul(self.W_query)  
        keys = modality2.matmul(self.W_key)      
        values = modality2.matmul(self.W_value) 

        attn_scores = queries.matmul(keys.transpose(-2, -1))
        attn_weights = torch.softmax(attn_scores / (self.d_out_kq ** 0.5), dim=-1)
        context_vector = attn_weights.matmul(values) 
        return context_vector
    