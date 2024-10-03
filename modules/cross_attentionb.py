import torch
import torch.nn as nn

class CrossAttentionB(nn.Module):
    def __init__(self, d_in_query, d_in_kv, d_out_kq, d_out_v):
        super(CrossAttentionB, self).__init__()
        self.d_out_kq = d_out_kq

        self.W_query = nn.Linear(d_in_query, d_out_kq)
        self.W_key = nn.Linear(d_in_kv, d_out_kq)
        self.W_value = nn.Linear(d_in_kv, d_out_v)

        # self.layer_norm = nn.LayerNorm(d_out_v)

        # self.dropout = nn.Dropout(0.1)

    def forward(self, query, key_value):
        queries = self.W_query(query)
        keys = self.W_key(key_value)
        values = self.W_value(key_value)

        attn_scores = queries.matmul(keys.transpose(-2, -1))
        attn_weights = torch.softmax(attn_scores / (self.d_out_kq ** 0.5), dim=-1)
        context_vector = attn_weights.matmul(values)

        # attn_scores = torch.matmul(queries, keys.transpose(-2,-1))
        # attn_scores = attn_scores / (self.d_out_kq ** 0.5)

        # attn_weights = torch.softmax(attn_scores, dim=-1)

        # attn_weights = self.dropout(attn_weights)

        # context_vector = torch.matmul(attn_weights, values)

        # context_vector = self.layer_norm(context_vector)
        
        return context_vector