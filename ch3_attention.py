import torch
import torch.nn as nn

class selfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec
        


inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],
     [0.57, 0.85, 0.64],
     [0.22, 0.58, 0.33],
     [0.77, 0.25, 0.10], 
     [0.05, 0.80, 0.55]]
)
print("Inputs shape : {}".format(inputs.shape))
# Dot products between the inputs
attn_score_2 = torch.empty(inputs.shape)
attn_score_2 = inputs @ inputs.T

# Apply softmax on the last dimension of the matrix to obtain attention weights
attn_score_2 = torch.softmax(attn_score_2, dim=-1)
print(attn_score_2)
print("\nAttention weight shape : {}".format(attn_score_2.shape))
# Compute context vectors
context = attn_score_2 @ inputs
print("Context matrix : \n{}".format(context))
print("Context shape : {}".format(context.shape))

# use class SelfAttention
torch.manual_seed(123)
sa_v1 = selfAttention_v1(3,2)
print("forward self action class: \n", sa_v1(inputs))