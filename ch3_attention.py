import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec

# Causal attention mechanism is added
class SelfAttention_v3(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T

        mask = torch.triu(torch.ones(attn_scores.shape), diagonal = 1)
        masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
        #print("masked : \n", masked)

        attn_weights = torch.softmax(
            masked / keys.shape[-1]**0.5, dim=-1
        )
        #print("Attention weights with causal attention \n : ", attn_weights)
        context_vec = attn_weights @ values
        return context_vec  


# Causal attention + dropouts mechanisms 
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        print("Register mask with shape : ", self.mask.shape)       
        
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1,2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        
        context_vec = attn_weights @ values
        return context_vec  

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, 
                dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(
                d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.d_out = d_out 
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), 
            diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # add a new dimension for heads
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        
        # num_heads before num_tokens
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        attn_scores = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1,2)
        
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        context_vec = self.out_proj(context_vec)
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
#print(attn_score_2)
#print("\nAttention weight shape : {}".format(attn_score_2.shape))

# Compute context vectors
context = attn_score_2 @ inputs
#print("Context matrix : \n{}".format(context))
#print("Context shape : {}".format(context.shape))

# use class SelfAttention
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(3,2)
#print("forward self action class v2: \n", sa_v2(inputs))

# Exercise 3.1
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(3,2)
sa_v1.W_key.data = sa_v2.W_key.weight.T
sa_v1.W_value.data = sa_v2.W_value.weight.T
sa_v1.W_query.data = sa_v2.W_query.weight.T
#print("forward self action class v1 : \n", sa_v1(inputs))

# Causal attention
torch.manual_seed(789)
sa_v3 = SelfAttention_v3(3,2)
#print("forward self action class v3: \n", sa_v3(inputs))

# Causal + Dropouts attention + batch
d_in = 3
d_out = 2
batch = torch.stack((inputs, inputs), dim=0)
#print("Shape after stacking inputs : ", batch.shape)
##context_length = batch.shape[1]
##torch.manual_seed(123)
##ca = CausalAttention(d_in,d_out, context_length, 0.0)
##context_vecs = ca(batch)
#print("context_vecs.shape :", context_vecs.shape)

# Multi-head
torch.manual_seed(789)
context_length = batch.shape[1]
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, 2
)
context_vecs = mha(batch)
print("context_vecs : \n", context_vecs)
print("context_vecs.shape : \n", context_vecs.shape)

# Exercice 3.2
# d_out set to 1 to have only two-dimensional vectors
d_in, d_out = 3, 1
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, 2
)
context_vecs = mha(batch)
print("context_vecs 3.2 : \n", context_vecs)
print("context_vecs.shape 3.2 : \n", context_vecs.shape)

# use MultiHeadAttentionClass
torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print("MultiheadAttentioClass\n", context_vecs)
print("MultiheadAttentioClass context_vecs.shape", context_vecs.shape)

# Exercise 3.3
# 12 attention heads, input and output embeddings size : 768

inputs = torch.rand((6, 768))
batch = torch.stack((inputs, inputs), dim=0)
batch_size, context_length, d_in = batch.shape
d_out = 768
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=12)
context_vecs = mha(batch)

print("MultiheadAttentioClass ex 3.3 context_vecs.shape", context_vecs.shape)
