import os
import urllib.request
import re
import tiktoken
import importlib
import torch

#Chapter 3

#Attention Mechanisms


#Unnormalized attention weights = attention scores
#Normalized attention scores (Σ = 1) = attention weights.

#Compute unnormalized attention score ω:
#Using the second input token (as an example) such that q^(2) = x^(2),
#We can compute unnormalized attention scores using dot products.

#from x(1)..x(T)
#ω21 = x(1)q(2)
#ω22 = x(2)q(2)
#...
#ω2T = x(T)q(2)

#In ω's subscript, we get 2X.
#For example: ω21
#The 2 represents the input sequence element 2 used as a query against
#input sequence element 1.

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1]  # 2nd input token is the query

attn_scores_2 = torch.empty(inputs.shape[0]) #attn_scores_2 =
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # dot product (transpose not necessary here since they are 1-dim vectors)

print(attn_scores_2)

#a dot product is essentially a shorthand for
#multiplying two vectors elements-wise and summing the resulting products:
#attention_score = sum(Q[i] * K[i] for i in range(d)) d=dimension, Q=Query, K=Key

res = 0.

for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * query[idx] #Res here is the dot product. Input[0] = d

print(res)
print(torch.dot(inputs[0], query))

#torch.dot can do the equations for us, given a dimension inputs[0].

#Now that our unnormalized attention scores (w, omega) are computed, lets
#normalize them to sum to 1.

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum() #Takes all 6 rows, divide by total sum

print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

print("\n Using soft max:")

#Instead, we use torches softmax function

#-------------------

#Heres a naive implementation:
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)

print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

#-------------------

#Here is the torch function in use:
print("\n")
print("We use the actual torch softmax function for more extreme values")
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())


#Now we have computed the sum to 1.
#Step 3:  compute the context vector z(2) by multiplying the embedded input tokens,
#x(i) with the attention weights and sum the resulting vectors:

print("\n")


query = inputs[1] # 2nd input token is the query

context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i

print(context_vec_2)

#Apply previous step 1 to all pairwise elements to compute the unnormalized attention score matrix:

print("\n")

attn_scores = torch.empty(6, 6)

attn_scores = inputs @ inputs.T #The @ operator is shorthand for matrix multiplication in PyTorch
print(attn_scores)

#Normalize the rows to sum to 1
print("\n")

attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)


all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

print("Previous 2nd context vector:", context_vec_2)

#3.4

#Implementing the self-attention mechanism step by step, we will start by introducing the three training
#weight matrics: w(q), w(k), w(v)
#q = Query
#k = Key
#v = Value

##These three matrices are used to project the embedded input tokens, x^(i), into query, key, and value vectors via matrix multiplication:
#Query vector: q^(i) = W(q) x^(i)
#Key vector: k^(i) = W(q)x^(i)
#Value vector: v^(i) = W(q)x^(i)

x_2 = inputs[1] # second input element
d_in = inputs.shape[1] # the input embedding size, d=3
d_out = 2 # the output embedding size, d=2

torch.manual_seed(123) ##sets our torch.rand() seet

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query # _2 because it's with respect to the 2nd input element
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print(query_2)
print(key_2)
print(value_2)

keys = inputs @ W_key
values = inputs @ W_value

print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

keys_2 = keys[1] # Python starts index at 0
attn_score_22 = query_2.dot(keys_2) # Dot product of query x key
print("Dot product of Query x Key vector:", attn_score_22)

attn_scores_2 = query_2 @ keys.T # All attention scores for given query. Transpose of keys tensor((6,2) == (2,6))
print(attn_scores_2)

#The difference to earlier is that we now scale the attention scores by dividing them by the square root of the embedding dimension

#embedding dimension = keys.shape[1] OR 2. (6,2)

d_k = keys.shape[1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)
print("After using softmax and normalzing it by dividing by the sqrt of embedding dimension, we get: ",attn_weights_2.sum())

#we now compute the context vector for input query vector 2:

context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

import torch.nn as nn


class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T  # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec


torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))


#2

class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec


torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))
"""
Note that nn.Linear in SelfAttention_v2 uses a different weight initialization
scheme as nn.Parameter(torch.rand(d_in, d_out)) used in SelfAttention_v1,
which causes both mechanisms to produce different results. To check that both
implementations, SelfAttention_v1 and SelfAttention_v2, are otherwise similar, 
we can transfer the weight matrices from a SelfAttention_v2 object to a SelfAttention_v1, 
such that both objects then produce the same results.

Your task is to correctly assign the weights from an instance of SelfAttention_v2
to an instance of SelfAttention_v1. To do this, you need to understand the relationship between the weights in both versions. 
(Hint: nn.Linear stores the weight
matrix in a transposed form.) After the assignment, you should observe that both
instances produce the same outputs.
"""

sa_v1.W_query = nn.Parameter(sa_v2.W_query.weight.T)
sa_v1.W_key = nn.Parameter(sa_v2.W_key.weight.T)
sa_v1.W_value = nn.Parameter(sa_v2.W_value.weight.T)
print(sa_v1(inputs))
print(sa_v2(inputs))

"""
End of exercise 3.1
"""

# Reuse the query and key weight matrices of the
# SelfAttention_v2 object from the previous section for convenience
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T

attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)

"""
The simplest way to mask out future attention weights is by creating a mask via PyTorch's tril
 function with elements below the main diagonal (including the diagonal itself) 
 set to 1 and above the main diagonal set to 0:
"""

context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)

masked_simple = attn_weights*mask_simple
print(masked_simple)

row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)


"""
Lets make this more effecient
"""

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1) ##triu = upper triangular matrix
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)

batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape) # 2 inputs with 6 tokens each, and each token has embedding dimension 3

class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # New
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New

    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2) # Changed transpose
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights) # New

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)

context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)

context_vecs = ca(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


torch.manual_seed(123)

context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


torch.manual_seed(123)

batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


"""
Exercise 3.2 Returning two-dimensional embedding vectors


Change the input arguments for the MultiHeadAttentionWrapper(..., num_
heads=2) call such that the output context vectors are two-dimensional instead of
four dimensional while keeping the setting num_heads=2. Hint: You don’t have to
modify the class implementation; you just have to change one of the other input
arguments.
"""

torch.manual_seed(123)

context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 1 ## HERE IS THE SOLUTION: We change d_out to one, so that we are back to single-head attention.
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

"""
END OF EXERCISE 3.2
"""

"""
Exercise 3.3 Initializing GPT-2 size attention modules

Using the MultiHeadAttention class, initialize a multi-head attention module that
has the same number of attention heads as the smallest GPT-2 model (12 attention
heads). Also ensure that you use the respective input and output embedding sizes
similar to GPT-2 (768 dimensions). Note that the smallest GPT-2 model supports a
context length of 1,024 tokens.
"""

context_length = 1024
d_in, d_out = 768, 768
num_heads = 12

mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads)


"""
END OF CHAPTER 3
"""





























