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

