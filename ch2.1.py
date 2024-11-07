import os
import urllib.request
import re
import importlib
import tiktoken

from importlib.metadata import version

tokenizer = tiktoken.get_encoding("gpt2")

test = (
    "Akwirw ier"
)

integers = tokenizer.encode(test)


#Print all individual tokens
print("Individual Token IDs: ",integers) #encodes

#Print the set {Token : Decoded word}
for num in integers:
    print(f"{num} -> {tokenizer.decode([num])}") #create list of decoded integers


#Reconstruct the word by decoding the decoded tokens
print("Decoded: ")
decoded = []

for num in integers:
    decoded.append(num)

print(tokenizer.decode(decoded))