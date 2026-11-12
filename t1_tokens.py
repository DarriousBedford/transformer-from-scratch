import torch

# 1. Load the text

with open("data/input.txt", encoding = 'utf-8') as f:
    text = f.read()

print(len(text)) # total number of characters)
print(text[:200]) #first 200 characters of the texts

# 2. Build the vocabulary
chars = sorted(list(set(text))) # unique characters in the text
vocab_size = len(chars) # size of the vocabulary
print(chars)
print(vocab_size)

# 3. Lookup tables
stoi = {ch: i for i, ch in enumerate(chars)}     # "string to integer"
itos = {i: ch for i, ch in enumerate(chars)}     # "integer to string"

# 4. Encoder and decoder
def encode(s):
    """Turn a string into a list of integers."""
    return [stoi[c] for c in s]

def decode(ids):
    """Turn a list of integers back into a string."""
    return ''.join([itos[i] for i in ids])

# 5. Test the round trip
print(encode("helo"))
print(decode(encode("helo")))

# 6. Encode the whole dataset as a PyTorch tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.type()) # shape and data type of the tensor
print(data[:100]) # first 100 elements of the tensor

# 7. Train / validation split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(len(train_data), len(val_data))