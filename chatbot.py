# We will first import a few code cells from the "Bi-Gram" model python notebook

import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import random
import mmap
import argparse

# Parsing command line arguments
parser = argparse.ArgumentParser(description = "This is a demonstration program")
# Here we add an argument to the parser, specifying the expected type, a help message, etc.
parser.add_argument('-batch_size', type = str, required = True, help = 'Please provide a batch size')

args = parser.parse_args()

# Now we can use the argument value in our program
print(f"The provided line is : {args.batch_size}")

# Changing Device : CPU --> GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

'''
Hyperparameters 🌟🔽

You can make use of auto-tuning, i.e., make use of
different set of values for each hyperparameter
and put that to the model - to check which ones give the best result
'''

block_size = 32
# batch_size = 128
batch_size = args.batch_size # Argument from command line
max_iters = 200
learning_rate = 3e-4
eval_iters = 100
n_embd = 384 # Each embedding vector will be of 384 characters long
n_layer = 1
n_head = 1
dropout = 0.2 # 20% of neurons will be set to '0'

'''
The amount of Memory (Say, GPU Memory) used for the model depends
on the below parameters 🔽
- block_size
- batch_size
- n_embd
- n_layer
- n_head
'''

# Opening the text file (vocab.txt)
chars = ""

with open("D:/Datasets/vocab.txt", "r", encoding = 'utf-8') as f: # Character encoding = 'utf-8'
    text = f.read()
    chars = sorted(list(set(text)))

vocab_size = len(chars)

# Encode - Decode

string_to_int = {ch:i for i,ch in enumerate(chars)} # Dictionary of encoded character values
int_to_string = {i:ch for i,ch in enumerate(chars)} # Dictionary of decoded character values

encode = lambda s : [string_to_int[c] for c in s]
decode = lambda l : ''.join([int_to_string[i] for i in l])

# data = torch.tensor(encode(text), dtype = torch.long)

'''
Validaton and Training Splits
'''

# n = int(0.8*len(data)) # Training Data Size
# train_data = data[:n]
# val_data = data[n:]

'''
Memory Mapping 🔽
Memory mapping is a technique where a file or device is mapped into the virtual memory space of a process, allowing the file 
to be accessed directly as if it were part of the system's memory. This enables faster file I/O by avoiding traditional 
read/write operations and provides more efficient access to large files.
'''


# def get_batch(split):
#     data = train_data if split == 'train' else val_data
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     # print(ix)
#     X = torch.stack([data[i:i+block_size] for i in ix])
#     y = torch.stack([data[i+1:i+block_size+1] for i in ix])
#     X, y = X.to(device), y.to(device) # Putting the data components in currently selected device (here, GPU)
#     return X, y



'''
Creating the GPT Language Model - Following the Transformer Architecture
Paper LINK : https://arxiv.org/abs/1706.03762
'''

class Head(nn.Module):
    """
    One Head of Self-Attention
    """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False) # K
        self.query = nn.Linear(n_embd, head_size, bias = False) # Q
        self.value = nn.Linear(n_embd, head_size, bias = False) # V
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # This to reduce some over-head computation (avoiding re-do)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input of size (B = Batch, T = Time-Step / Sequence, C = Channels)
        # Output of size (B = Batch, T = Time-Step / Sequence, C = head_size)
        B, T, C = x.shape
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Computing Attention Scores ("Affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, head_size) @ (B, head_size, T) --> (B, T, T)
        
        tril = torch.tril(torch.ones(T, T, device=x.device))  # Adjusting the mask dynamically
        wei = wei.masked_fill(tril == 0, float('-inf'))

        '''
        In the above 2 lines 🔻
        Masking : Adjusted the triangular mask in the 
        Head class to match the sequence length dynamically.
        '''

        wei = F.softmax(wei, dim = -1) # (B, T, T) --> if one value is big / dominant over others (in magnitude), Softmax will make it stand out (also replacing '-inf' with zeroes)
        wei = self.dropout(wei)

        # Performing the weighted aggregation of the values
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) --> (B, T, head_size)
        return out


'''
Masked Inputs (with '-inf' (here)) - not revealing the later input values in the present time-steps🔻

t=1 --> [1, -inf, -inf]
t=2 --> [1, 0.8, -inf] (revealing 0.9 in t=2)
t=3 --> [1, 0.8, 0.33] (revealing 0.33 in t=3)

Masking is important as it will not allow the model to learn the later values in present. 
The values revealed at every time-steps can be tokens
'''


class MultiHeadAttention(nn.Module):
    """
    Multiple Heads of Self-Attention in Parallel
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range (num_heads)]) # Multiple Heads set to work in parallel (head_size = 4 --> 4 heads running in parallel)
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = 2) # Concatenating each head to the last dimension (B, T, F) --> F = Feature Dimension
        # (B, T, F) --> (B, T, [h1, h1, h1, h1, h2, h2, h2 ,h2, h3, h3, h3, h3]) ==> 4 features for each head (head_size = 3 (say))
        out = self.dropout(self.proj(out))
        return out



class FeedForward(nn.Module):
    """
    A simple linear layer followed by a non-linear layer
    """
    def __init__ (self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd , n_embd), # [n_embd X 4*n_embd] X [4*n_embd X n_embd] = [n_embd X n_embd]
            nn.Dropout(dropout), # Dropout --> to prevent over-fitting
        )

    def forward(self, x):
        return self.net(x)



class Block(nn.Module):
    """
    Transformer Block : Communication Followed by Computation
    """
    def __init__(self, n_embd, n_head):
        # n_embd : Embedding Dimension, n_head ==> the no. of heads we'd like
        super().__init__() # n_head ==> no. of heads
        head_size = n_embd // n_head # No. of features caputured by each head during Mutli-Head Attention ==> head_size
        self.sa = MultiHeadAttention(n_head, head_size) # sa ==> self-attention
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x) # Self-Attention
        x = self.ln1(x+y) # Adding Norm (Layer Norm)
        y = self.ffwd(x) # Feed-Forward
        x = self.ln2(x+y) # Adding Norm (Layer Norm)
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # Embedding Matrix
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # Positional Embedding Matrix

        # Decoder Blocks - Running sequentially 🔽
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)]) # n_layers --> these many layers of Decoders will be present

        self.ln_f = nn.LayerNorm(n_embd) # Final Layer Normalization
        self.lm_head = nn.Linear(n_embd, vocab_size) # Language Model Head

        self.apply(self._init_weights) # Initializing weights around certain Standard Deviations

    def _init_weights(self, module): # Function for Initializing weights around certain Standard Deviations
        if isinstance (module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance (module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        # logits = self.token_embedding_table(index) # logits is 3-dimensional
        # B, T, C = logits.shape
        B, T = index.shape

        '''
        B (Batch Size) : Number of sequences or data samples processed in parallel.
        T (Time-Step/Sequence) : Number of time steps (or sequence length) for each input sample.
        C (Channels) : Number of features or channels for each time step (such as the depth or dimensionality of the input at each time step).
        '''

        # idx and target are both (B, T) tensor in integers
        tok_emb = self.token_embedding_table(index) # (B, T, C)
        
        T = index.shape[1]

        '''
        In the above line 🔻
        Positional Embeddings : Ensured T is consistently 
        defined based on index.shape[1] for sequence length.
        '''

        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)
 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # logits.view(a, b) ==> a = batch size ; b = no. of classes
            targets = targets.view(B*T) # targets.view(a) ==> a = no. of classes
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range (max_new_tokens):
            # Cropping idx to the last block_size token(s)
            
            # Ensuring we don't slice too much if the sequence is shorter than block_size
            if index.shape[1] > block_size:
                index_cond = index[:, -block_size:]
            else:
                index_cond = index

            # Getting the predictions
            logits, loss = self.forward(index_cond)

            # Focussing only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            
            # Applying Softmax to get probabilities
            probs = F.softmax(logits, dim = -1) # (B, C) ==> dim = -1 (as we are focussing on the last dimension)
            index_next = torch.multinomial(probs, num_samples = 1) # (B, 1)
            
            # Appending sampled index to the running sequence
            index = torch.cat((index, index_next), dim = 1) # (B, T+1)

            '''
            Below 2 lines🔻
            Trimming : Added logic to trim the sequence if it exceeds block_size during text generation.
            '''

            if index.shape[1] > block_size:
                index = index[:, -block_size:]

        return index
    

model = GPTLanguageModel(vocab_size)

# Loading Saved Model
print("Loading Model Parameters...")
with open('model_gpt_v1_2.pkl', 'rb') as f:
    model = pickle.load(f)
print("Model Loaded Successfully!")

m = model.to(device)


# Now this is where we will be building our chatbot (on CMD)🔻
while True:
    prompt = input("Prompt : \n") # Getting he prompt
    context = torch.tensor(encode(prompt), dtype = torch.long, device = device)
    generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens = 150)[0].tolist())
    print(f"Completion : \n{generated_chars}")
