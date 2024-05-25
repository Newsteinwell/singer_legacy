
import torch
import torch.nn as nn
from torch.nn import functional as F
from datetime import datetime
import yaml
import math

    
with open('config.yaml', 'r', encoding='utf-8') as yaml_file:
    config = yaml.safe_load(yaml_file)

n_embed = config['n_embed'] 
batch_size = config['batch_size'] 
seq_len = config['seq_len']
device = config['device']
total_training_steps =  config['total_training_steps'] 
learning_rate = config['learning_rate']
eval_interval = config['eval_interval']
eval_iters = config['eval_iters'] 
n_layers = config['n_layers']
num_heads = config['num_heads']  # num_heads * head_size = n_embed
dropout = config['dropout']
vocab_size = config['vocab_size'] # it depends on the tokenizer scheme

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False) # to show that what is I am interested in
        self.key = nn.Linear(n_embed, head_size, bias=False)   # to show that what I have 
        self.value = nn.Linear(n_embed, head_size, bias=False) # to show that if you are interested with me, what I will communicate with you
        self.register_buffer('tril', torch.tril(torch.ones(seq_len, seq_len)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape  # batch_size, seq_len, channels (n_embed)

        # a single head self-attention
        q = self.query(x) # (B, T, head_size)
        k = self.key(x) # (B, T, head_size)
        # weight = q @ k.transpose(-2, -1) # (B, T, head_size) @ (B, head_size, T) ===> (B, T, T)
        weight = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, head_size) @ (B, head_size, T) ===> (B, T, T)

        # tril = torch.tril(torch.ones(T, T))
        # weight = torch.zeros((T,T))
        weight = weight.masked_fill(self.tril[:T, :T]==0, float('-inf')) # to make sure token_t only talk with previous tokens_(1,...,t-1), if you want a encoder block of transformer, remove this line
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        # xbow3 = weight @ x # ()
        v = self.value(x) # (B, T, head_size)
        output = weight @ v # (B, T, head_size)
        return output

class MultiHeadAttention(nn.Module):
    "multi head attention"
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed) # W_o
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x

# class CausalSelfAttention(nn.Module):
class MultiHeadAttention_new(nn.Module):
    "multi head attention"
    def __init__(self, num_heads, head_size):
        super().__init__()
        assert n_embed % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embed, 3 * n_embed, bias=False)
        # output projection
        self.c_proj = nn.Linear(n_embed, n_embed, bias=False)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = num_heads
        self.n_embed = n_embed
        self.dropout = dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        # self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.flash = True 
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(seq_len, seq_len))
                                        .view(1, 1, seq_len, seq_len))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class FeedForward(nn.Module):
    " a simple linear layer followed by a non-linearity "
    
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x) # (B, T, n_embed)
    
class LayerNorm(nn.Module):
    "the layernorm in transformer block"
    
    def __init__(self, dim, eps= 1e-5):
        super().__init___()
        self.eps = eps
        self.gamma = torch.ones(dim) # here it should be trainable one !
        self.beta = torch.zeros(dim) # here it should be trainable one !
    
    def __call__(self, x):
        # calculate the forward pass
        xmean = x.mean(1, keepdim=True)
        xvar = x.var(1, keepdim=True)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        output = self.gamma * xhat + self.beta
        return output
    
    def parameters(self):
        return [self.gamma, self.beta]
class TransformerBlock(nn.Module):
    "the transformer blocks"
    
    def __init__(self, n_embed, num_heads):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa_heads = MultiHeadAttention(num_heads=num_heads, head_size=head_size) 
        self.ffwd = FeedForward(n_embed)
        self.layernorm1 = nn.LayerNorm(n_embed)
        self.layernorm2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_heads(self.layernorm1(x))
        x = x + self.ffwd(self.layernorm2(x))
        return x
        
# bigramlanguage model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(seq_len, n_embed) 
        # self.sa_head = Head(n_embed)xbow3 
        # self.sa_heads = MultiHeadAttention(num_heads=num_heads, head_size = n_embed//num_heads) # i.e. num_heads (4) heads of head_size (8)-dimensional single head self-attention
        # self.ffwd = FeedForward(n_embed)
        # self.transformerblock = nn.Sequential(
        #     TransformerBlock(n_embed=n_embed, num_heads=num_heads),
        #     TransformerBlock(n_embed=n_embed, num_heads=num_heads),
        #     TransformerBlock(n_embed=n_embed, num_heads=num_heads),
        #     nn.LayerNorm(n_embed)
        #     )

        self.transformerblocks = nn.Sequential(*[TransformerBlock(n_embed=n_embed, num_heads=num_heads) for _ in range(n_layers)])
        self.layernorm_final = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # idx and targets are both (B. T) tensor of integers
        token_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C), nn.Embedding is not like nn.Linear, 
                                                                                # it will not run a matrix multiplication simply
        x = token_emb + pos_emb # (B, T, C)
        # x = self.sa_heads(x) # to apply single head of self attention, (B, T, C)
        # x = self.ffwd(x) # (B, T, C), think the data individually
        x = self.transformerblocks(x)
        x = self.layernorm_final(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        # B: batch_size, T: seq_len, or seq_len, C: vocab_size 
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, decode=None):
        # idx is (B, T) array of the indices of the current context
        for _ in range(max_new_tokens):
            # crop idx to the make sure length of input of transformer (position embedding table) won't be longer than seq_len
            idx_cond = idx[:, -seq_len:]
            # get the predictions
            logits, loss = self(idx_cond)  # logits: (B, T, C)
            # for predicting next token, only focus on the last time step
            logits = logits[:, -1, :] # logits become : (B, C) 
            # apply softmax to get the probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            if decode is not None:
                print (f'{decode([idx_next.item()])}', end=' ')
            # append the sampled idx to the existed context
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
