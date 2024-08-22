
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm, trange
import cv2

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
vocab_size = n_patches = 8
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
# device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
eval_iters = 200
n_embd = 256
# ------------

torch.manual_seed(1337)
n_head = 8
n_blocks = 4
dropout = 0.1

## Model hyperparameters
action_bins = 10
image_shape = [64, 64, 3]

from datasets import load_dataset

ds = load_dataset("EleutherAI/cifarnet")

# data = torch.tensor(ds)

# np.reshape(np.array(x["img"][i].getdata(), dtype=np.float32)
trim = 1000000 ## Lets see how little data is needed to still get good performance. 1000 is not enough.
dataset = {}
dataset["train"]= {
           "img": torch.tensor(np.array(ds["train"]["img"][:trim], dtype=np.uint8)).to(device),
           "label": torch.tensor(np.array(ds["train"]["label"][:trim], dtype=np.uint8)).to(device) 
          }         
dataset["test"]=  {
           "img": torch.tensor(np.array(ds["test"]["img"][:trim], dtype=np.uint8)).to(device),
           "label": torch.tensor(np.array(ds["test"]["label"][:trim], dtype=np.uint8)).to(device)
         }


# data loading
def get_batch_oxe(split):
    # generate a small batch of inputs x and targets y
    data = train_set if split == 'train' else val_set
    ix = torch.randint(len(data), (batch_size,))
    # print ( data[ix])
    x = torch.stack([data[i][0] for i in ix])
    y = torch.stack([data[i][1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# data loading
def get_batch_vit(split):
    # generate a small batch of inputs x and targets y
    data = dataset['train'] if split == 'train' else dataset['test']
    ix = np.random.randint(int(len(data["img"])), size=(batch_size,))
    x = torch.tensor(data["img"][ix], dtype=torch.float)
    y = torch.tensor(data["label"][ix], dtype=torch.long)
    # x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_vit(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_patches(images):
  batch_size, channels, height, width = images.shape

  assert height == width, "square images only"

  patches = torch.zeros(batch_size, n_patches ** 2, height * width * channels // n_patches ** 2)
  patch_size = height // n_patches

  for idx, image in enumerate(images):
      for row in range(n_patches):
          for column in range(n_patches):
            ## Channel first
            patch = image[:, row * patch_size: (row + 1) * patch_size, column * patch_size: (column + 1) * patch_size]
            patches[idx, row * n_patches + column] = patch.flatten()

  return patches

def get_patches_fast(images):
    from einops import rearrange
    batch_size, channels, height, width = images.shape
    patch_size = height // n_patches

    p = patch_size # P in maths

    patches = rearrange(images, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
    return patches

def calc_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

## This is an encoder head (full attention)
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        # wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T) ## Remove masking
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class VIT(nn.Module):
  def __init__(self, mlp_ratio=4):
    super(VIT, self).__init__()
    # assert shape[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
    # assert shape[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
    self.patch_size = (image_shape[0] / n_patches, image_shape[1] / n_patches)

    #Positional embedding
    # self.pos_embed = nn.Parameter(torch.tensor(positional_embeddings(n_patches ** 2 + 1, embedding_size)))
    # self. pos_embed.requires_grad = False
    # self.register_buffer('positional_embeddings', calc_positional_embeddings(n_patches ** 2 + 1, n_embd), persistent=False)
    self.position_embedding_table = nn.Embedding(n_patches ** 2 + 1, n_embd)
    
    self.class_tokens = nn.Parameter(torch.rand(1, n_embd))

    self.input_d = int(image_shape[2] * self.patch_size[0] * self.patch_size[1])

    self.lin_map = nn.Linear(self.input_d, n_embd, bias=False) 

    # 4) Transformer encoder blocks
    self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_blocks)])

    # 5) Classification MLPk
    self.mlp = nn.Sequential(
        nn.Linear(n_embd, action_bins),
        nn.Softmax(dim=-1)
    )

  def forward(self, images, targets=None):
    # Dividing images into patches
    n, c, h, w = images.shape
    patches = get_patches_fast(images).to(device)
    
    # Running linear layer tokenization
    # Map the vector corresponding to each patch to the hidden size dimension
    out = self.lin_map(patches)
    
    # Adding classification token to the tokens
    out = torch.cat((self.class_tokens.expand(n, 1, -1), out), dim=1)
    
    # Adding positional embedding
    # out = out + self.positional_embeddings.repeat(n, 1, 1)
    pos_emb = self.position_embedding_table(torch.arange(n_patches ** 2 + 1, device=device)) # (T,C)
    out = out + pos_emb
    
    # Transformer Blocks
    for block in self.blocks:
        out = block(out)

    # Getting the classification token only
    out = out[:, 0]
    logits = self.mlp(out)
        
    if targets is None:
        loss = None
    else:
        # B,T,C = 4,8,2 # batch, time, channels
        B, C = logits.shape
        # logits = logits.view(B*T, C)
        targets = targets.view(B)
        loss = F.cross_entropy(logits, targets)
    return (logits, loss)

if __name__ == "__main__":
    import wandb
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="mini-vit",

        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.02,
        "architecture": "VIT",
        "dataset": "EleutherAI/cifarnet",
        "epochs": max_iters,
        }
    )
    wandb.run.log_code(".")
    model = VIT()
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            wandb.log({"train loss": losses['train'], "val loss": losses['val']})

        # sample a batch of data
        xb, yb = get_batch_vit('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    # print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
    wandb.finish()