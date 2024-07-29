
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST
import tensorflow_datasets as tfds

import numpy as np
from tqdm import tqdm, trange
import cv2

# hyperparameters
batch_size = 65 # how many independent sequences will we process in parallel?
block_size = 5 # what is the maximum context length for predictions?
n_patches = 8
max_iters = 5000
eval_interval = 10
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
eval_iters = 200
n_embd = 128
n_embed_text = 256
# ------------

torch.manual_seed(1337)
n_head = 8
n_blocks = 4
dropout = 0.0

## Data specific hyper parameters
output_size = action_bins = 20
image_shape = [64,64,3]

# ------------
# Train and test splits
# Loading data
transform = ToTensor()
# create RLDS dataset builder
num_episodes = 3 ## How many episodes to grab from the dataset for training
builder = tfds.builder_from_directory(builder_dir='gs://gresearch/robotics/bridge/0.1.0/')
datasetRemote = builder.as_dataset(split='train[:' + str(num_episodes) + ']')
dataset = []
text, shortest_goal_txt= "", 100000000 ## Get the charaters for the goals and use them for the encoding.
for episode in datasetRemote:
    episode_ = {'steps': [] }
    episode = list(episode['steps'])
    for i in range(len(episode)): ## Resize images to reduce computation
        obs = cv2.resize(np.array(episode[i]['observation']['image']), (image_shape[0], image_shape[1])) 
        action = episode[i]['action']['world_vector']
        goal = episode[0]['observation']['natural_language_instruction'].numpy().decode()
        episode_['steps'].append({"obs": obs, "action": action, "goal": goal})
        if len(goal) < shortest_goal_txt: shortest_goal_txt = len(goal)
        text = text + str(goal)
    dataset.append(episode_) ## Save these episodes locally for training.
block_size = shortest_goal_txt ## This will determine block size
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode_txt = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode_txy = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

## Get the actions and encode them as well.
actions = [] ## Get the charaters for the goals and use them for the encoding.
for episode in dataset:
    steps = list(episode['steps'])
    actions.extend([step['action'] for step in steps]) 
actions = np.array(actions)
a_min = actions.min(axis=0) ## Get the min and max bound for the actions to use for bining 
a_max = actions.max(axis=0) 
a_max = a_max + ((a_max - a_min) / float(action_bins * 2)) ## + a little to avoid using action_bins + 1 for the action = max
spacing = (a_max - a_min)/float(action_bins)
encode_action = lambda af:   np.floor((af - a_min)/spacing).astype(np.int32) # encoder: take a float, output an integer
decode_action = lambda binN: (binN * spacing) + a_min  # decoder: take a list of integers, output a string

# data = torch.tensor(encode(text), dtype=torch.long)
train_set = dataset
val_data = dataset


# data loading
def get_batch_grp_oxe(split):
    import cv2
    # generate a small batch of inputs x and targets y
    data = train_set if split == 'train' else val_data
    ## Get indicies for random episodes
    ex = torch.randint(len(data), (batch_size,))
    ## Get random timesetp for variable length episodes
    goals, obs, actions = ([] for i in range(3))
    for e in ex:
        idx = torch.randint(len(data[e]['steps']), (1,1))
        steps = list(data[e]['steps'])
        goals.append(encode_txt(steps[idx]['goal'][:shortest_goal_txt])) ## Trim to shortest goal length
        obs.append(steps[idx]['obs'])
        # obs.append(cv2.resize(np.array(steps[idx]['observation']['image']), (image_shape[0], image_shape[1])))
        actions.append(encode_action(steps[idx]['action'])[0])
    goals, obs, actions = torch.tensor(np.array(goals), dtype=torch.long), torch.tensor(np.array(obs), dtype=torch.float32), torch.tensor(np.array(actions), dtype=torch.long)
    goals, obs, actions = goals.to(device), obs.to(device), actions.to(device)
    return goals, obs, actions

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X1, X2, Y = get_batch_grp_oxe(split)
            logits, loss = model(X1, X2, Y)
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

    patches = rearrange(images, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = p, p2 = p) ## Channel last
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

class GRP(nn.Module): # Generalist Robot Policy
  def __init__(self, mlp_ratio=4):
    super(GRP, self).__init__()
    ## Text processing portion
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size + (n_patches ** 2 + 1), n_embd)
    
    self.patch_size = (image_shape[0] / n_patches, image_shape[1] / n_patches)

    #Positional embedding
    # self.pos_embed = nn.Parameter(torch.tensor(positional_embeddings(n_patches ** 2 + 1, embedding_size)))
    # self. pos_embed.requires_grad = False
    # self.register_buffer('positional_embeddings', calc_positional_embeddings(n_patches ** 2 + 1, n_embd), persistent=False)
    
    self.class_tokens = nn.Parameter(torch.rand(1, n_embd))

    self.input_d = int(image_shape[2] * self.patch_size[0] * self.patch_size[1])

    self.lin_map = nn.Linear(self.input_d, n_embd, bias=False) ## Here what I am interested in

    # concat = torch.cat([self.token_embedding_table, self.lin_map], dim=1)
    # 4) Transformer encoder blocks
    self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_blocks)])

    # 5) Classification MLP
    self.mlp = nn.Sequential(
        nn.Linear(n_embd, output_size),
        nn.Softmax(dim=-1)
    )

  def forward(self, idx, images, targets=None):
    ## Text processing first 
    B, T = idx.shape
    # Dividing images into patches
    n, c, h, w = images.shape
    patches = get_patches_fast(images).to(device) # (B, N_P, C)
    _, n_patches, _ = patches.shape

    # idx and targets are both (B,T) tensor of integers
    tok_emb_txt = self.token_embedding_table(idx) # (B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(T + n_patches + 1, device=device)) # (T,C)
    x_text = tok_emb_txt # + pos_emb_txt # (B,T,C)

    
    # Running linear layer tokenization
    # Map the vector corresponding to each patch to the hidden size dimension
    out = self.lin_map(patches) # (B, n_embed)
    
    # Adding classification token to the tokens
    out = torch.cat((self.class_tokens.expand(n, 1, -1), out), dim=1) # (B, n_embed +1)
    
    # Adding positional embedding
    # out = out + self.positional_embeddings.repeat(n, 1, 1) # (B, n_embed + 1)

    # out = out + x_text
    out = torch.cat([out, x_text], dim=1)

    ## Add position embedding for text and image patches
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


model = GRP()
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

    # sample a batch of data
    x1b, x2b, yb = get_batch_grp_oxe('train')

    # evaluate the loss
    logits, loss = model(x1b, x2b, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))