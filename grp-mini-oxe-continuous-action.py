
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
n_embd = 64
# ------------

torch.manual_seed(1337)
n_head = 8
n_blocks = 4
dropout = 0.1

## Model hyperparameters
action_bins = 3
image_shape = [64, 64, 3]

from datasets import load_dataset

# ------------
# Train and test splits
# Loading data
# create RLDS dataset builder
num_episodes = 5 ## How many episodes to grab from the dataset for training
builder = tfds.builder_from_directory(builder_dir='gs://gresearch/robotics/bridge/0.1.0/')
datasetRemote = builder.as_dataset(split='train[:' + str(num_episodes) + ']')
dataset_tmp = {"img": [], "action": [], "goal": [], "goal_img": []}
shortest_goal_txt = 10000000000
for episode in datasetRemote:
    episode_ = {'steps': [] }
    episode = list(episode['steps'])
    goal_img = cv2.resize(np.array(episode[-1]['observation']['image'], dtype=np.float32), (image_shape[0], image_shape[1]))  
    for i in range(len(episode)): ## Resize images to reduce computation
        obs = cv2.resize(np.array(episode[i]['observation']['image'], dtype=np.float32), (image_shape[0], image_shape[1])) 
        action = np.array(episode[i]['action']['world_vector'])
        goal = episode[i]['observation']['natural_language_instruction'].numpy().decode()
        # action = torch.as_tensor(action) # grab first dimention
        dataset_tmp["img"].append(obs)
        dataset_tmp["action"].append(action)
        dataset_tmp["goal"].append(goal)
        dataset_tmp["goal_img"].append(goal_img)
        if len(goal) < shortest_goal_txt: shortest_goal_txt = len(goal)

# here are all the unique characters that occur in this text
chars = sorted(list(set([item for row in dataset_tmp["goal"] for item in row]))) ## Flatten to a long string
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode_txt = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode_txy = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
print("vocab_size:", vocab_size)
print("example text encode:", encode_txt(dataset_tmp["goal"][0]))

print("Dataset shape:", len(dataset_tmp["img"]))
dataset_tmp["img"] = np.array(dataset_tmp["img"], dtype=np.uint8)
dataset_tmp["action"] = np.array(dataset_tmp["action"], dtype=np.float32)
# dataset_tmp["goal"] = np.array(dataset_tmp["goal"], dtype=np.float32)
dataset_tmp["goal_img"] = np.array(dataset_tmp["goal_img"], dtype=np.uint8)

n = int(0.9*len(dataset_tmp["img"])) # first 90% will be train, rest val
dataset = {"train": dataset_tmp, "test": dataset_tmp} 


# data loading
def get_batch(split):
    # generate a small batch of inputs x and targets y
    data = dataset['train'] if split == 'train' else dataset['test']
    ix = np.random.randint(int(len(data["img"])), size=(batch_size,))
    x = torch.tensor(data["img"][ix], dtype=torch.float)
    x_goal_img = torch.tensor(data["goal_img"][ix], dtype=torch.float)
    y = torch.tensor(data["action"][ix], dtype=torch.float)
    goal_e = [encode_txt(data["goal"][ix[i]][:shortest_goal_txt]) for i in range(len(ix))]
    x2 = torch.tensor(goal_e, dtype=torch.long).to(device)
    return x.to(device), x2.to(device), x_goal_img.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, X2, X3, Y = get_batch(split)
            logits, loss = model(X, X2, X3, Y)
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
    ## Text processing portion
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    # self.position_embedding_table_goal = nn.Embedding(block_size, n_embd)

    # assert shape[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
    # assert shape[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
    self.patch_size = (image_shape[0] / n_patches, image_shape[1] / n_patches)

    #Positional embedding
    self.position_embedding_table = nn.Embedding(block_size + (n_patches ** 2) + (n_patches ** 2) + 1, n_embd)
    
    self.class_tokens = nn.Parameter(torch.rand(1, n_embd))

    self.input_d = int(image_shape[2] * self.patch_size[0] * self.patch_size[1])

    self.lin_map = nn.Linear(self.input_d, n_embd, bias=False) 
    self.lin_map_goal_img = nn.Linear(self.input_d, n_embd, bias=False) 

    # 4) Transformer encoder blocks
    self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_blocks)])

    # 5) Regression MLPk
    self.mlp = nn.Sequential(
        nn.Linear(n_embd, action_bins),
        # nn.Softmax(dim=-1)
        # nn.ReLU(),
        nn.Tanh()
    )

  def forward(self, images, goals, goal_img, targets):
    # Dividing images into patches
    n, c, h, w = images.shape
    B, T = goals.shape
    patches = get_patches_fast(images).to(device)
    patches_goal_img = get_patches_fast(goal_img).to(device)
    goals_e = self.token_embedding_table(goals)
    
    # Running linear layer tokenization
    # Map the vector corresponding to each patch to the hidden size dimension
    out = self.lin_map(patches)
    out_goal_img = self.lin_map(patches_goal_img)
    
    # Adding classification token to the tokens
    out = torch.cat((self.class_tokens.expand(n, 1, -1), out), dim=1)
    out = torch.cat([out_goal_img, goals_e, out], dim=1) ## Add text and goal image encoding to begining of encoding.
    
    # Adding positional embedding
    # out = out + self.positional_embeddings.repeat(n, 1, 1)
    # pos_emb_goal_txt = self.position_embedding_table(torch.arange(n, device=device)) # (T,C)
    pos_emb = self.position_embedding_table(torch.arange(T + c + c + 1, device=device)) # (T,C)
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
        # targets = targets.view(B)
        loss = F.mse_loss(logits, targets)
    return (logits, loss)

if __name__ == "__main__":
    import wandb
    # start a new wandb run to track this script
    model = VIT()
    m = model.to(device)
    wandb.init(
        # set the wandb project where this run will be logged
        project="mini-grp",

        # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "architecture": "VIT",
        "dataset": "EleutherAI/cifarnet",
        "epochs": max_iters,
        }
    )
    wandb.run.log_code(".")
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.8f}, val loss {losses['val']:.8f}")
            wandb.log({"train loss": losses['train'], "val loss": losses['val']})

        # sample a batch of data
        xb, x2b, x3b, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, x2b, x3b, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    # context = torch.zeros((1, 1), dtype=torch.long, device=device)
    # print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
    wandb.finish()