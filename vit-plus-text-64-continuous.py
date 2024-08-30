
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
batch_size = 512 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
vocab_size = n_patches = 8
max_iters = 10000
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
dropout = 0.0

## Model hyperparameters
action_bins = 10
image_shape = [64, 64, 3]
name = 'mini-bridge-mini64pix'

from datasets import load_dataset, load_from_disk
dataset = load_dataset("gberseth/" + name, split='train')
## dataset = load_from_disk("datasets/mini-bridge.hf")
print('Features:', dataset.features)
# dataset = dataset.with_format("np")
# print('Features:', dataset.features)
dataset_tmp = {
    "img": np.array(dataset["img"], dtype=np.uint8), ## This cast seems to take a long time...
    "action": np.concatenate((np.array(dataset["action"]), 
                              np.array(dataset["rotation_delta"]), 
                              np.array(dataset["open_gripper"])), axis=1),
    "goal_img": np.array(dataset["goal_img"], dtype=np.uint8),
    "goal": dataset["goal"]
}
block_size = shortest_goal_txt = min([len(txt) for txt in dataset["goal"]])

print("Dataset shape:", len(dataset_tmp["img"]))
batch_size = min(batch_size, len(dataset_tmp["img"]))
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



## Get the actions and encode them to map to [-1, 1]
a_min = dataset_tmp["action"].min(axis=0) - 0.001 ## Get the min and max bound for the actions to use for bining 
a_max = dataset_tmp["action"].max(axis=0) 
a_std, a_mean = (dataset_tmp["action"].std(axis=0) + 0.001) * 1.0, dataset_tmp["action"].mean(axis=0)
action_bins = len(a_mean)
s_std, s_mean = dataset_tmp["img"].std(axis=0), dataset_tmp["img"].mean(axis=0) 
a_max = a_max + ((a_max - a_min) / 20.0) ## + a little to avoid using action_bins + 1 for the action = max
encode_action = lambda af:   (((af - a_mean)/(a_std))).astype(np.float32) # encoder: take a float, output an integer
encode_state = lambda af:   ((af/(255.0)*2.0)-1.0).astype(np.float32) # encoder: take a float, output an integer
resize_state = lambda sf:   cv2.resize(np.array(sf, dtype=np.float32), (image_shape[0], image_shape[1]))  # resize state
decode_action = lambda binN: (binN * a_std ) + a_mean  # Undo mapping to [-1, 1]
# for i in range(len(dataset_tmp["action"])): ## Convert to classes
dataset_tmp["action"] = encode_action(dataset_tmp["action"])
dataset_tmp["img"] = encode_state(dataset_tmp["img"])
dataset_tmp["goal_img"] = encode_state(dataset_tmp["goal_img"])

dataset_tmp = {
    "img": torch.tensor(encode_state(dataset_tmp["img"])).to(device),
    "action": torch.tensor(dataset_tmp["action"], dtype=torch.float).to(device),            
    "goal_img": torch.tensor(encode_state(dataset_tmp["goal_img"])).to(device),
    "goal": torch.tensor([encode_txt(goal[:shortest_goal_txt]) for goal in dataset_tmp["goal"]]).to(device)
}

dataset = {"train": dataset_tmp, "test": dataset_tmp} 


# data loading
def get_batch(split):
    # generate a small batch of inputs x and targets y
    data = dataset['train'] if split == 'train' else dataset['test']
    ix = np.random.randint(int(len(data["img"])), size=(batch_size,))
    x = data["img"][ix]
    y = data["action"][ix]
    goal_e = data["goal"][ix]
    return x, goal_e, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, X2, Y = get_batch(split)
            logits, loss = model(X, X2, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

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

class GRP(nn.Module):
  def __init__(self, mlp_ratio=4):
    super(GRP, self).__init__()
    ## Text processing portion
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.register_buffer('positional_embeddings', calc_positional_embeddings(block_size + (n_patches ** 2) + 1, n_embd), persistent=False)
    self.patch_size = (image_shape[0] / n_patches, image_shape[1] / n_patches)

    #Positional embedding
    
    self.class_tokens = nn.Parameter(torch.rand(1, n_embd))

    self.input_d = int(image_shape[2] * self.patch_size[0] * self.patch_size[1])

    self.lin_map = nn.Linear(self.input_d, n_embd, bias=False) 
    self.lin_map_goal_img = nn.Linear(self.input_d, n_embd, bias=False) 

    # 4) Transformer encoder blocks
    self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_blocks)])

    # 5) Classification MLPk
    self.mlp = nn.Sequential(
        # nn.LayerNorm(n_embd),
        nn.Linear(n_embd, action_bins),
    )

  def forward(self, images, goals, targets):
    # Dividing images into patches
    n, c, h, w = images.shape
    B, T = goals.shape
    patches = get_patches_fast(images).to(device)
    goals_e = self.token_embedding_table(goals)
    
    # Running linear layer tokenization
    # Map the vector corresponding to each patch to the hidden size dimension
    out = self.lin_map(patches)
    
    # Adding classification token to the tokens
    out = torch.cat((self.class_tokens.expand(n, 1, -1), goals_e, out), dim=1)
    # out = torch.cat([out, goals_e], dim=1) ## Add text and goal image encoding to begining of encoding.
    
    # Adding positional embedding
    out = out + self.positional_embeddings.repeat(n, 1, 1)
    
    # Transformer Blocks
    for block in self.blocks:
        out = block(out)

    # Getting the classification token only
    out = out[:, 0]
    out = self.mlp(out)
        
    if targets is None:
        loss = None
    else:
        B, C = targets.shape
        # targets = targets.view(B)
        out = out.view(B, C)
        # diff = torch.abs(torch.mean(out - targets, axis=0))
        # print ("diff:", diff) ## Let's see which dimensions have the larges errors
        loss = F.mse_loss(out, targets) ## B, C
    return (out, loss)

if __name__ == "__main__":
    import wandb
    import torch.optim.lr_scheduler as lr_scheduler
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="mini-grp",
        # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "architecture": "GRP",
        "dataset": name,
        "epochs": max_iters,
        },
        save_code=True
    )
    wandb.run.log_code(".")
    model = GRP()
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=max_iters)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            ## Really want the loss to start out 0.2 or less.
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            wandb.log({"train loss": losses['train'], "val loss": losses['val']})

        # sample a batch of data
        xb, x2b, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, x2b, yb)
        optimizer.zero_grad(set_to_none=False)
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 1)
        optimizer.step()

    # generate from the model
    # context = torch.zeros((1, 1), dtype=torch.long, device=device)
    # print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
    wandb.finish()