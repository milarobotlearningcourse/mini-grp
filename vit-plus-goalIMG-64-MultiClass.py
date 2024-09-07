
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


# data loading
def get_batch_vit(split, dataset, batch_size):
    # generate a small batch of inputs x and targets y
    data = dataset['train'] if split == 'train' else dataset['test']
    ix = np.random.randint(int(len(data["img"])), size=(batch_size,))
    x = torch.tensor(data["img"][ix], dtype=torch.float)
    x_goal = torch.tensor(data["goal_img"][ix], dtype=torch.float)
    y = torch.tensor(data["action"][ix], dtype=torch.long)
    # x, y = x.to(device), y.to(device)
    return x, x_goal, y

def calc_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(model._cfg.eval_iters)
        for k in range(model._cfg.eval_iters):
            X, x_goal, Y = get_batch_vit(split, model._dataset, model._cfg.batch_size)
            logits, loss = model(X, x_goal, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_patches_fast(images):
    from einops import rearrange
    batch_size, channels, height, width = images.shape
    patch_size = height // 8 ## n_patches = 8

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

    def __init__(self, head_size, n_embd, dropout):
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

    def __init__(self, num_heads, head_size, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd=n_embd, dropout=dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
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

    def __init__(self, n_embd, n_head, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd=n_embd, dropout=dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GRP(nn.Module):
  def __init__(self, dataset, cfg, mlp_ratio=4):
    super(GRP, self).__init__()
    self._dataset = dataset
    self._cfg = cfg
    self.patch_size = (cfg.image_shape[0] / cfg.n_patches, cfg.image_shape[1] / cfg.n_patches)

    #Positional embedding
    self.register_buffer('positional_embeddings', calc_positional_embeddings(1 + cfg.n_patches ** 2 + cfg.n_patches ** 2, cfg.n_embd), persistent=False)
    
    self.class_tokens = nn.Parameter(torch.rand(1, cfg.n_embd))

    self.input_d = int(cfg.image_shape[2] * self.patch_size[0] * self.patch_size[1])

    self.lin_map = nn.Linear(self.input_d, cfg.n_embd, bias=False) 

    # 4) Transformer encoder blocks
    self.blocks = nn.ModuleList([Block(cfg.n_embd, cfg.n_head, dropout=self._cfg.dropout) for _ in range(cfg.n_blocks)])

    # 5) Classification MLPk
    self.mlp = nn.Sequential(
        nn.Linear(cfg.n_embd, cfg.action_bins * 7),
        nn.Softmax(dim=-1)
    )

  def forward(self, images, x_goal, targets=None):
    # Dividing images into patches
    n, c, h, w = images.shape
    patches = get_patches_fast(images)
    patches_goal = get_patches_fast(x_goal)
    
    # Running linear layer tokenization
    # Map the vector corresponding to each patch to the hidden size dimension
    out = self.lin_map(patches)
    out_g = self.lin_map(patches_goal)
    
    # Adding classification and goal_img tokens to the tokens
    out = torch.cat((self.class_tokens.expand(n, 1, -1), out, out_g), dim=1)
    
    # Adding positional embedding
    out = out + self.positional_embeddings.repeat(n, 1, 1)
    # pos_emb = self.position_embedding_table(torch.arange(n_patches ** 2 + 1, device=device)) # (T,C)
    # out = out + pos_emb
    
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
        B, C = targets.shape
        # targets.view(B, 7)
        logits = logits.view(B, self._cfg.action_bins, C)
        # targets = targets.view(B, C)
        loss = F.cross_entropy(logits, targets)
    return (logits, loss)

import hydra, json
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf", config_name="bridge-64-multiClass")
def my_main(cfg: DictConfig):
    torch.manual_seed(cfg.r_seed)
    print ("cfg:", OmegaConf.to_yaml(cfg))
    # print (vars(cfg))
    print (OmegaConf.to_container(cfg))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    from datasets import load_dataset, load_from_disk

    dataset = load_dataset(cfg.dataset, split='train')
    ## dataset = load_from_disk("datasets/mini-bridge.hf")
    print('Features:', dataset.features)

    dataset_tmp = {
        "img": np.array(dataset["img"]),
        "action": np.concatenate((np.array(dataset["action"]), 
                                np.array(dataset["rotation_delta"]), 
                                np.array(dataset["open_gripper"]) + np.random.normal(0.0, 0.001, size = (len(dataset["open_gripper"]), 1))
        ), axis=1),
        "goal_img": np.array(dataset["goal_img"]),
        "goal": dataset["goal"]
    }
    block_size = shortest_goal_txt = min([len(txt) for txt in dataset["goal"]])

    import pandas as pd
    action_labels = [pd.qcut(dataset_tmp["action"][:,i], 
                             cfg.action_bins, labels=False) for i in range(dataset_tmp["action"].shape[1])] ## Split the classes equally across options, -1 because open/closed gripper is already a bin of 2.
    # action_labels.append(dataset["open_gripper"]) ## Add gripper closed/open
    print("bins:", action_labels)
    # hist, bin_edges = np.histogram(action_labels, density=True, bins=cfg.action_bins)
    # print("action histogram:", hist)

    ## Get the actions and encode them to map to [-1, 1]
    encode_state = lambda af:   ((af/(255.0)*2.0)-1.0).astype(np.float32) # encoder: take a float, output an integer
    resize_state = lambda sf:   cv2.resize(np.array(sf, dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))  # resize state
    # decode_action = lambda binN: (binN * a_std ) + a_mean  # Undo mapping to [-1, 1]


    dataset_tmp = {
        "img": torch.tensor(encode_state(dataset_tmp["img"])).to(device),
        "action": torch.tensor(np.reshape(action_labels, (dataset_tmp["action"].shape[0], 7)), dtype=torch.uint8).to(device),            
        "goal_img": torch.tensor(encode_state(dataset_tmp["goal_img"])).to(device),
        # "goal": torch.tensor([encode_txt(goal[:shortest_goal_txt]) for goal in dataset_tmp["goal"]]).to(device)
    }

    print("Dataset shape:", len(dataset_tmp["img"]))
    dataset = {"train": dataset_tmp, "test": dataset_tmp} 
    # print ("Results:", results)
    import wandb
    import torch.optim.lr_scheduler as lr_scheduler
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="mini-grp",

        # track hyperparameters and run metadata
        config= OmegaConf.to_container(cfg)
    )
    wandb.run.log_code(".")
    model = GRP(dataset, cfg)
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=cfg.max_iters)

    for iter in range(cfg.max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % cfg.eval_interval == 0 or iter == cfg.max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            wandb.log({"train loss": losses['train'], "val loss": losses['val']})

        # sample a batch of data
        xb, xg, yb = get_batch_vit('train', dataset, cfg.batch_size)

        # evaluate the loss
        logits, loss = model(xb, xg, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    # print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
    wandb.finish()
    return losses['val']

if __name__ == "__main__":
    import os
    results = my_main()
    print("results:", results)