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
def get_batch_grp(split, dataset, batch_size):
    # generate a small batch of inputs x and targets y
    data = dataset['train'] if split == 'train' else dataset['test']
    ix = np.random.randint(int(len(data["img"])), size=(batch_size,))
    x = torch.tensor(data["img"][ix], dtype=torch.float)
    x_goal = torch.tensor(data["goal"][ix], dtype=torch.long)
    x_goal_img = torch.tensor(data["goal_img"][ix], dtype=torch.float)
    y = torch.tensor(data["action"][ix], dtype=torch.float)
    # x, y = x.to(device), y.to(device)
    return x, x_goal, x_goal_img, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(model._cfg.eval_iters)
        for k in range(model._cfg.eval_iters):
            X, x_goal, x_goal_img, Y = get_batch_grp(split, model._dataset, model._cfg.batch_size)
            logits, loss = model(X, x_goal, x_goal_img, Y)
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

    def forward(self, x, mask=None):
        B,T,C = x.shape
        if mask == None:
            mask = torch.ones((T, ), device=self.device) ## (1, T)
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        ### Block masked attention
        wei = wei.masked_fill(mask == 0, float('-inf')) # (B, T, T)
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

    def forward(self, x, mask=None):
        out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
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
        return self.net(x,)

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

    def forward(self, x, mask=None):
        x = x + self.sa(self.ln1(x), mask)
        x = x + self.ffwd(self.ln2(x))
        return x

class GRP(nn.Module):
  def __init__(self, dataset, cfg, mlp_ratio=4):
    super(GRP, self).__init__()
    self._dataset = dataset
    self._cfg = cfg

    self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
    self.patch_size = (self._cfg.image_shape[0] / self._cfg.n_patches, self._cfg.image_shape[1] / self._cfg.n_patches)

    #Positional embedding
    self.register_buffer('positional_embeddings', calc_positional_embeddings(1 + self._cfg.n_patches ** 2 + self._cfg.block_size + self._cfg.n_patches ** 2, cfg.n_embd), persistent=False)
    
    self.class_tokens = nn.Parameter(torch.rand(1, cfg.n_embd))

    self.input_d = int(self._cfg.image_shape[2] * self.patch_size[0] * self.patch_size[1])

    self.lin_map = nn.Linear(self.input_d, self._cfg.n_embd, bias=False) 

    # 4) Transformer encoder blocks
    self.blocks = nn.ModuleList([Block(self._cfg.n_embd, self._cfg.n_head, dropout=self._cfg.dropout) for _ in range(self._cfg.n_blocks)])

    # 5) Classification MLPk
    self.mlp = nn.Sequential(
        nn.Linear(self._cfg.n_embd, self._cfg.action_bins),
        # nn.Softmax(dim=-1)
    )

  def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
          if module.bias is not None:
              torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, images, goals, goal_imgs, targets=None):
    # Dividing images into patches
    n, c, h, w = images.shape
    B, T = goals.shape
    patches = get_patches_fast(images)
    patches_g = get_patches_fast(goal_imgs)
    goals_e = self.token_embedding_table(goals)
    
    # Running linear layer tokenization
    # Map the vector corresponding to each patch to the hidden size dimension
    out = self.lin_map(patches)
    out_g = self.lin_map(patches_g)
    
    # Adding classification and goal_img tokens to the tokens
    out = torch.cat((self.class_tokens.expand(n, 1, -1), out, goals_e, out_g), dim=1)
    
    # Adding positional embedding
    out = out + self.positional_embeddings.repeat(n, 1, 1)

    ## Compute blocked masks
    mask = torch.ones((1 + c + T + c, ), device=self._cfg.device) ## (1, T)
    if targets is None:
        pass
    elif (torch.rand(1)[0] > 0.66):  
        mask[1 + c: 1 + c+ T] = torch.zeros((1,T), device=self._cfg.device) ## Mask goal string
    elif (torch.rand(1)[0] > 0.33):
        mask[1 + c + T: 1 + c + T + c] = torch.zeros((1,c), device=self._cfg.device) ## Mask goal image
        
    # Transformer Blocks
    for block in self.blocks:
        out = block(out, mask)

    # Getting the classification token only
    out = out[:, 0]
    out = self.mlp(out)
        
    if targets is None:
        loss = None
    else:
        B, C = out.shape
        loss = F.mse_loss(out, targets) ## B, C
    return (out, loss)

import hydra, json
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf", config_name="bridge-64-light")
def my_main(cfg: DictConfig):
    torch.manual_seed(cfg.r_seed)
    print ("cfg:", OmegaConf.to_yaml(cfg))
    # print (vars(cfg))
    print (OmegaConf.to_container(cfg))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    cfg.device = device
    from datasets import load_dataset, load_from_disk

    dataset = load_dataset(cfg.dataset, split='train')
    ## dataset = load_from_disk("datasets/mini-bridge.hf")
    print('Features:', dataset.features)

    dataset_tmp = {
        "img": np.array(dataset["img"]),
        "action": np.concatenate((np.array(dataset["action"]) 
                                ,np.array(dataset["rotation_delta"])
                                # .np.array(dataset["open_gripper"])
                                ), axis=1),
        "goal_img": np.array(dataset["goal_img"]),
        "goal": dataset["goal"]
    }
    shortest_text_len = min([len(txt) for txt in dataset["goal"]])
    cfg.block_size = shortest_text_len

    # here are all the unique characters that occur in this text
    chars = sorted(list(set([item for row in dataset_tmp["goal"] for item in row]))) ## Flatten to a long string
    cfg.vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode_txt = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode_txy = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
    print("vocab_size:", cfg.vocab_size)
    print("example text encode:", encode_txt(dataset_tmp["goal"][0]))

    a_std, a_mean = (dataset_tmp["action"].std(axis=0) + 0.001) * 1.5, dataset_tmp["action"].mean(axis=0)
    cfg.action_bins = len(a_mean)
    encode_action = lambda af:   (((af - a_mean)/(a_std))).astype(np.float32) # encoder: take a float, output an integer

    ## Get the actions and encode them to map to [-1, 1]
    encode_state = lambda af:   ((af/(255.0)*2.0)-1.0).astype(np.float32) # encoder: take a float, output an integer
    resize_state = lambda sf:   cv2.resize(np.array(sf, dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))  # resize state
    decode_action = lambda binN: (binN * a_std) + a_mean  # Undo mapping to [-1, 1]

    dataset_tmp = {
        "img": torch.tensor(encode_state(dataset_tmp["img"])).to(device),
        "action": torch.tensor(encode_action(dataset_tmp["action"]), dtype=torch.float).to(device),            
        "goal_img": torch.tensor(encode_state(dataset_tmp["goal_img"])).to(device),
        "goal": torch.tensor([encode_txt(goal[:cfg.block_size]) for goal in dataset_tmp["goal"]]).to(device)
    }

    print("Dataset shape:", len(dataset_tmp["img"]))
    dataset_tmp = {"train": dataset_tmp, "test": dataset_tmp} 
    # print ("Results:", results)
    if not cfg.testing:
        import wandb
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="mini-grp-continuous",

            # track hyperparameters and run metadata
            config= OmegaConf.to_container(cfg)
        )
        wandb.run.log_code(".")
    model = GRP(dataset_tmp, cfg)
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    # import simpler_env
    # from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    # task_name = "widowx_carrot_on_plate"  # @param ["google_robot_pick_coke_can", "google_robot_move_near", "google_robot_open_drawer", "google_robot_close_drawer", "widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]
    # if 'env' in locals():
    #     print("Closing existing env")
    #     env.close()
    #     del env
    # env = simpler_env.make(task_name)

    for iter in range(cfg.max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % cfg.eval_interval == 0 or iter == cfg.max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if not cfg.testing:
                wandb.log({"train loss": losses['train'], "val loss": losses['val']})

            # obs, reset_info = env.reset()
            # instruction = env.get_language_instruction()
            # print("Reset info", reset_info)
            # print("Instruction", instruction)
            # frames, rewards = [], []
            # done, truncated = False, False
            # while not (done or truncated):
            #     # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
            #     # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
            #     image = get_image_from_maniskill2_obs_dict(env, obs)
            #     action, loss = model.forward(torch.tensor(np.array([encode_state(resize_state(image))])).to(device)
            #                            ,torch.tensor(np.array([encode_txt(instruction)[:cfg.block_size]])).to(device)
            #                         #    ,torch.tensor(np.array([encode_state(resize_state(image))])).to(device)
            #                            )
            #     # action = env.action_space.sample() # replace this with your policy inference
            #     action = np.concatenate((decode_action(action.cpu().detach().numpy()[0]), [0]), axis = -1) ## Add in the gripper close action
            #     # print("action: ", action)
            #     obs, reward, done, truncated, info = env.step(action)
            #     frames.append(image)
            #     rewards.append(reward)
            
            # episode_stats = info.get('episode_stats', {})
            # print("Episode stats", episode_stats)
            # print(f"avg reward {np.mean(rewards):.8f}")
            # wandb.log({"avg reward": np.mean(rewards)})
            # import moviepy.editor as mpy
            # clip = mpy.ImageSequenceClip(list(frames), fps=20)
            # clip.write_videofile("./data/sim-env-"+str(0)+".mp4", fps=20)
            # wandb.log({"example": wandb.Video("./data/sim-env-"+str(0)+".mp4")})

        # sample a batch of data
        xb, xg, xgi, yb = get_batch_grp('train', dataset_tmp, cfg.batch_size)

        # evaluate the loss
        logits, loss = model(xb, xg, xgi, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    # print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
    if not cfg.testing:
        wandb.finish()
    return losses['val']

if __name__ == "__main__":
    results = my_main()
    print("results:", results)
