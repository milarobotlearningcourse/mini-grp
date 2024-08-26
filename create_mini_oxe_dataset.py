
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
eval_iters = 20
n_embd = 64
# ------------

torch.manual_seed(1337)
n_head = 8
n_blocks = 4
dropout = 0.1

## Model hyperparameters
image_shape = [64, 64, 3]
num_episodes = 1000 ## How many episodes to grab from the dataset for training

from datasets import load_dataset

# ------------
# Train and test splits
# Loading data
# create RLDS dataset builder
builder = tfds.builder_from_directory(builder_dir='gs://gresearch/robotics/bridge/0.1.0/')
datasetRemote = builder.as_dataset(split='train[:' + str(num_episodes) + ']')
dataset_tmp = {"img": [], "action": [], "goal": [], "goal_img": [],
                "rotation_delta": [], "open_gripper": [] }
shortest_goal_txt = 10000000000
for episode in datasetRemote:
    episode_ = {'steps': [] }
    episode = list(episode['steps'])
    goal_img = cv2.resize(np.array(episode[-1]['observation']['image'], dtype=np.float32), (image_shape[0], image_shape[1]))  
    for i in range(len(episode)): ## Resize images to reduce computation
        obs = cv2.resize(np.array(episode[i]['observation']['image'], dtype=np.float32), (image_shape[0], image_shape[1])) 
        goal = episode[i]['observation']['natural_language_instruction'].numpy().decode()
        # action = torch.as_tensor(action) # grab first dimention
        dataset_tmp["img"].append(obs)
        dataset_tmp["action"].append(np.array(episode[i]['action']['world_vector']))
        dataset_tmp["rotation_delta"].append(np.array(episode[i]['action']['rotation_delta']))
        dataset_tmp["open_gripper"].append(np.array(episode[i]['action']['open_gripper']))
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

dataset = {"train": dataset_tmp} 

from datasets import Dataset
ds = Dataset.from_dict(dataset_tmp)
print("Dataset: ", ds)
ds.push_to_hub("gberseth/mini-oxe")
