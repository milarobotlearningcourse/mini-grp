## Code to fetch data and create an easy dataset.
import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm, trange
import cv2
from PIL import Image


## Model hyperparameters
image_shape = [64, 64, 3]
num_episodes = 2000 ## How many episodes to grab from the dataset for training
name = 'mini-bridge-mini64pix'

from datasets import load_dataset

# ------------
# Train and test splits
# Loading data
# create RLDS dataset builder
builder = tfds.builder_from_directory(builder_dir='gs://gresearch/robotics/bridge/0.1.0/')
datasetRemote = builder.as_dataset(split='train[:' + str(num_episodes) + ']')
dataset_tmp = {"img": [], "action": [], "goal": [], "goal_img": [],
                "rotation_delta": [], "open_gripper": [] }
for episode in datasetRemote:
    episode_ = {'steps': [] }
    episode = list(episode['steps'])
    goal_img = cv2.resize(np.array(episode[-1]['observation']['image'], dtype=np.float32), (image_shape[0], image_shape[1]))  
    for i in range(len(episode)): ## Resize images to reduce computation
        # action = torch.as_tensor(action) # grab first dimention
        obs = cv2.resize(np.array(episode[i]['observation']['image'], dtype=np.float32), (image_shape[0], image_shape[1]))
        dataset_tmp["img"].append(Image.fromarray(obs.astype('uint8') ))
        dataset_tmp["action"].append(episode[i]['action']['world_vector'])
        dataset_tmp["rotation_delta"].append(episode[i]['action']['rotation_delta'])
        dataset_tmp["open_gripper"].append([np.array(episode[i]['action']['open_gripper'], dtype=np.uint8)])
        dataset_tmp["goal"].append(episode[i]['observation']['natural_language_instruction'].numpy().decode())
        dataset_tmp["goal_img"].append(Image.fromarray(goal_img.astype('uint8') ))

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
dataset = {}
dataset["img"] = dataset_tmp["img"]
dataset["action"] = np.array(dataset_tmp["action"], dtype=np.float32)
dataset["rotation_delta"] = np.array(dataset_tmp["rotation_delta"], dtype=np.float32)
dataset["open_gripper"] = np.array(dataset_tmp["open_gripper"], dtype=np.uint8)
dataset["goal"] = dataset_tmp["goal"]
dataset["goal_img"] = dataset_tmp["goal_img"]

# dataset = {"train": dataset_tmp} 

from datasets import Dataset
import datasets
from datasets import ClassLabel, Value, Image, Features, Array2D, Array4D, Sequence, Array3D
# features = Features({
#     'goal': Value('string'),
#     'img': Image(),
#     'goal_img': Image(),
#     'rotation_delta': Array2D(shape=(1, 3), dtype="float32"),
#     'open_gripper': Array2D(shape=(1, 1), dtype="uint8"),
#     'action': Array2D(shape=(1, 3), dtype="float32"),
#     ## Sequence(feature=Value(dtype='float32', id=None), length=-1, id=None)
# })

ds = Dataset.from_dict(dataset)
# ds.add_column(name="img", column=dataset["img"])
# ds = ds.train_test_split(test_size=0.1)
print("Dataset: ", ds)
# ds = ds.with_format("np")
print("Dataset: ", ds)

new_features = ds.features.copy()
new_features["img"] = Image()
# new_features["img"] = Sequence(Array3D(shape=dataset_tmp["img"].shape[1:], dtype='uint8'))
# new_features["goal_img"] = Array3D(shape=dataset["goal_img"].shape[1:], dtype='uint8')
# new_features["action"] = Value('float')
# new_features["rotation_delta"] = Value('float')
# new_features["open_gripper"] = Value('bool')
# new_features["goal"] = Value('string'),
ds.cast(new_features)
print('Features:', ds.features)
ds.save_to_disk("datasets/" + name + ".hf")
ds.push_to_hub("gberseth/" + name)
