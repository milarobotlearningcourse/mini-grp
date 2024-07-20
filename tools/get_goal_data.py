import cv2
# import jax
# import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
import rlds, numpy as np
import mediapy as media
from PIL import Image
from IPython import display

# create RLDS dataset builder
builder = tfds.builder_from_directory(builder_dir='gs://gresearch/robotics/bridge/0.1.0/')
dataset = builder.as_dataset(split='train[:100]')

# read it in to inspect it
robot_text = open('dataset/goal_action_info.txt', 'w', encoding='utf-8') 

import numpy as np
for episode in dataset:
    steps = list(episode['steps'])
    true_action = np.concatenate((
            np.array(steps[0]['action']['world_vector']).astype(np.float32),
            np.array(steps[0]['action']['rotation_delta']).astype(np.float32),
            np.array(steps[0]['action']['open_gripper']).astype(np.float32)[None]
        ), axis=-1
    )
    goal = steps[0]['observation']['natural_language_instruction'].numpy().decode()
    line = str(goal) + ": " + str(true_action) + "\n"
    print (line)
    robot_text.write(line)

robot_text.close()