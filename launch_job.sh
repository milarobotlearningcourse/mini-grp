#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=23:59:00

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# Ensure only anaconda/3 module loaded.
module --quiet purge
# This example uses Conda to manage package dependencies.
# See https://docs.mila.quebec/Userguide.html#conda for more information.
module load cudatoolkit/12.1
module load miniconda/3

# Creating the environment for the first time:
# conda create -y -n pytorch python=3.9 pytorch torchvision torchaudio \
#     pytorch-cuda=11.7 -c pytorch -c nvidia
# Other conda packages:
# conda install -y -n pytorch -c conda-forge rich tqdm

# Activate pre-existing environment.
conda activate roble
 
# Fixes issues with MIG-ed GPUs with versions of PyTorch < 2.0
unset CUDA_VISIBLE_DEVICES
 
# Execute Python script
python vit-plus-goalIMG-64.py --multirun
# Call main.py with all arguments passed to this script.
# This allows you to call the script many times with different arguments.
# Quotes around $@ prevent splitting of arguments that contain spaces.
