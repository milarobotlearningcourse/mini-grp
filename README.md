
# octo-mini

Minimialist reimplimentation of the Octo Generalist Robotics Policy.

## Install

'''module load cudatoolkit/11.8 miniconda/3'''

conda create -n roble python=3.10
conda activate roble
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia 

pip install torch==2.4.0
pip install hydra-submitit-launcher --upgrade

### Install MilaTools

pip install milatools==0.1.14 decorator==4.4.2 moviepy==1.0.3

## Dataset

https://rail-berkeley.github.io/bridgedata/

## Install SimpleEnv

Prerequisites:

    CUDA version >=11.8 (this is required if you want to perform a full installation of this repo and perform RT-1 or Octo inference)
    An NVIDIA GPU (ideally RTX; for non-RTX GPUs, such as 1080Ti and A100, environments that involve ray tracing will be slow). Currently TPU is not supported as SAPIEN requires a GPU to run.

Create an anaconda environment:

```
conda create -n simpler_env python=3.10 (any version above 3.10 should be fine)
conda activate simpler_env
```

Clone this repo:

```
git clone https://github.com/simpler-env/SimplerEnv --recurse-submodules
```

Install numpy<2.0 (otherwise errors in IK might occur in pinocchio):

```
pip install numpy==1.24.4
```

Install ManiSkill2 real-to-sim environments and their dependencies:

```
cd {this_repo}/ManiSkill2_real2sim
pip install -e .
```

Install this package:

```
cd {this_repo}
pip install -e .
```


### License

MIT
