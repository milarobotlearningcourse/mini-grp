
# octo-mini

Minimialist reimplimentation of the Octo Generalist Robotics Policy.

## Install

'''module load cudatoolkit/11.8 miniconda/3'''

conda create -n roble python=3.10
conda activate roble
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia 

pip install torch==2.4.0

### Install MilaTools

pip install milatools decorator==4.4.2 moviepy==1.0.3

## Dataset

https://rail-berkeley.github.io/bridgedata/

### License

MIT
