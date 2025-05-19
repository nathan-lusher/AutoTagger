# WSL Setup for Tensorflow processing

## Install WSL

```
wsl --unregister Ubuntu ## this destroys the existing installation
wsl --install
wsl -d Ubuntu
```

## Install python

```
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv build-essential ffmpeg ffprobe
```

## Install NVIDIA GPU modules

```
nvidia-smi
```

## Setup python virtual environment

```
cd ~
mkdir tf-gpu
cd tf-gpu

python3 -m venv tf-gpu
source tf-gpu/bin/activate
```

## Install tensorflow and prerequisites

```
pip install --upgrade pip
pip install tensorflow[and-cuda] pillow scikit-learn tqdm numpy
```

## Test GPU detection

```
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Create models directory

```
mkdir ~/tf-models
```

## Run script

```
wsl -d Ubuntu

cd ~
source tf-gpu/bin/activate
python /mnt/d/DATA/NATHAN/Code/Tagging/wsl/02_build_model.py
```

## View files from Windows

```
\\wsl$\Ubuntu\home\nathan\tf-data\
```