# RoboticArmSimulator

## Preparation

### Prerequisites

* python 3.6

### Enviromental Installation
```
$sudo apt-get update
$sudo apt-get install cmake 
$sudo apt-get install zlib1g-dev
$sudo apt install aptitude
$sudo aptitude install libopenmpi-dev 
$sudo aptitude install python3-dev
$pip install pybullet
$pip install gym
$conda install -c conda-forge imageio
$conda install tensorflow=1.14
$pip install stable-baselines
$pip install tqdm
```
### Data Preparation
```
$mkdir data
$cd data
$ln -s $ShapeNet . 
```

## Run
```
$python baselines/train_tm700_cam_grasping.py --gym_env ‘rgbd’ --algorithm ‘DQN’ --discrete True --lr 0.0001
```
