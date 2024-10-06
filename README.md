# Model-based Policy Optimization with Trajectory Self-correction and Q-function Discriminator

This repository contains a TensorFlow implementation of [SC-QD-MBPO]

期刊：Applied Intelligence


## Requirements

1. Install [MuJoCo 1.50](https://www.roboti.us/index.html) at `~/.mujoco/mjpro150` and copy your license key to `~/.mujoco/mjkey.txt`
2. `pip install -r requirements.txt`

## Running
Configuration files can be found in [`config/`](config). 

Run
```
python main.py --config=config.ant
```



## Acknowledgments
The code implementation is mainly modified based on the [MBPO](https://github.com/JannerM/mbpo) and [AMPO](https://github.com/RockySJ/ampo) codebase.
