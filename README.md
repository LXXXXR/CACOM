# Context-aware Communication for Multi-agent Reinforcement Learning

This is the implementation of our paper "[Context-aware Communication for Multi-agent Reinforcement Learning](https://arxiv.org/abs/2312.15600)" in AAMAS 2024. This repo is based on the open-source [pymarl](https://github.com/oxwhirl/pymarl) framework, and please refer to that repo for more documentation.

## Installation instructions

Set up StarCraft II and SMAC:

```shell
bash install_sc2.sh
```

```bash
export SC2PATH=[Your SC2 folder like /abc/xyz/3rdparty/StarCraftII]
```

Install Python environment with conda:

```bash
conda create -n cacom python=3.7 -y
conda activate pymarl
```

then install with `requirements.txt` using pip:

```bash
pip install -r requirements.txt
```

## Run an experiment 

```shell
python src/main.py --config=[Algorithm name] --env-config=[Env name] --exp-config=[Experiment name]
```

The config files are all located in `src/config`.

`--config` refers to the config files in `src/config/algs`.
`--env-config` refers to the config files in `src/config/envs`.
`--exp-config` refers to the config files in `src/config/exp`. If you want to change the configuration of a particular experiment, you can do so by modifying the yaml file here.

All results will be stored in the `work_dirs` folder.

For example, run CACOM on MMM3:

```
python src/main.py --exp-config=mmm3_s0 --config=cacom --env-config=sc2
```

## Citing

If you use this code in your research or find it helpful, please consider citing our paper:
```
@article{li2024context,
  title={Context-aware Communication for Multi-agent Reinforcement Learning},
  author={Li, Xinran and Zhang, Jun},
  booktitle={accepted by International Conference on Autonomous Agents and Multiagent Systems (AAMAS)},
  year={2024}
}
```

