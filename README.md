
# Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment

[![Paper](https://img.shields.io/badge/arXiv-Paper-red)](🔗 _arxiv_link_here_)  


[![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-yellow)](🔗 [Checkpoints](https://huggingface.co/MINT-SJTU/Evo-1/tree/main))  

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](🔗 [MetaWorld Dataset](https://huggingface.co/datasets/MINT-SJTU/Evo-1_MetaWorld))  

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)





## 📰 News

- [2025-11-06] Released Meta-World & LIBERO evaluation script  
- [2025-11-06] Upload model weights to HuggingFace
- [2025-11-06] Released Offical Code 

## ✅ To-Do List

- ⬜ Release RoboTwin evaluation  script and checkpoints
- ⬜ Release results of all 50 RoboTwin tasks
- ⬜ Adding Evo-1 to lerobot framework for so100


## Installation

Prepare the environment for Evo-1

```bash
# Clone this repo
git clone https://github.com/DorayakiLin/Evo_1_clean.git

# Create a Conda environment
conda create -n Evo1 python=3.10 -y
conda activate Evo1

# Install requirements
cd Evo_1
pip install -r requirements.txt

# You may need to reduce the MAX_JOBS to suit your computer
MAX_JOBS=64 pip install -v flash-attn --no-build-isolation
```

## Simulation Benchmark

### Meta-World Benchmark

#### 1 Prepare the environment for Meta-World

```bash
conda create -n metaworld python=3.10 -y
conda activate metaworld
pip install mujoco
pip install metaworld
pip install websockets
pip install opencv-python
pip install packaging
```

#### 2 Model Preparation

##### 2.1 Download Model Weight

[Link to Model Weight for Meta-World](https://huggingface.co/MINT-SJTU/Evo-1/tree/main/Evo1_Simulation_Benchmark_Checkpoints/MetaWorld/Evo1_MetaWorld_checkpoint)

##### 2.2 Modify config

Modify checkpoint dir: [Evo1_server.py#L149](Evo_1/scripts/Evo1_server.py#L149)  
(Optional) Modify server port: [Evo1_server.py#L152](Evo_1/scripts/Evo1_server.py#L152)  
(Optional) Modify client port: [mt50_evo1_client_prompt.py#L40](MetaWorld_evaluation/mt50_evo1_client_prompt.py#L40)

#### 3 Run the simulation evaluation

```bash
# Terminal 1
conda activate Evo1

cd Evo_1

python scripts/Evo1_server.py
```

```bash
# Terminal 2
conda activate metaworld

cd MetaWorld_evaluation

python mt50_evo1_client_prompt.py
```

### LIBERO Benchmark

#### 1 Prepare the environment for LIBERO

```bash
conda create -n libero python=3.8.13 -y

conda activate libero

cd LIBERO_evaluation/

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git

cd LIBERO

pip install -r requirements.txt

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -e .

pip install websockets
```

#### 2 Model Preparation

##### 2.1 Download Model Weight

[Link to Model Weight for LIBERO](https://huggingface.co/liujiting/evo1_libero/tree/main)

##### 2.2 Modify config

Modify checkpoint dir: [Evo1_server.py#L149](Evo_1/scripts/Evo1_server.py#L149)  
(Optional) Modify server port: [Evo1_server.py#L152](Evo_1/scripts/Evo1_server.py#L152)  
(Optional) Modify client port: [libero_client_4tasks.py#L23](LIBERO_evaluation/libero_client_4tasks.py#L23)  
Modify ckpt name: [libero_client_4tasks.py#L24](LIBERO_evaluation/libero_client_4tasks.py#L24)

#### 3 Run the simulation evaluation

```bash
# Terminal 1
conda activate Evo1

cd Evo_1

python scripts/Evo1_server.py
```

```bash
# Terminal 2
conda activate libero

cd LIBERO_evaluation

python libero_client_4tasks.py
```

## Training on Your Own Dataset

We support lerobot v2.1 format, please convert your data to this format.

We use MetaWorld Dataset here as an example.

```bash
cd Evo1_training_dataset/

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/DorayakiLin/metaworld_dataset_v2.1

cd metaworld_dataset_v2.1/

git lfs pull
```

#### 2 Modify config

#### 2.1 Modify config.yaml

You need to modify the [config.yaml](Evo_1_clean/Evo_1/dataset/config.yaml)

#### 2.2 Set the cache path

You need to change the [cache_dir](Evo_1/dataset/lerobot_dataset_pretrain_mp.py)

#### 3 Start Training

We only train the integration module and action expert in stage 1.

If you are training with multiple GPU, set --num_processes to the GPU number

You need to change the --run_name,--save_dir,--resume_path base on your own config.
### Setup deepspeed
```bash
accelerate config     
```
You can check this [setup guide](Evo_1_clean/deepspeed_steup_example.txt)


### Stage 1

```bash
conda activate Evo1

cd Evo_1/

accelerate launch --num_processes 1 --num_machines 1 --deepspeed_config_file ds_config.json scripts/train.py --run_name Evo1_metaworld_dataset_v2.1_stage1 --action_head flowmatching --use_augmentation --lr 1e-5 --dropout 0.2 --weight_decay 1e-3 --batch_size 16 --image_size 448 --max_steps 5000 --log_interval 10 --ckpt_interval 2500 --warmup_steps 1000 --grad_clip_norm 1.0 --num_layers 8 --horizon 50 --finetune_action_head --disable_wandb --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config.yaml --per_action_dim 24 --state_dim 24 --save_dir /your/path/checkpoints/stage1
```

### Stage 2

```bash
conda activate Evo1

cd Evo_1/

accelerate launch --num_processes 1 --num_machines 1 --deepspeed_config_file ds_config.json scripts/train.py --run_name Evo1_metaworld_dataset_v2.1_stage2 --action_head flowmatching --use_augmentation --lr 1e-5 --dropout 0.2 --weight_decay 1e-3 --batch_size 16 --image_size 448 --max_steps 5000 --log_interval 10 --ckpt_interval 2500 --warmup_steps 1000 --grad_clip_norm 1.0 --num_layers 8 --horizon 50 --finetune_vlm --finetune_action_head --disable_wandb --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config.yaml --per_action_dim 24 --state_dim 24 --save_dir /your/path/checkpoints/stage2 --resume --resume_pretrain --resume_path /your/path/checkpoints/stage1/step_5000
```

## 📚 Citatation

## 📬 Contact

If you encounter any issues or have suggestions,  
please open an issue or start a discussion on GitHub.  
We sincerely welcome your feedback and contributions.
