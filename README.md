# Reviwo

## Python Environment Configuration
1. Update the `prefix` parameter in `environment.yml`
2. Build Python environment with following command
```bash
conda env create -f environment.yml
```

## View-invariant Encoder Training
1. Collect the multi-view data from Metaworld with the following command, make sure you have installed mujoco, and we recommend using mujoco-210.
```bash
python collect_data/collect_multi_view_data.py
```

2. Train the view-invariant encoder by running, the configs of training is referred to path`configs/config.yaml`:
```bash
python tokenizer_main.py
```

## Running COMBO with the learnt view-invariant encoder
1. Collect the single-view data for COMBO with the following command:
```bash
python collect_data/collect_world_model_training_data.py --env_name ${your_metaworld_env_name}
```

2. Running COMBO with the following command. We provide three setting for evaluation:
* Training: the evaluation viewpoint is the same as the training one
```bash

``` 




1. Download Llama-2-7b-chat-hf from https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
2. Move the downloaded Llama-2-7b-chat-hf to path `base_models/llama2-hf-chat-7b`
3. Move the OfflineRL dataset to path `data/${offlinerl_dataset_name}`. We provide 2 toy datasets for testing: `data/clevr_robot.npy` and `data/meta_world.npy`
4. Update the `num_processes` parameter to the num of GPUs you want to use in `config/ds_clevr.yaml` and `config/ds_meta.yaml`
5. Update the paths and `CUDA_VISIBLE_DEVICES` in `scripts/train_clevr.sh` and `scripts/train_meta.sh`
6. fine-tune the LLM
* CLEVR-Robot
```bash
bash scripts/train_clevr.sh
```
* Meta-World
```bash
bash scripts/train_meta.sh
```

## Rollout Generation
1. Move the fine-tuned LLM to path `finetuned_models/${model_name}`
2. Generate rollouts with the fine-tuned LLM
* CLEVR-Robot
```bash
python3 src/clevr_generate.py --model_path ${model_path} --prompt_path ${prompt_path} --output_path ${output_path} --level ${level}
```
* Meta-World
```bash
python3 src/meta_generate.py --model_path ${model_path} --output_path ${output_path} --level ${level}
```
3. We provide 1 toy instruction prompt dataset for testing(Generation on Meta-World does not need dataset): `data/clevr_rephrase_prompt.npy`
```bash
python3 src/clevr_generate.py --model_path ${model_path} --prompt_path data/clevr_rephrase_prompt.npy --output_path ${output_path} --level rephrase_level
```
* Meta-World
```bash
python3 src/meta_generate.py --model_path ${model_path} --output_path ${output_path} --level rephrase_level
```

## OfflineRL Training
1. Move the imaginary dataset to path `data/${imaginary_dataset_name}`
2. Train the OfflineRL policy with the OfflineRL dataset and imaginary datast
* CLEVR-Robot
```bash
python3 src/clevr_offline_train.py --ds_type ${ds_type} --agent_name ${agent_name} --dataset_path ${dataset_path} --device ${device} --seed ${seed}
```
* Meta-World
```bash
python3 src/meta_offline_train.py --ds_type ${ds_type} --agent_name ${agent_name} --dataset_path ${dataset_path} --device ${device} --seed ${seed}
```
3. We provide 2 toy offlineRL datasets for testing: `data/clevr_robot.hdf5` and `data/meta_world.hdf5`
* CLEVR-Robot
```bash
python3 src/clevr_offline_train.py --ds_type rephrase_level --agent_name ${agent_name} --dataset_path data/clevr_robot.hdf5 --device ${device} --seed ${seed}
```
* Meta-World
```bash
python3 src/meta_offline_train.py --ds_type rephrase_level --agent_name ${agent_name} --dataset_path data/meta_world.hdf5 --device ${device} --seed ${seed}
```