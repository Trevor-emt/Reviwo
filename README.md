# [ICLR 2025] Learning View-invariant World Models for Visual Robotic Manipulation

## 🔧 Python Environment Configuration
1. Make sure you have installed mujoco-210.

2. Create a conda env with:
```bash
conda create -n reviwo python=3.9
conda activate reviwo
```

3. Install the required packages with:
```bash
pip install -r requirements.txt
```


## 📊 Dataset and Checkpoint
1. Download the OXE dataset and VIE's checkpoint from:[BaiduNetdisk]https://pan.baidu.com/s/11OHKx8fcqR0q4rwaXh7jog?pwd=g3at.  

2. Put "model.pth" file into "/checkpoints/multiview_v0" in your local directory. Put "openx" file into "/data/openx" in your local directory.

## 🚀 View-invariant Encoder Training
1. Collect the multi-view data from Metaworld with the following command.
```bash
python collect_data/collect_multi_view_data.py
```

2. Train the view-invariant encoder with the collected data from Metaworld by running the following code, the configs of training is referred to path`configs/config.yaml`.
```bash
python tokenizer_main.py --training_style tokenizer
```

3. To train the view-invariant encoder with the collected data from Metaworld along with part of data from OXE, run the following command:
```bash
python tokenizer_main.py --training_style union_tokenizer
```

## 🦾 Running COMBO with the learnt view-invariant encoder
1. Collect the single-view data for COMBO with the following command:
```bash
python collect_data/collect_world_model_training_data.py --env_name ${your_metaworld_env_name}
```

2. Running COMBO with the following command. The default checkpoint is "checkpoints/multiview_v0/model.pth" with the default model config in "configs/config.yaml". We provide three settings for evaluation:
* Training View: 
```bash
python rl_main.py --env_name ${your_metaworld_env_name} --env_mode "normal"
``` 
* Novel View(CIP): 
```bash
python rl_main.py --env_name ${your_metaworld_env_name} --env_mode "novel" --camera_change ${change_of_azimuth}
``` 
* Shaking View(CSH): 
```bash
python rl_main.py --env_name ${your_metaworld_env_name} --env_mode "shake"
``` 

## 😊 Acknowledgement
We would like to thank the authors of [OfflineRLKit](https://github.com/yihaosun1124/OfflineRL-Kit) for their great work and generously providing source codes, which inspired our work and helped us a lot in the implementation.

## 📚 Citation
If you find our work helpful, please consider citing:
```bibtex
@inproceedings{pang2025reviwo,
  title={Learning View-invariant World Models for Visual Robotic Manipulation},
  author={Pang, jingcheng and Tang, nan and Li, kaiyuan, and Tang, Yuting and Cai, Xin-Qiang and Zhang, Zhen-Yu and Niu, Gang and Masashi, Sugiyama and Yu, yang},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```