import ruamel.yaml as yaml
import os
import sys
project_dir = str(os.path.dirname(__file__))
sys.path.append(project_dir)

import common
from pathlib import Path
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_style", type=str, default="tokenizer", help="The implementation for training.", choices=["tokenizer", "union_tokenizer"])
    return parser.parse_args()

def pretrain_tokenizer(config, logger, use_deepspeed=False):
    from common.trainers.multiview_trainer import MultiViewViTTrainer

    train_envs = common.ALL_ENVIRONMENTS
    camera_id_dict, camera_config_dict = common.load_camera_id_config(config.load_train_data_path, train_envs)
    
    tokenizer_trainer = MultiViewViTTrainer(config, 
                                            train_envs, 
                                            camera_id_dict, 
                                            camera_config_dict, 
                                            logger, 
                                            use_deepspeed=use_deepspeed
                                            )
    tokenizer_trainer.train()

def pretrain_union_tokenizer(config, logger, use_deepspeed=False):
    from common.trainers.union_trainer import UnionMultiViewViTTrainer

    train_envs = common.ALL_ENVIRONMENTS
    camera_id_dict, camera_config_dict = common.load_camera_id_config(config.load_train_data_path, train_envs)
    
    tokenizer_trainer = UnionMultiViewViTTrainer(config, 
                                            train_envs, 
                                            camera_id_dict, 
                                            camera_config_dict, 
                                            logger, 
                                            use_deepspeed=use_deepspeed
                                            )
    tokenizer_trainer.train()

def train(config, 
          training_style: str):
    if training_style == "tokenizer":
        log_dir = common.make_log_dirs("pretrain_tokenizer", config, [])
        train_func = pretrain_tokenizer
    elif training_style == "union_tokenizer":
        log_dir = common.make_log_dirs("pretrain_tokenizer", config, [])
        train_func = pretrain_union_tokenizer
    else:
        raise NotImplementedError("Wrong training style!")
    
    output_config = {
        "consoleout_backup": "stdout",
        "tb": "tensorboard"
    }
    config.update(log_dir=log_dir)
    logger = common.Logger(log_dir, output_config)
    logger.log_hyperparameters(config)
    train_func(config, logger)
    

def main(training_style: str):  
    configs = yaml.safe_load(
        (Path(project_dir + "/configs") / "config.yaml").read_text()
    )
    config_name = "defaults"

    parsed, remaining = common.Flags(configs=[config_name]).parse(known_only=True)
    config = common.Config(configs[config_name])
    
    for name in parsed.configs:
        config = config.update(configs[name])
    config = common.Flags(config).parse(remaining, known_only=True)[0]
    
    train(config, training_style)
    
if __name__ == "__main__":
    args = get_args()
    main(training_style=args.training_style)
    

    