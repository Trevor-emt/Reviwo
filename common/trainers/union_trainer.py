import torch
import numpy as np
from collections import deque
from PIL import Image
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from common.loggers import Logger
from common.utils import (
    project_dir, 
    normalize_tensor, 
    create_adaptive_weight_map, 
    compute_similarity,
    visualize_indices,
    WeightedMSELoss
)
from common.datasets.multiview_dataset import ImageDataloaderWithPartialView
from common.datasets.oxe_dataset import OXEDataLoader
from common.datasets.union_dataset import UnionImageDatasetForPytorch
from common.trainers.multiview_trainer import MultiViewViTTrainer

class UnionMultiViewViTTrainer(MultiViewViTTrainer):
    def __init__(self, 
                 config, 
                 train_envs: list, 
                 camera_id_dict: dict,
                 camera_config_dict: dict,
                 logger: Logger, 
                 use_deepspeed: bool=False):

        super().__init__(config, train_envs, camera_id_dict, camera_config_dict, logger, use_deepspeed)

    def init_dataset(self, 
                     config, 
                     ):
        # Get the primitive dataloader
        image_dataloader = ImageDataloaderWithPartialView(self.train_envs, self.load_train_data_path, self.camera_id_dict, view_num=20)
        oxe_dataloader = OXEDataLoader(self.config.load_oxe_dir)

        # Get eval batch
        self.eval_batchs = [image_dataloader.sample(batch_size=self.batch_size, camera_num=self.camera_num, seed=config.seed+i) for i in range(100)]
        self.eval_batchs += [oxe_dataloader.sample(batch_size=self.batch_size, camera_num=self.camera_num, seed=config.seed+i) for i in range(100)]
        
        # Wrap the dataloader into pytorch dataset
        dataloaders = [image_dataloader, oxe_dataloader]
        dataloader = UnionImageDatasetForPytorch(dataloaders, config.batch_size, config.camera_num, config.seed)
        return dataloader
        
    
    def train(self):
        timestep = 0
        min_eval_error = 10000
        
        for epoch in range(self.num_epoch):
            for _, (x1, x2) in tqdm(enumerate(self.dataloader)):
                self.model.train()
                timestep += 1
                loss_dict1 = self.get_loss(x1)
                loss_dict2 = self.get_loss(x2)
                loss_dict = {k: loss_dict1[k] + loss_dict2[k] for k in loss_dict1.keys()}

                start_time = time.time()
                
                self.step_loss(loss_dict["loss"])
                
                loss_dict["training_time"] += (time.time() - start_time)
            
                # For logging
                self.logger.set_timestep(timestep)
                for k,v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    self.logger.logkv_mean("train/" + k, v)
                
                if timestep % self.config.log_step == 0 or timestep == 1:
                    print(f"iter {timestep}, the training loss is: {loss_dict['loss'].item()}")
                    
                    if timestep % self.config.vis_step == 0 or timestep == 1:
                        eval_x = torch.tensor(self.dataloader.dataset.image_dataloaders[0].sample_for_eval(batch_size=8, camera_num=1) / 127.5 - 1, dtype=torch.float).squeeze().to(self.device)
                        eval_images, latent_encoding_indices, _ = self.model.visualize(eval_x)
                        self.logger.logkv("eval/eval image", eval_images.transpose(2, 0, 1))
                        hist_image = visualize_indices(latent_encoding_indices, save_dir=project_dir)
                        self.logger.logkv("eval/latent_encoding frequency hist", hist_image.transpose(2, 0, 1))

                        eval_x = torch.tensor(self.dataloader.dataset.image_dataloaders[1].sample_for_eval(batch_size=8, camera_num=1) / 127.5 - 1, dtype=torch.float).squeeze().to(self.device)
                        eval_images, latent_encoding_indices, _ = self.model.visualize(eval_x)
                        self.logger.logkv("eval/eval image for OXE", eval_images.transpose(2, 0, 1))
                        hist_image = visualize_indices(latent_encoding_indices, save_dir=project_dir)
                        self.logger.logkv("eval/latent_encoding frequency hist for OXE", hist_image.transpose(2, 0, 1))
                    
                    if timestep % self.config.eval_step == 0 or timestep == 1:
                        eval_error = self.test()
                        self.logger.logkv("eval/eval error", eval_error)
                        if eval_error < min_eval_error:
                            min_eval_error = eval_error
                            self.save_checkpoint(save_dir=self.logger.model_dir)

                    self.logger.dumpkvs()
                    
                if timestep % self.config.save_step == 0 or timestep == 1:
                    self.logger.save_model(self.save_checkpoint, 3, save_dir=f"step_{timestep}")

            self.lr_scheduler.step(timestep) 
