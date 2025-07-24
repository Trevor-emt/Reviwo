import torch
import numpy as np
from collections import deque
from PIL import Image
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from common.trainers.base_trainer import BaseTrainer
from common.loggers import Logger
from common.utils import (
    project_dir, 
    normalize_tensor, 
    create_adaptive_weight_map, 
    compute_similarity,
    visualize_indices,
    WeightedMSELoss
)
from common.models.multiview_vae import MultiViewBetaVAE
from common.datasets.multiview_dataset import ImageDatasetForPytorch, ImageDataloaderWithPartialView

class MultiViewViTTrainer(BaseTrainer):
    def __init__(self, 
                 config, 
                 train_envs: list, 
                 camera_id_dict: dict,
                 camera_config_dict: dict,
                 logger: Logger, 
                 use_deepspeed: bool=False):
        self.use_deepspeed = use_deepspeed
        self.logger = logger
        self.config = config
        self.camera_id_dict = camera_id_dict
        self.camera_config_dict = camera_config_dict
        self.loss_form = config.loss_form
        self.train_envs = train_envs
        self.num_epoch = config.num_epoch
        self.camera_num = config.camera_num
        self.batch_size = config.batch_size
            
        self.vq_coef = config.vq_coef
        self.latent_consistency_coef = config.latent_consistency_coef
        self.view_consistency_coef = config.view_consistency_coef
        self.latent_contrastive_coef = config.latent_contrastive_coef
        self.view_contrastive_coef = config.view_contrastive_coef
        self.temperature = config.temperature
        self.shuffled_v_coef = config.shuffled_v_coef
        self.shuffled_l_coef = config.shuffled_l_coef
        self.shuffled_vl_coef = config.shuffled_vl_coef
        self.load_train_data_path = config.load_train_data_path

        super().__init__(config, logger, use_deepspeed)

    def init_model(self, 
                   config):
        model = MultiViewBetaVAE(config.view_encoder_config, config.latent_encoder_config, config.decoder_config, config.view_cb_config, config.latent_cb_config, config.img_size, config.patch_size, config.fusion_style, config.use_latent_vq, config.use_view_vq, config.is_view_ae)
        return model

    def init_dataset(self, 
                     config, 
                     ):
        # Get the primitive dataloader
        image_dataloader = ImageDataloaderWithPartialView(self.train_envs, self.load_train_data_path, self.camera_id_dict, view_num=20)

        # Get eval batch
        self.eval_batchs = [image_dataloader.sample(batch_size=self.batch_size, camera_num=self.camera_num, seed=config.seed+i) for i in range(100)]
        
        # Wrap the dataloader into pytorch dataset
        image_dataloader = ImageDatasetForPytorch(image_dataloader, config.batch_size, config.camera_num, config.seed)
        return image_dataloader
        
    def get_loss(self, data):
        view_contrastive_loss = 0
        latent_contrastive_loss = 0
        training_time = 0
        start_time = time.time()
        
        _, B, A, C, H, W = data.shape
        assert H == W
        x = (data.reshape(B * A, C, H, W) / 127.5 - 1).to(self.device).float()

        batch_size, camera_num = B, A
        
        z_v, view_embed_loss, z_l, latent_embed_loss, latent_encoding_indices = self.model.encode(x)
        
        y = self.model.decode(z_v, z_l)
        
        shuffle_indices = torch.randperm(camera_num)
        shuffled_z_l = z_l.reshape(self.batch_size, camera_num, -1)[:, shuffle_indices, :].reshape(self.batch_size * camera_num, -1, z_l.shape[-1])
        y_shuffle_l = self.model.decode(z_v, shuffled_z_l)
        
        shuffled_indices_v = torch.randperm(self.batch_size)
        shuffled_z_v = z_v.reshape(self.batch_size, camera_num, -1)[shuffled_indices_v, :, :].reshape(self.batch_size * camera_num, -1, z_v.shape[-1])
        y_shuffle_v = self.model.decode(shuffled_z_v, z_l)
        y_shuffle_vl = self.model.decode(shuffled_z_v, shuffled_z_l)

        normalized_z_l = normalize_tensor(z_l).reshape(self.batch_size, camera_num, -1)
        normalized_z_v = normalize_tensor(z_v).reshape(self.batch_size, camera_num, -1)
        
        latent_contrastive_loss = 0
        view_contrastive_loss = 0
        temperature = 0.25
        lower_bound = 0.9
        contrastive_start_time = time.time()
        
        #### View Consistency
        for j in range(camera_num):
            for i in range(self.batch_size):
                positive = sum([(compute_similarity(normalized_z_v[i, j], normalized_z_v[i_, j], dim=-1, way="cosine-similarity", lower_bound=lower_bound) / temperature).exp() for i_ in range(self.batch_size) if i_!=i])
                
                negative_state_idxs = [x for x in range(camera_num) if x!=j]
                negative = (compute_similarity(normalized_z_v[i:i+1, j:j+1].repeat(self.batch_size, camera_num-1, 1), normalized_z_v[:, negative_state_idxs], dim=-1, way="cosine-similarity", lower_bound=lower_bound) / temperature).exp().sum()
                view_contrastive_loss -= (positive / (positive + negative)).log()
        
        
        #### Latent Consistency
        for i in range(self.batch_size):
            for j in range(camera_num):
                positive = sum([(compute_similarity(normalized_z_l[i, j], normalized_z_l[i, j_], dim=-1, way="cosine-similarity", lower_bound=lower_bound) / temperature).exp() for j_ in range(camera_num) if j_!=j])
                
                negative_state_idxs = [x for x in range(self.batch_size) if x!=i]
                negative = (compute_similarity(normalized_z_l[i:i+1, j:j+1].repeat(self.batch_size-1, camera_num, 1), normalized_z_l[negative_state_idxs], dim=-1, way="cosine-similarity", lower_bound=lower_bound) / temperature).exp().sum()
                latent_contrastive_loss -= (positive / (positive + negative)).log()
                    
                
        view_contrastive_loss = view_contrastive_loss / batch_size / camera_num
        latent_contrastive_loss = latent_contrastive_loss / batch_size / camera_num
                
        latent_consistency_loss = torch.mean((torch.mean(normalized_z_l, dim=1, keepdim=True) - normalized_z_l).abs())
        view_consistency_loss = torch.mean((torch.mean(normalized_z_v, dim=0, keepdim=True) - normalized_z_v).abs())
        
        if self.loss_form == "MAE":
            rec_loss = torch.abs(x - y).mean()
            shuffled_l_rec_loss = torch.abs(x - y_shuffle_l).mean()
            shuffled_v_rec_loss = torch.abs(x - y_shuffle_v).mean()
            shuffled_vl_rec_loss = torch.abs(x - y_shuffle_vl).mean()
        elif self.loss_form == "MSE":
            rec_loss = torch.mean((x - y)**2) 
            shuffled_l_rec_loss = torch.mean((x - y_shuffle_l)**2)
            shuffled_v_rec_loss = torch.mean((x - y_shuffle_v)**2)
            shuffled_vl_rec_loss = torch.mean((x - y_shuffle_vl)**2)
        elif self.loss_form == "Weighted_MSE":
            criterion = WeightedMSELoss()
            rec_loss = torch.mean((x - y)**2) 
            shuffled_l_rec_loss = torch.mean((x - y_shuffle_l)**2)
            shuffled_v_rec_loss = torch.mean((x - y_shuffle_v)**2)
            weight_map_shuffle_vl = create_adaptive_weight_map(x, y_shuffle_vl)
            shuffled_vl_rec_loss = criterion(x, y_shuffle_vl, weight_map_shuffle_vl)
        else:
            raise NotImplementedError
        vq_loss = latent_embed_loss.mean() + view_embed_loss.mean()
        loss = rec_loss + self.shuffled_l_coef * shuffled_l_rec_loss + self.shuffled_v_coef * shuffled_v_rec_loss + self.shuffled_vl_coef * shuffled_vl_rec_loss + self.vq_coef * vq_loss + self.latent_consistency_coef * latent_consistency_loss + self.view_consistency_coef * view_consistency_loss + self.latent_contrastive_coef * latent_contrastive_loss + self.view_contrastive_coef * view_contrastive_loss
        
        training_time = time.time() - start_time
        
        return {
            'loss': loss,
            'rec_loss': rec_loss,
            'shuffled_l_rec_loss': shuffled_l_rec_loss,
            'shuffled_v_rec_loss': shuffled_v_rec_loss,
            'shuffled_vl_rec_loss': shuffled_vl_rec_loss,
            'vq_loss': vq_loss,
            'view_consistency_loss': view_consistency_loss,
            'latent_consistency_loss': latent_consistency_loss,
            'view_contrastive_loss': view_contrastive_loss,
            'latent_contrastive_loss': latent_contrastive_loss,
            'training_time': training_time
        }

    
    def train(self):
        timestep = 0
        min_eval_error = 10000
        
        for epoch in range(self.num_epoch):
            for _, x in tqdm(enumerate(self.dataloader)):
                self.model.train()
                timestep += 1
                loss_dict = self.get_loss(x)

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
                        eval_x = torch.tensor(self.dataloader.dataset.image_dataloader.sample_for_eval(batch_size=8, camera_num=1) / 127.5 - 1, dtype=torch.float).squeeze().to(self.device)
                        eval_images, latent_encoding_indices, _ = self.model.visualize(eval_x)
                        self.logger.logkv("eval/eval image", eval_images.transpose(2, 0, 1))
                        hist_image = visualize_indices(latent_encoding_indices, save_dir=project_dir)
                        self.logger.logkv("eval/latent_encoding frequency hist", hist_image.transpose(2, 0, 1))
                    
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

    def test(self, ):
        self.model.eval()
        eval_errors = []
        with torch.no_grad():
            for eval_batch in self.eval_batchs:
                x = torch.tensor(eval_batch).unsqueeze(0)
                loss_dict = self.get_loss(x)
                eval_errors.append(loss_dict["loss"].item())
        return np.mean(eval_errors)
         
    def save_checkpoint(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model.save_checkpoint(save_dir + "/model.pth")
        print(f"Save model to path {save_dir} successfully!")
    
    def load_checkpoint(self, load_dir):
        self.model.load_checkpoint(load_dir + "/model.pth")
        print(f"Load model from path {load_dir} successfully!")