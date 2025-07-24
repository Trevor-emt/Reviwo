import imageio 
from PIL import Image   
import cv2
import numpy as np
import pickle
import random
import torch
import os
import copy
from torch.utils.data import Dataset, DataLoader

project_dir = str(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class OXEDataLoader:
    def __init__(self, load_dir, azimuths: list = ["side", "top", "wrist45", "wrist225"]) -> None:
        self.data_dict = {}
        self.azimuths = [item for item in azimuths]
        for azimuth in self.azimuths:
            self.data_dict[azimuth] = []
        self.load_data(load_dir)
            
    def load_data(self, load_dir):
        # load_dir = project_dir + "/data/openx_100trajs"
        for i in range(101):
            load_mp4_dir = load_dir + f"/traj_num_{i}/videos"
            length = -1
            for azimuth in self.azimuths:
                cap = cv2.VideoCapture(load_mp4_dir + f"/{azimuth}.mp4")
                frames = []
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame = np.array(Image.fromarray(frame).resize((128, 128)))
                    frames.append(frame)

                cap.release()

                self.data_dict[azimuth] += frames
                if length == -1:
                    length = len(frames)
                assert len(frames) == length
                
        cv2.destroyAllWindows()
                
        for azimuth in self.azimuths:
            assert len(self.data_dict[azimuth]) == len(self.data_dict[self.azimuths[0]]) 

            
    def sample(self, batch_size, camera_num, identical_azimuth: bool = True, seed:int=None):
        ### The sampled image batch data is: B * azimuth_num * W * H * 3
        camera_num = min(len(self.azimuths), camera_num) 
        
        if isinstance(seed, int):
            random.seed(seed)  # 设置 Python 的随机种子
            np.random.seed(seed)  # 设置 NumPy 的随机种子 
        
        if identical_azimuth:
            sampled_azimuths = random.sample(self.azimuths, camera_num)
            sampled_image_ids = random.sample(list(range(len(self.data_dict[sampled_azimuths[0]]))), batch_size)
            sampled_images = []
            for azimuth in sampled_azimuths:
                sampled_azimuth_image = np.stack([self.data_dict[azimuth][id] for id in sampled_image_ids], axis=0)
                sampled_images.append(sampled_azimuth_image)
            batch_data = np.stack(sampled_images, axis=0).transpose(1, 0, 4, 2, 3)
        else:
            batch_data = []
            sampled_image_ids = random.sample(list(range(len(self.data_dict[self.azimuths[0]]))), batch_size)
            for id in sampled_image_ids:
               sampled_azimuths = random.sample(self.azimuths, camera_num)
               images = [self.data_dict[azimuth][id] for azimuth in sampled_azimuths]
               batch_data.append(np.stack(images, axis=0))
            batch_data = np.stack(batch_data, axis=0).transpose(0, 1, 4, 2, 3)  
        # print(f"Sampled data's shape is: {batch_data.shape}")
        return batch_data
    
    def sample_with_nonequal_azimuth(self, ):
        batch_data = []
        sampled_image_ids = random.sample(list(range(len(self.data_dict[self.azimuths[0]]))), 8)
        sampled_azimuths = random.sample(self.azimuths, 8)
        for i, id in enumerate(sampled_image_ids):
            images = [self.data_dict[sampled_azimuths[i]][id]]
            batch_data.append(np.stack(images, axis=0))
        batch_data = np.stack(batch_data, axis=0).transpose(0, 1, 4, 2, 3)  
        return batch_data
    
    def sample_with_single_azimuth(self, batch_size, azimuth_id = 0):
        batch_data = []
        sampled_image_ids = random.sample(list(range(len(self.data_dict[self.azimuths[azimuth_id]]))), batch_size)
        for id in sampled_image_ids:
            sampled_azimuths = [self.azimuths[azimuth_id]]
            images = [self.data_dict[azimuth][id] for azimuth in sampled_azimuths]
            batch_data.append(np.stack(images, axis=0))
        batch_data = np.stack(batch_data, axis=0).transpose(0, 1, 4, 2, 3)  
        return batch_data
    
    def sample_for_eval(self, batch_size, camera_num):
        return self.sample(batch_size, camera_num, identical_azimuth=False)

class OXEImageDatasetForPytorch(Dataset):
    def __init__(self, 
                 image_dataloader: OXEDataLoader, 
                 batch_size: int, 
                 camera_num: int,
                 seed: int, 
                 world_size: int = 100
                 ):
        
        random.seed(seed)  # 设置 Python 的随机种子
        np.random.seed(seed)  # 设置 NumPy 的随机种子
        torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
        
        self.image_dataloader = image_dataloader
        self.world_size = world_size
        self.batch_size = batch_size
        self.camera_num = camera_num
        
        self.data = [image_dataloader.sample(batch_size=batch_size, camera_num=camera_num) for _ in range(world_size)]
        
        self.idx_list = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self.idx_list.append(idx)
        return_item = copy.deepcopy(self.data[idx])
        if len(self.idx_list) == len(self.data):
            self.idx_list = []
            self.data = [self.image_dataloader.sample(batch_size=self.batch_size, camera_num=self.camera_num) for _ in range(self.world_size)]                           
        return return_item