import copy
from typing import List, Optional, Union
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader

from .multiview_dataset import ImageDataloaderWithPartialView
from .oxe_dataset import OXEDataLoader


class UnionImageDatasetForPytorch(Dataset):
    def __init__(self, 
                 image_dataloaders: List[Union[ImageDataloaderWithPartialView, OXEDataLoader]], 
                 batch_size: int, 
                 camera_num: int,
                 seed: int, 
                 world_size: int = 100
                 ):
        
        random.seed(seed)  # 设置 Python 的随机种子
        np.random.seed(seed)  # 设置 NumPy 的随机种子
        torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
        
        self.image_dataloaders = image_dataloaders
        self.world_size = world_size
        self.batch_size = batch_size
        self.camera_num = camera_num
        
        self.data = [[dataloader.sample(batch_size=batch_size, camera_num=camera_num) for dataloader in image_dataloaders] for _ in range(world_size)]
        
        self.idx_list = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self.idx_list.append(idx)
        return_item = copy.deepcopy(self.data[idx])
        if len(self.idx_list) == len(self.data):
            self.idx_list = []
            self.data = [[dataloader.sample(batch_size=self.batch_size, camera_num=self.camera_num) for dataloader in self.image_dataloaders] for _ in range(self.world_size)]                        
        return tuple(return_item)