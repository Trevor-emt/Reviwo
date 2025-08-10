import random
import numpy as np
import os

import random  
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from collect_data.utils.metaworld_env import CameraConfig
def sample_azimuth(sample_num):
    sampled_azimuths = []
    for i in range(sample_num):
        while True:
            azimuth = int(random.random() * 180)
            if azimuth < 45:
                pass
            elif azimuth > 135:
                azimuth += 180
            else:
                azimuth += 90
            if not azimuth in sampled_azimuths:
                break
        sampled_azimuths.append(azimuth)
    return sampled_azimuths
        
def sample_distance(sample_num):
    sampled_distances = []
    for i in range(sample_num):
        distance = round(random.random() * 0.3 + 0.3, 1)
        sampled_distances.append(distance)
    return sampled_distances
    
def sample_elevation(sample_num):
    sampled_elevations = []
    for i in range(sample_num):
        elevation = round(-random.random() * 2 - 20, 1)
        sampled_elevations.append(elevation)
    return sampled_elevations
        
def sample_lookat(sample_num):
    sampled_lookats = []
    for i in range(sample_num):
        lookat0 = random.random() * 0.05
        lookat1 = random.random() * 0.05 + 0.5
        lookat2 = random.random() * 0.05
        lookat = np.array([lookat0, lookat1, lookat2])
        sampled_lookats.append(lookat)
    return sampled_lookats
    
def sample_camera_config(sample_num: int=100):
    sampled_azimuths = [22.5, 337.5, 202.5] + sample_azimuth(sample_num - 3)
    sampled_distances = [1.5, 1.5, 1.5] + sample_distance(sample_num - 3)
    sampled_elevations = [-20, -20, -20] + sample_elevation(sample_num - 3)
    sampled_lookats = [np.array([0, 0.5, 0]) for _ in range(3)] + sample_lookat(sample_num - 3)
    sampled_camera_configs = []
    for i in range(sample_num):
        camera_config = CameraConfig()
        camera_config.azimuth = sampled_azimuths[i]
        camera_config.distance = sampled_distances[i]
        camera_config.elevation = sampled_elevations[i]
        camera_config.lookat = sampled_lookats[i]
        sampled_camera_configs.append(camera_config)
    return sampled_camera_configs

def world_model_training_view_generation():
    sampled_azimuths = [22.5, 337.5, 202.5]
    sampled_distances = [1.5, 1.5, 1.5]
    sampled_elevations = [-20, -20, -20]
    lookats = [np.array([0, 0.5, 0]) for _ in range(3)]
    sampled_camera_configs = []
    for i in range(1):
        camera_config = CameraConfig()
        camera_config.azimuth = sampled_azimuths[i]
        camera_config.distance = sampled_distances[i]
        camera_config.elevation = sampled_elevations[i]
        camera_config.lookat = lookats[i]
        sampled_camera_configs.append(camera_config)
    return sampled_camera_configs

def multi_view_generation(view_num, seed=None):
    if seed is not None:
        random.seed(seed)
    else:
        random.seed(7)
    sampled_distances = [1.5 for _ in range(view_num)]
    lookats = [np.array([0, 0.5, 0]) for _ in range(view_num)]
    sampled_azimuths = sample_azimuth(view_num)
    sampled_elevations = sample_elevation(view_num)
    
    sampled_camera_configs = []
    for i in range(view_num):
        camera_config = CameraConfig()
        camera_config.azimuth = sampled_azimuths[i]
        camera_config.distance = sampled_distances[i]
        camera_config.elevation = sampled_elevations[i]
        camera_config.lookat = lookats[i]
        sampled_camera_configs.append(camera_config)
    return sampled_camera_configs
    
    