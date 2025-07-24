import metaworld
import numpy as np
import argparse
import os
import pickle
import gymnasium as gym1
import sys
project_dir = str(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_dir)
from utils.collect_expert_dataset import collect_multi_view_episode as collect_episode
from gymnasium.envs import register
from metaworld import Task, policies
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
from utils.evaluate import EvalMetricTracker
import random
from view_generator import multi_view_generation as view_generation

from common.utils import ALL_ENVIRONMENTS

for env_name, env_cls in ALL_V2_ENVIRONMENTS.items():
    ID = f"mw_{env_name}"
    register(id=ID, entry_point="utils.metaworld_env:SawyerEnv4MultiCameraConfig", kwargs={"env_name": env_name})

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="bin-picking-v2", help="A Metaworld task name, like drawer-open-v2")
    parser.add_argument("--noise_type", type=str, default="no_noise")
    parser.add_argument("--expert_epoch", type=int, default=0)
    parser.add_argument("--non_expert_epoch", type=int, default=2)
    parser.add_argument("--random_expert_epoch", type=int, default=0)
    parser.add_argument("--gaussian_expert_epoch", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.1, help="magnitude of gaussian noise.")
    parser.add_argument("--log_path", type=str, default=project_dir + "/logs", help="output path")
    parser.add_argument("--save_trajectory_path", type=str, default=project_dir + "/data/metaworld", help="output path")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    metric_dict = {}

    flag = True
    
    if not os.path.exists(args.save_trajectory_path):
        os.makedirs(args.save_trajectory_path)   
    if not os.path.exists(os.path.join(args.save_trajectory_path, f"camera_configs.pkl")):
        camera_ids = [i for i in range(20)]
        camera_configs = view_generation(view_num=20)
        with open(os.path.join(args.save_trajectory_path, f"camera_configs.pkl"), 'wb') as f:
            pickle.dump((camera_ids, camera_configs), f)
    else:
        with open(os.path.join(args.save_trajectory_path, f"camera_configs.pkl"), 'rb') as f:
            camera_ids, camera_configs = pickle.load(f)
        print("Load the old camera config successfully!")
    
    for env_name in ALL_ENVIRONMENTS:
        # test
        # env_name = "door-close-v2"

        # if env_name == "shelf-place-v2":
        #     continue
        print(f"Start collect data for env {env_name}")
        if os.path.exists(os.path.join(args.save_trajectory_path, f"traj_{env_name}.pkl")):
            print("The env has been collected! Pass! ")
            continue

        args.env = env_name
    
        env = gym1.make("mw_" + env_name, 
                       camera_ids=camera_ids,
                       camera_configs=camera_configs,
                       collect_data=True,
                       use_camera=True,
                       max_eps_step=128
                       )
        # env.set_camera_ids_configs(camera_ids, camera_configs)
        observable_env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name + "-goal-observable"]()
        # Create the expert policy
        policy_name = "".join([s.capitalize() for s in env_name.split("-")])
        policy_name = policy_name.replace("PegInsert", "PegInsertion")
        policy_name = "Sawyer" + policy_name + "Policy"
        policy = vars(policies)[policy_name]()
        metric_tracker = EvalMetricTracker()
        trajectory_list = []
        
        ### Round 1
        for i in range(args.expert_epoch):
            obs = env.reset()

            # Get the the random vector
            _last_rand_vec = env.unwrapped._last_rand_vec
            data = dict(rand_vec=_last_rand_vec)
            data["partially_observable"] = False
            data["env_cls"] = type(env.unwrapped._env)
            task = Task(env_name=args.env, data=pickle.dumps(data))  # POTENTIAL ERROR
            observable_env.set_task(task)

            trajectory = collect_episode(
                env,
                observable_env,
                policy,
                metric_tracker,
                epsilon=args.epsilon,
                noise_type='gaussian',
                init_obs=obs,
                must_success=True
            )
            trajectory_list.append(trajectory)
            print(f"Env {env_name} finished collect {len(trajectory_list)} th trajectory, the length is:", len(trajectory["state"]))
        
        ### Round 2
        observable_env._freeze_rand_vec = False
        for i in range(args.gaussian_expert_epoch):
            obs = env.reset()

            # Get the the random vector
            _last_rand_vec = env.unwrapped._last_rand_vec
            data = dict(rand_vec=_last_rand_vec)
            data["partially_observable"] = False
            data["env_cls"] = type(env.unwrapped._env)
            task = Task(env_name=args.env, data=pickle.dumps(data))  # POTENTIAL ERROR
            observable_env.set_task(task)

            trajectory = collect_episode(
                env,
                observable_env,
                policy,
                metric_tracker,
                epsilon=args.epsilon,
                noise_type='gaussian',
                init_obs=obs,
            )
            trajectory_list.append(trajectory)
            
            print(f"Env {env_name} finished collect {len(trajectory_list)} th trajectory, the length is:", len(trajectory["state"]))
        
        ### Round 3
        for i in range(args.random_expert_epoch):
            obs = env.reset()

            # Get the the random vector
            _last_rand_vec = env.unwrapped._last_rand_vec
            data = dict(rand_vec=_last_rand_vec)
            data["partially_observable"] = False
            data["env_cls"] = type(env.unwrapped._env)
            task = Task(env_name=args.env, data=pickle.dumps(data))  # POTENTIAL ERROR
            observable_env.set_task(task)

            trajectory = collect_episode(
                env,
                observable_env,
                policy,
                metric_tracker,
                epsilon=args.epsilon,
                noise_type='random',
                init_obs=obs,
            )
            trajectory_list.append(trajectory)
            
            print(f"Env {env_name} finished collect {len(trajectory_list)} th trajectory, the length is:", len(trajectory["state"]))
        
        ### Round 4
        obs_env_names = [name[: -len("-goal-observable")] for name in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys() if name != "shelf-place-v2-goal-observable"]
        for i in range(args.non_expert_epoch):
            obs = env.reset()
            
            obs_env_name = random.choice(obs_env_names)
            observable_env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[obs_env_name + "-goal-observable"]()

            # Create the expert policy for this env.
            policy_name = "".join([s.capitalize() for s in obs_env_name.split("-")])
            policy_name = policy_name.replace("PegInsert", "PegInsertion")
            policy_name = "Sawyer" + policy_name + "Policy"
            policy = vars(policies)[policy_name]()

            trajectory = collect_episode(
                env,
                observable_env,
                policy,
                metric_tracker,
                epsilon=args.epsilon,
                noise_type='random',
                init_obs=obs,
            )
            trajectory_list.append(trajectory)
            
            print(f"Env {env_name} finished collect {len(trajectory_list)} th trajectory, the length is:", len(trajectory["state"]))
           
        with open(os.path.join(args.save_trajectory_path, f"traj_{env_name}.pkl"), 'wb') as f:
            pickle.dump(trajectory_list, f)
        
