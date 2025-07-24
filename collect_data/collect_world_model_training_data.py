import metaworld
import numpy as np
import argparse
import os
import pickle
import gymnasium as gym
import sys
project_dir = str(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.collect_expert_dataset import collect_single_view_episode as collect_episode
from gymnasium.envs import register
from metaworld import Task, policies
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
from utils.evaluate import EvalMetricTracker
import random
from view_generator import world_model_training_view_generation as view_generation

for env_n, env_cls in ALL_V2_ENVIRONMENTS.items():
    ID = f"mw_{env_n}"
    register(id=ID, entry_point="utils.metaworld_env:SawyerEnv4SingleView", kwargs={"env_name": env_n})

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="button-press-v2", help="A Metaworld task name, like drawer-open-v2")
    parser.add_argument("--noise_type", type=str, default="no_noise")
    parser.add_argument("--expert_epoch", type=int, default=200)
    parser.add_argument("--non_expert_epoch", type=int, default=200)
    parser.add_argument("--random_expert_epoch", type=int, default=100)
    parser.add_argument("--gaussian_expert_epoch", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.1, help="magnitude of gaussian noise.")
    parser.add_argument("--log_path", type=str, default=project_dir + "/logs", help="output path")
    parser.add_argument("--save_trajectory_path", type=str, default=project_dir + "/data/world_model_training", help="output path")
    
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--end_idx", default=50, type=int)

    args = parser.parse_args()
    return args

def worker(process_id, env_name="button-press-v2"):
    
    np.random.seed(1)
    random.seed(1)
    
    args = get_args()
    args.env_name = env_name
    
    camera_configs = view_generation()
    camera_ids = [0]
    if not os.path.exists(args.save_trajectory_path):
        os.makedirs(args.save_trajectory_path) 
        
    # TRAJ_NUM = 100
    # args.expert_epoch = TRAJ_NUM // 2
    # args.gaussian_expert_epoch = TRAJ_NUM // 4
    # args.random_expert_epoch = TRAJ_NUM // 4
    # args.non_expert_epoch = TRAJ_NUM
      
    total_trajs = []
    for camera_id in camera_ids:
        camera_config = camera_configs[camera_id]
        print(f"Start collect data for process {process_id}, env {env_name}")

        env = gym.make("mw_" + env_name, 
                    camera_id=camera_id,
                    camera_config=camera_config,
                    collect_data=True,
                    use_camera=True,
                        #    save_demo_path="/home/tangnan/projects/llm_reward/demos/multi_view_demo_50tasks",
                    max_eps_step=256
                    )

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
            task = Task(env_name=args.env_name, data=pickle.dumps(data))  # POTENTIAL ERROR
            observable_env.set_task(task)

            trajectory = collect_episode(
                env,
                observable_env,
                policy,
                metric_tracker,
                epsilon=args.epsilon,
                noise_type='gaussian',
                init_obs=obs,
                must_success=False
            )
            trajectory_list.append(trajectory)
            print(f"Process {process_id} finish the {len(trajectory_list)}th trajectory for the {camera_id}th camera, the length is:", len(trajectory["state"]))             
        
        ### Round 2
        observable_env._freeze_rand_vec = False
        for i in range(args.gaussian_expert_epoch):
            obs = env.reset()

            # Get the the random vector
            _last_rand_vec = env.unwrapped._last_rand_vec
            data = dict(rand_vec=_last_rand_vec)
            data["partially_observable"] = False
            data["env_cls"] = type(env.unwrapped._env)
            task = Task(env_name=args.env_name, data=pickle.dumps(data))  # POTENTIAL ERROR
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
            print(f"Process {process_id} finish the {len(trajectory_list)}th trajectory for the {camera_id}th camera, the length is:", len(trajectory["state"]))
        
        ### Round 3
        for i in range(args.random_expert_epoch):
            obs = env.reset()

            # Get the the random vector
            _last_rand_vec = env.unwrapped._last_rand_vec
            data = dict(rand_vec=_last_rand_vec)
            data["partially_observable"] = False
            data["env_cls"] = type(env.unwrapped._env)
            task = Task(env_name=args.env_name, data=pickle.dumps(data))  # POTENTIAL ERROR
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
            print(f"Process {process_id} finish the {len(trajectory_list)}th trajectory for the {camera_id}th camera, the length is:", len(trajectory["state"]))
        
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
            print(f"Process {process_id} finish the {len(trajectory_list)}th trajectory for the {camera_id}th camera, the length is:", len(trajectory["state"]))
        
        total_trajs = total_trajs + trajectory_list
        
    return total_trajs


if __name__ == '__main__':
    args = get_args()
    final_total_trajs = worker(0, env_name=args.env_name)             
           
    with open(os.path.join(args.save_trajectory_path, f"traj_{args.env_name}.pkl"), 'wb') as f:
        pickle.dump(final_total_trajs, f)

