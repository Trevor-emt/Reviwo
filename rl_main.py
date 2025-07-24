import os
import pickle
import numpy as np
import argparse
# import gym
import gymnasium as gym
import random
import torch
import sys
project_dir = str(os.path.dirname(__file__))
sys.path.append(str(os.path.dirname(__file__)) + "/OfflineRLKit")

from OfflineRLKit.offlinerlkit.utils.load_dataset import make_mv_data
from OfflineRLKit.offlinerlkit.nets import MLP
from OfflineRLKit.offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel, Actor
from OfflineRLKit.offlinerlkit.dynamics import EnsembleDynamics
from OfflineRLKit.offlinerlkit.utils.scaler import StandardScaler
from OfflineRLKit.offlinerlkit.utils.termination_fns import get_termination_fn
from OfflineRLKit.offlinerlkit.buffer import ReplayBuffer
from OfflineRLKit.offlinerlkit.utils.logger import Logger, make_log_dirs
from OfflineRLKit.offlinerlkit.policy_trainer import MBPolicyTrainer
from OfflineRLKit.offlinerlkit.policy import COMBOPolicy

from collect_data.view_generator import world_model_training_view_generation
from common.tokenizer_wrappers import init_multiview_tokenizer as init_tokenizer, MultiViewTokenizer

# from gym.envs import register
from gymnasium import register
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
for env_n, env_cls in ALL_V2_ENVIRONMENTS.items():
    ID = f"mw_{env_n}"
    register(id=ID, entry_point="collect_data.utils.metaworld_env:SawyerEnv4SingleViewWithLatentEmbedding", kwargs={"env_name": env_n})
    
def set_cuda():
    try:
        import pynvml, numpy as np, os

        pynvml.nvmlInit()
        deviceCount = pynvml.nvmlDeviceGetCount()
        print(f'available cuda count: {deviceCount}')
        deviceMemory = []
        for i in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            deviceMemory.append(mem_info.free)
        deviceMemory = np.array(deviceMemory, dtype=np.int64)
        maxMemory = np.max(deviceMemory) / 1024 / 1024
        if maxMemory > 2000:
            # best_device_index = np.argmax(deviceMemory)
            best_device_index = 0
        else:
            best_device_index = -1

        print("Best cuda:", best_device_index, "remain memory:", maxMemory)
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(best_device_index)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '13245'  
        return int(best_device_index)
    except Exception as e:
        print("Error when choosing CUDA_VISIBLE_DEVICES, skip this procedure:", e)
        return

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", 
        type=str, 
        default="button-press-v2", 
        help="A Metaworld task name, like drawer-open-v2"
        )
    
    parser.add_argument("--load_tokenizer_path", 
        type=str, 
        default=project_dir + "/checkpoints/multiview_v0/model.pth", 
        help="The load dir of the raw image data"
        )

    
    parser.add_argument("--load_raw_data_path", 
        type=str, 
        default=project_dir + "/data/world_model_training/traj_{env_name}.pkl", 
        help="The load dir of the raw image data"
        )
    
    parser.add_argument("--save_preprocessed_data_path", 
        type=str, 
        default=project_dir + "/data/world_model_training/traj_latent_embedding_{env_name}_{ckpt_name}.pkl", 
        help="The save dir of data with latent embedding"
        )
    
    parser.add_argument("--use_latent_embedding_data", 
        type=bool, 
        default=True,
        help="Whether we use the latent embedding of the input image or purely use the vector state as obeservation space"
        )

    parser.add_argument("--episode_worker", 
        type=int, 
        default=3,
        help="how many episodes each worker should handle"
        )

    parser.add_argument("--env_mode", 
        type=str, 
        default='normal',
        help="how env view move"
        )
    
    parser.add_argument("--max_eps_step", 
        type=int, 
        default=256,
        help="env max step for an episode"
        )

    
    parser.add_argument("--use_reward_shaping", 
        type=bool, 
        default=True,
        help="Whether we use reward_shaping"
        )

    parser.add_argument("--success_only", 
        type=bool, 
        default=True,
        help="Whether we only use success traj"
        )
    
    parser.add_argument("--single_camera", 
        type=bool, 
        default=True,
        help="Whether we only use camera_0 traj"
        )
    
    parser.add_argument("--camera_change", 
        type=float, 
        default=-10,
        help="change of camera for env_mode: distance, elevation"
        )
    
    parser.add_argument("--dir_base", 
        type=str, 
        default=project_dir + '/logs/rl',
        help="place of logging dir"
        )

    parser.add_argument("--algo-name", type=str, default="combo")
    parser.add_argument("--task", type=str, default="button-press-v2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[2048, 512, 256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--cql-weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-q-backup", type=bool, default=False)
    parser.add_argument("--deterministic-backup", type=bool, default=True)
    parser.add_argument("--with-lagrange", type=bool, default=False)
    parser.add_argument("--lagrange-threshold", type=float, default=10.0)
    parser.add_argument("--cql-alpha-lr", type=float, default=3e-4)
    parser.add_argument("--num-repeat-actions", type=int, default=10)
    parser.add_argument("--uniform-rollout", type=bool, default=False)
    parser.add_argument("--rho-s", type=str, default="mix", choices=["model", "mix"])

    parser.add_argument("--dynamics-lr", type=float, default=1e-3)
    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--dynamics-weight-decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--n-ensemble", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=1000)
    parser.add_argument("--rollout-length", type=int, default=1)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.5)
    parser.add_argument("--load-dynamics-path", type=str, default=None)

    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    return args

def preprocess_dataset(
    tokenizer: MultiViewTokenizer, 
    env_name: str,
    ckpt_name: str,
    load_raw_data_path: str,
    save_preprocessed_data_path: str
    ) -> list:
    load_raw_data_path = load_raw_data_path.format(env_name=env_name)
    save_preprocessed_data_path = save_preprocessed_data_path.format(env_name=env_name, ckpt_name=ckpt_name)
    with open(load_raw_data_path, "rb") as f:
        raw_data = pickle.load(f)
        
    # raw_data = raw_data[:20]
        
    print(f"Generating latent embedding data: {save_preprocessed_data_path}")
    processed_data = tokenizer.preprocess_image(raw_data)
    
    with open(save_preprocessed_data_path, "wb") as f:
        pickle.dump(processed_data, f)
    print(f"The first time loading the data, the generated latent embedding data has been saved successfully to {save_preprocessed_data_path}")
    return processed_data


def get_combo_policy(args, env) -> tuple[COMBOPolicy, torch.optim.lr_scheduler.CosineAnnealingLR, EnsembleDynamics]:
    # create dynamics
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=np.prod(args.obs_shape),
        action_dim=args.action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        device=args.device
    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=args.dynamics_lr
    )
    scaler = StandardScaler()
    termination_fn = get_termination_fn(task=args.task)
    dynamics = EnsembleDynamics(
        dynamics_model,
        dynamics_optim,
        scaler,
        termination_fn
    )

    if args.load_dynamics_path:
        dynamics.load(args.load_dynamics_path)
    
    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True,
        max_mu=args.max_action
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, 1000)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)
        args.target_entropy = target_entropy
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    policy = COMBOPolicy(
        dynamics,
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        cql_weight=args.cql_weight,
        temperature=args.temperature,
        max_q_backup=args.max_q_backup,
        deterministic_backup=args.deterministic_backup,
        with_lagrange=args.with_lagrange,
        lagrange_threshold=args.lagrange_threshold,
        cql_alpha_lr=args.cql_alpha_lr,
        num_repeart_actions=args.num_repeat_actions,
        uniform_rollout=args.uniform_rollout,
        rho_s=args.rho_s
    )
    return policy, lr_scheduler, dynamics



def train(args, dataset, tokenizer):
    # create env and dataset
    camera_id = 0
    camera_config = world_model_training_view_generation()[0]
    if args.env_mode == 'novel':
        camera_config.azimuth += args.camera_change
    elif args.env_mode == 'shake':
        camera_config.azimuth -= 5
    elif args.env_mode == 'distance':
        camera_config.distance += args.camera_change
    elif args.env_mode == 'elevation':
        camera_config.elevation += args.camera_change
    
    gym_config = {
        'id': "mw_" + args.env_name,
        'use_camera': args.use_latent_embedding_data,
        'collect_data': False,
        'is_gymnasium': False,
        'max_eps_step': args.max_eps_step,
        'camera_id': camera_id,
        'camera_config': camera_config,
        'tokenizer': None
        }
    gym_config['tokenizer'] = tokenizer
    # env = gym.make(**gym_config)
    env = gym.make(**gym_config).env.env
    gym_config['tokenizer'] = None
    if args.env_mode == 'shake':
        gym_config['do_shake'] = True

    args.obs_shape = (np.array(env.reset()).shape[0], )
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    load_dynamics_model = True if args.load_dynamics_path else False

    # create policy
    if args.algo_name == 'combo':
        policy, lr_scheduler, dynamics = get_combo_policy(args=args, env=env)
        dynamics.step_clip = True
        dynamics.obs_max = dataset['observations'].max()
        dynamics.obs_min = dataset['observations'].min()
        dynamics.reward_max = dataset['rewards'].max()
        dynamics.reward_min = dataset['rewards'].min()
        print(f'dynamic use step clip, obs range:[{dynamics.obs_min}, {dynamics.obs_max}], reward range:[{dynamics.reward_min}, {dynamics.reward_max}]')
    else:
        raise NotImplementedError

    # create buffer
    real_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    real_buffer.load_dataset(dataset)
    fake_buffer = ReplayBuffer(
        buffer_size=args.rollout_batch_size*args.rollout_length*args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )

    # log
    encoder_name = 'mv'
    if not args.single_camera:
        encoder_name += '_multi'
    if ('log_a' not in args.dir_base) and ('log_re' not in args.dir_base):
        log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args), record_params=['env_mode'], encoder_name=encoder_name, dir_base=args.dir_base)
    else:
        log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args), record_params=['env_mode', 'camera_change'], encoder_name=encoder_name, dir_base=args.dir_base)
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    if args.algo_name == 'combo':
        output_config['dynamics_training_progress'] = 'csv'
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))
    gym_config['save_demo_path'] = log_dirs + '/demo/epoch_{epoch}' if 'seed_1' in log_dirs else None
    # gym_config['save_demo_path'] = '/mnt/data/lky/multiview/demo/' + log_dirs.split('log/')[-1] + '/epoch_{epoch}'

    # create policy trainer
    trainer_func_args = {"load_ckpt_path": args.load_tokenizer_path}
    if args.algo_name == 'combo':
        policy_trainer = MBPolicyTrainer(
            policy=policy,
            eval_env=env,
            real_buffer=real_buffer,
            fake_buffer=fake_buffer,
            logger=logger,
            rollout_setting=(args.rollout_freq, args.rollout_batch_size, args.rollout_length),
            epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            batch_size=args.batch_size,
            real_ratio=args.real_ratio,
            eval_episodes=args.eval_episodes,
            lr_scheduler=lr_scheduler,
            gym_config=gym_config,
            trainer_func=init_tokenizer,
            trainer_func_args=trainer_func_args,
            episode_worker=args.episode_worker,
        )
    else:
        raise NotImplementedError

    # train
    if (not load_dynamics_model) and (args.algo_name == 'combo'):
        print("Start training dynamic model!")
        dynamics.train(real_buffer.sample_all(), logger, max_epochs_since_update=5)
        print("Dynamic model trained successfully!")
    
    policy_trainer.train()
        
    
def main():
    args = get_args()
    DEVICE_RANK = set_cuda()
    args.task = args.env_name
    # assert args.env_mode in ["normal", "novel", "shake"], f'unsupported env_mode: {args.env_mode}'

    mv_embedding_tokenizer = init_tokenizer(args.load_tokenizer_path)

    ckpt_name = "_".join(args.load_tokenizer_path.split('/')[-3:])

    if os.path.exists(args.save_preprocessed_data_path.format(env_name=args.env_name, ckpt_name=ckpt_name)):
        with open(args.save_preprocessed_data_path.format(env_name=args.env_name, ckpt_name=ckpt_name), "rb") as f:
            data = pickle.load(f)
    else:
        data = preprocess_dataset(mv_embedding_tokenizer, args.env_name, ckpt_name, args.load_raw_data_path, args.save_preprocessed_data_path)
    offline_dataset = make_mv_data(data, args, args.use_reward_shaping, args.success_only, args.single_camera)
    offline_dataset['rewards'] = offline_dataset['rewards'] / 10
    print("Initialize the offline dataset under the latent space successfully!")
    print(f"The range of the obs is: {offline_dataset['observations'].min()} to {offline_dataset['observations'].max()}")
    print(f"The range of the reward is: {offline_dataset['rewards'].min()} to {offline_dataset['rewards'].max()}")

    train(args, offline_dataset, mv_embedding_tokenizer)
    
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
