import sys
sys.path.append(".")

import torch

import os
import os.path as osp

import numpy as np

from nupic.embodied.soft_modularization.torchrl.utils import get_args
from nupic.embodied.soft_modularization.torchrl.utils import get_params

from nupic.embodied.soft_modularization.torchrl.utils import Logger

args = get_args()
params = get_params(args.config)

import nupic.embodied.soft_modularization.torchrl.policies as policies
import nupic.embodied.soft_modularization.torchrl.networks as networks
from nupic.embodied.soft_modularization.torchrl.algo import MTSAC

from nupic.embodied.soft_modularization.torchrl.collector.para.async_mt import AsyncMultiTaskParallelCollectorUniform

from nupic.embodied.soft_modularization.torchrl.replay_buffers.shared import AsyncSharedReplayBuffer

from nupic.embodied.soft_modularization.metaworld_utils.meta_env import get_meta_env

import random

def experiment(args):

    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")

    env, cls_dicts, cls_args = get_meta_env( params['env_name'], params['env'], params['meta_env'])

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.backends.cudnn.deterministic=True
    
    buffer_param = params['replay_buffer']

    experiment_name = os.path.split( os.path.splitext( args.config )[0] )[-1] if args.id is None \
        else args.id
    logger = Logger( experiment_name , params['env_name'], args.seed, params, args.log_dir )

    params['general_setting']['env'] = env
    params['general_setting']['logger'] = logger
    params['general_setting']['device'] = device

    params['net']['base_type']=networks.MLPBase

    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    from nupic.embodied.soft_modularization.torchrl.networks.init import normal_init

    example_ob = env.reset()
    example_embedding = env.active_task_one_hot

    pf = policies.ModularGuassianGatedCascadeCondContPolicy(
        input_shape=env.observation_space.shape[0],
        em_input_shape=np.prod(example_embedding.shape),
        output_shape=2 * env.action_space.shape[0],
        **params['net'])

    if args.pf_snap is not None:
        pf.load_state_dict(torch.load(args.pf_snap, map_location='cpu'))

    qf1 = networks.FlattenModularGatedCascadeCondNet(
        input_shape=env.observation_space.shape[0] + env.action_space.shape[0],
        em_input_shape=np.prod(example_embedding.shape),
        output_shape=1,
        **params['net'])
    qf2 = networks.FlattenModularGatedCascadeCondNet( 
        input_shape=env.observation_space.shape[0] + env.action_space.shape[0],
        em_input_shape=np.prod(example_embedding.shape),
        output_shape=1,
        **params['net'])

    if args.qf1_snap is not None:
        qf1.load_state_dict(torch.load(args.qf2_snap, map_location='cpu'))
    if args.qf2_snap is not None:
        qf2.load_state_dict(torch.load(args.qf2_snap, map_location='cpu'))
    
    example_dict = { 
        "obs": example_ob,
        "next_obs": example_ob,
        "acts": env.action_space.sample(),
        "rewards": [0],
        "terminals": [False],
        "task_idxs": [0],
        "embedding_inputs": example_embedding
    }

    replay_buffer = AsyncSharedReplayBuffer(int(buffer_param['size']),
            args.worker_nums
    )
    replay_buffer.build_by_example(example_dict)

    params['general_setting']['replay_buffer'] = replay_buffer

    epochs = params['general_setting']['pretrain_epochs'] + \
        params['general_setting']['num_epochs']

    print(env.action_space)
    print(env.observation_space)
    params['general_setting']['collector'] = AsyncMultiTaskParallelCollectorUniform(
        env=env, pf=pf, replay_buffer=replay_buffer,
        env_cls = cls_dicts, env_args = [params["env"], cls_args, params["meta_env"]],
        device=device,
        reset_idx=True,
        epoch_frames=params['general_setting']['epoch_frames'],
        max_episode_frames=params['general_setting']['max_episode_frames'],
        eval_episodes = params['general_setting']['eval_episodes'],
        worker_nums=args.worker_nums, eval_worker_nums=args.eval_worker_nums,
        train_epochs = epochs, eval_epochs= params['general_setting']['num_epochs']
    )
    params['general_setting']['batch_size'] = int(params['general_setting']['batch_size'])
    params['general_setting']['save_dir'] = osp.join(logger.work_dir,"model")
    agent = MTSAC(
        pf = pf,
        qf1 = qf1,
        qf2 = qf2,
        task_nums=env.num_tasks,
        **params['sac'],
        **params['general_setting']
    )
    agent.train()

if __name__ == "__main__":
    experiment(args)
