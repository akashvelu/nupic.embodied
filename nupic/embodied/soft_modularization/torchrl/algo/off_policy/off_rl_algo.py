import time
import numpy as np
from nupic.embodied.soft_modularization.torchrl.algo.rl_algo import RLAlgo
from nupic.torch.modules.sparse_weights import rezero_weights


class OffRLAlgo(RLAlgo):
    """
    Base RL Algorithm Framework
    """

    def __init__(self,
                 pretrain_epochs=0,
                 min_pool=0,
                 target_hard_update_period=1000,
                 use_soft_update=True,
                 tau=0.001,
                 opt_times=1,
                 **kwargs
                 ):
        super(OffRLAlgo, self).__init__(**kwargs)

        # environment relevant information
        self.pretrain_epochs = pretrain_epochs

        # target_network update information
        self.target_hard_update_period = target_hard_update_period
        self.use_soft_update = use_soft_update
        self.tau = tau

        # training information
        self.opt_times = opt_times
        self.min_pool = min_pool

        self.sample_key = ["obs", "next_obs", "acts", "rewards", "terminals"]

    def update_per_timestep(self):
        if self.replay_buffer.num_steps_can_sample() > max(self.min_pool, self.batch_size):
            for _ in range(self.opt_times):
                batch = self.replay_buffer.random_batch(self.batch_size, self.sample_key)
                infos = self.update(batch)
                self.logger.add_update_info(infos)

    def update_per_epoch(self):
        for _ in range(self.opt_times):
            batch = self.replay_buffer.random_batch(self.batch_size, self.sample_key)
            infos = self.update(batch)
            self.post_gradient_step()
            self.logger.add_update_info(infos)

    def post_gradient_step(self):
        for net in self.networks:
            net.apply(rezero_weights)

    def pretrain(self):
        total_frames = 0
        self.pretrain_epochs * self.collector.worker_nums * self.epoch_frames

        for pretrain_epoch in range(self.pretrain_epochs):

            start = time.time()

            self.start_epoch()

            training_epoch_info = self.collector.train_one_epoch()
            for reward in training_epoch_info["train_rewards"]:
                self.training_episode_rewards.append(reward)

            finish_epoch_info = self.finish_epoch()

            total_frames += self.collector.active_worker_nums * self.epoch_frames

            infos = {}

            infos["Train_Epoch_Reward"] = training_epoch_info["train_epoch_reward"]
            infos["Running_Training_Average_Rewards"] = np.mean(self.training_episode_rewards)
            infos.update(finish_epoch_info)

            self.logger.add_epoch_info(pretrain_epoch, total_frames, time.time() - start, infos, csv_write=False)

        self.pretrain_frames = total_frames

        self.logger.log("Finished Pretrain")
