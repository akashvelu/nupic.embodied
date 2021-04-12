import torch
from nupic.embodied.soft_modularization.torchrl.policies.distribution import TanhNormal
from nupic.embodied.soft_modularization.networks.nets import DendriticMLP
from nupic.embodied.soft_modularization.torchrl.policies import EmbeddingGuassianContPolicyBase

import abc
from torch import nn

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class DendriticGuassianContPolicy(DendriticMLP, EmbeddingGuassianContPolicyBase):
    def forward(self, x, embedding_input):
        x = super().forward(x, embedding_input)

        mean, log_std = x.chunk(2, dim=-1)

        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        return mean, std, log_std

    def eval_act(self, x, embedding_input):
        with torch.no_grad():
            mean, std, log_std = self.forward(x, embedding_input)

        return torch.tanh(mean.squeeze(0)).detach().cpu().numpy()

    def explore(self, x, embedding_input, return_log_probs=False, return_pre_tanh=False):
        mean, std, log_std = self.forward(x, embedding_input)
        dic = {}

        dis = TanhNormal(mean, std)

        ent = dis.entropy().sum(-1, keepdim=True)

        dic.update({
            "mean": mean,
            "log_std": log_std,
            "ent": ent
        })

        if return_log_probs:
            action, z = dis.rsample(return_pretanh_value=True)
            log_prob = dis.log_prob(
                action,
                pre_tanh_value=z
            )
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            dic["pre_tanh"] = z.squeeze(0)
            dic["log_prob"] = log_prob
        else:
            if return_pre_tanh:
                action, z = dis.rsample(return_pretanh_value=True)
                dic["pre_tanh"] = z.squeeze(0)
            action = dis.rsample(return_pretanh_value=False)

        dic["action"] = action.squeeze(0)
        return dic
