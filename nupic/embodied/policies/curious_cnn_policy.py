# ------------------------------------------------------------------------------
#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
# ------------------------------------------------------------------------------

# from https://github.com/qqadssp/Pytorch-Large-Scale-Curiosity/

import numpy as np
import torch
from nupic.embodied.utils.distributions import make_pdtype

from nupic.embodied.utils.model_parts import small_convnet, unflatten_first_dim


class CnnPolicy(object):
    """Cnn Policy of the PPO agent.

    Parameters
    ----------
    ob_space : Space
        Observation space properties (from env.observation_space).
    ac_space : Space
        Action space properties (from env.action_space).
    ob_mean : array
        Mean value of observations collected by a random agent. (Of same size as the
        observations)
    ob_std : float
        Standard deviation of observations collected by a random agent.
    feature_dim : int
        Number of neurons in the hidden layer of the feature network.
    hidden_dim : int
        Number of neurons in the hidden layer of the policy network.
    layernormalize : bool
        Whether to normalize last layer.
    nonlinear : torch.nn
        nonlinear activation function to use.
    scope : str
        Scope name.
    device: torch.device
        Which device to optimize the model on.

    Attributes
    ----------
    ac_pdtype : type
        Description of attribute `ac_pdtype`.
    pd : type
        Description of attribute `pd`.
    vpred : type
        Description of attribute `vpred`.
    features_model : torch.Sequential
        Small conv net to extract features from observations.
    pd_hidden : type
        Hidden layer of the policy network of size hidden_dim (2 layer, relu).
    pd_head : type
        Linear FC layer following pd_hidden with policy output.
    vf_head : type
        Linear FC layer following pd_hidden with value output (1).
    param_list : type
        List of parameters to be optimized.
    flat_features : type
        flattened feature vector.
    ac : array
        Current action.
    ob : array
        Current observation.

    """

    def __init__(
        self,
        ob_space,
        ac_space,
        ob_mean,
        ob_std,
        feature_dim,
        hidden_dim,
        layernormalize,
        nonlinear,
        device,
        scope="policy",
    ):
        if layernormalize:
            print(
                """Warning: policy is operating on top of layer-normed features.
                It might slow down the training."""
            )
        self.layernormalize = layernormalize
        self.nonlinear = nonlinear
        self.ob_mean = ob_mean
        self.ob_std = ob_std
        self.ob_space = ob_space
        self.ac_space = ac_space
        # Get the type of probabiliyt distribution to use dependng on the environments
        # action space type.
        self.ac_pdtype = make_pdtype(ac_space)

        self.pd = self.vpred = None
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.scope = scope
        self.device = device
        pdparamsize = self.ac_pdtype.param_shape()[0]

        # Initialize the feature model as a small conv net (3 conv layer + 1 linear fc)
        self.features_model = small_convnet(
            self.ob_space,
            nonlinear=self.nonlinear,
            feature_dim=self.feature_dim,
            last_nonlinear=None,
            layernormalize=self.layernormalize,
            batchnorm=False,
            device=self.device,
        ).to(self.device)

        # Policy network following the feature extraction network (2 fc layers, relu)
        self.pd_hidden = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
        ).to(self.device)
        # policy and value function head of the policy network.
        self.pd_head = torch.nn.Linear(self.hidden_dim, pdparamsize).to(self.device)
        self.vf_head = torch.nn.Linear(self.hidden_dim, 1).to(self.device)

        # Define parameters to be optimized
        self.param_list = [
            *self.features_model.parameters(),
            *self.pd_hidden.parameters(),  # pd: policy (pi)
            *self.pd_head.parameters(),  #
            *self.vf_head.parameters(),  # value function head
        ]

        self.flat_features = None
        self.pd = None
        self.vpred = None
        self.ac = None
        self.ob = None

    def update_features(self, ob, ac):
        """Set self.flat_features, pd and vpred to match with current observation.
        Also sets self.ac to the last actions at end of rollout..

        Parameters
        ----------
        ob : array
            Current observations.
            ob.shape = [nenvs, H, W, C] during rollout when calling get_ac_value_nlp()

            ob.shape = [1, n_steps_per_seg, H, W, C] when called from calculate_loss in
            dynamics module (dynamics.calculate_loss() at end of rollout).

        ac : array or None
            Batch of actions (at end of rollout, otherwise None).

        """

        sh = ob.shape
        # get the corresponding features of the observations (shape = [N, feature_dim])
        flat_features = self.get_features(ob)
        self.flat_features = flat_features
        # Process the features with the policy network
        hidden = self.pd_hidden(flat_features)
        self.hidden_pi = hidden
        # get policy parameters from the hidden activations
        pdparam = self.pd_head(hidden)
        # Get value estimate from the hidden activations
        vpred = self.vf_head(hidden)
        # Set global class variables
        self.vpred = unflatten_first_dim(vpred, sh)  # [nenvs, n_steps_per_seg, v]
        self.pd = self.ac_pdtype.pdfromflat(
            pdparam
        )  # pd: probability density   # use to calculate entropy and get KL-div
        self.ac = ac
        self.ob = ob

    def get_features(self, ob):
        """Get the features corresponding to an observation.

        Parameters
        ----------
        ob : array
            Observation input to the feature model.

        Returns
        -------
        array
            Output of the feature model.

        """
        # Get a shape of [N, H, W, C]
        ob = ob.reshape((-1,) + ob.shape[-len(self.ob_space.shape) :])

        if len(ob.shape) == 5:
            print("Timesteps are not implemented yet.")
        # Normalize observations
        ob = (ob - self.ob_mean) / self.ob_std
        # reshape observations: [N, H, W, C] --> [N, C, H, W]
        ob = np.transpose(ob, [i for i in range(len(ob.shape) - 3)] + [-1, -3, -2])
        # Run observations through feature model
        ob = self.features_model(torch.tensor(ob).to(self.device))

        return ob

    def get_ac_value_nlp(self, ob):
        """Given an observation get the value, action and negative log probability.

        Parameters
        ----------
        ob : array
            Observation.

        Returns
        -------
        (np.array, np.array, np.array)
            List of (sampled action, value estimate, negative log prob of the sampled
            action)

        """
        self.update_features(ob, None)
        a_samp = self.pd.sample()
        nlp_samp = self.pd.neglogp(a_samp)
        return (
            a_samp.squeeze().cpu().data.numpy(),
            self.vpred.squeeze().cpu().data.numpy(),
            nlp_samp.squeeze().cpu().data.numpy(),
        )
