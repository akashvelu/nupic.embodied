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
import torch.nn as nn
import torch
from nupic.research.frameworks.dendrites import (
    AbsoluteMaxGatingDendriticLayer,
    DendriticAbsoluteMaxGate1d,
    DendriticGate1d,
    OneSegmentDendriticLayer,
)
from nupic.torch.modules import KWinners
from nupic.research.frameworks.pytorch.models.le_sparse_net import (
    add_sparse_linear_layer,
)
from nupic.torch.modules.sparse_weights import rezero_weights

class DendriticMLP(nn.Module):
    """
    A dendritic network which is similar to a MLP with a two hidden layers, except that
    activations are modified by dendrites. The context input to the network is used as
    input to the dendritic weights. Adapted from:
    nupic.research/blob/master/projects/dendrites/supermasks/random_supermasks.py
                    _____
                   |_____|    # Classifier layer, no dendrite input
                      ^
                      |
                  _________
    context -->  |_________|  # Second linear layer with dendrites
                      ^
                      |
                  _________
    context -->  |_________|  # First linear layer with dendrites
                      ^
                      |
                    input
    """

    def __init__(self,
                 input_size,
                 output_dim,
                 dim_context,
                 hidden_sizes=(32, 32),
                 num_segments=(5, 5),
                 weight_sparsity=0.5,
                 k_winners=True,
                 relu=False,
                 k_winners_percent_on=0.25,
                 output_nonlinearity=None,
                 dendritic_layer_class=AbsoluteMaxGatingDendriticLayer):
        super().__init__()
        if all([num_seg == 1 for num_seg in num_segments]):
            dendritic_layer_class = OneSegmentDendriticLayer
        assert dendritic_layer_class in {AbsoluteMaxGatingDendriticLayer,
                                         DendriticAbsoluteMaxGate1d,
                                         DendriticGate1d,
                                         OneSegmentDendriticLayer}

        # The nonlinearity can either be k-Winners or ReLU, but not both
        assert not (k_winners and relu)
        self.num_segments = num_segments
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_dims = output_dim
        self.dim_context = dim_context
        self.k_winners = k_winners
        self.relu = relu

        self._layers = nn.ModuleList()
        self._activations = nn.ModuleList()
        prev_dim = input_size
        for i in range(len(hidden_sizes)):
            curr_dend = dendritic_layer_class(
                module=nn.Linear(prev_dim, hidden_sizes[i]),
                num_segments=num_segments[i],
                dim_context=dim_context,
                module_sparsity=weight_sparsity,
                dendrite_sparsity=weight_sparsity
            )
            if k_winners:
                curr_activation = KWinners(n=hidden_sizes[i],
                                           percent_on=k_winners_percent_on,
                                           k_inference_factor=1.0,
                                           boost_strength=0.0,
                                           boost_strength_factor=0.0)
            else:
                curr_activation = nn.ReLU()

            self._layers.append(curr_dend)
            self._activations.append(curr_activation)
            prev_dim = hidden_sizes[i]

        self._output_layer = nn.Sequential()
        linear_layer = nn.Linear(prev_dim, output_dim)
        self._output_layer.add_module("linear", linear_layer)

        if output_nonlinearity:
            self._output_layer.add_module("non_linearity",
                                    output_nonlinearity)
        self.apply(rezero_weights)

    def forward(self, x, context):
        if len(x.shape) > 2:
            original_shape = x.shape
            x = torch.flatten(x, start_dim=0, end_dim=1)
            context = torch.flatten(context, start_dim=0, end_dim=1)
            for layer, activation in zip(self._layers, self._activations):
                x = activation(layer(x, context))
            x = self._output_layer(x)
            x = x.view(original_shape[0], original_shape[1], x.shape[-1])
            return x
        for layer, activation in zip(self._layers, self._activations):
            x = activation(layer(x, context))
            return self._output_layer(x)


class FlattenDendriticMLP(DendriticMLP):
    def forward(self, x, context):
        x = torch.cat(x, dim=-1)
        return super().forward(x, context)


class SparseMLP(nn.Module):
    def __init__(self, input_dim,
                 output_dim,
                 output_nonlinearity=None,
                 hidden_sizes=(32, 32),
                 k_winners_percent_on=(0.1, 0.1),
                 weight_sparsity=(0.4, 0.4),
                 boost_strength=1.67,
                 boost_strength_factor=0.9,
                 duty_cycle_period=1000,
                 k_inference_factor=1.5,
                 use_batch_norm=True,
                 dropout=0.0,
                 consolidated_sparse_weights=False,
                 ):
        super(SparseMLP, self).__init__()
        assert len(hidden_sizes) == len(weight_sparsity)
        assert len(k_winners_percent_on) == len(weight_sparsity)

        self._hidden_base = nn.Sequential()
        self._hidden_base.add_module("flatten", nn.Flatten())
        # Add Sparse Linear layers
        for i in range(len(hidden_sizes)):
            add_sparse_linear_layer(
                network=self._hidden_base,
                suffix=i + 1,
                input_size=input_dim,
                linear_n=hidden_sizes[i],
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                weight_sparsity=weight_sparsity[i],
                percent_on=k_winners_percent_on[i],
                k_inference_factor=k_inference_factor,
                boost_strength=boost_strength,
                boost_strength_factor=boost_strength_factor,
                duty_cycle_period=duty_cycle_period,
                consolidated_sparse_weights=consolidated_sparse_weights,
            )
            input_dim = hidden_sizes[i]

        self._output_layer = nn.Sequential()
        linear_layer = nn.Linear(input_dim, output_dim)
        self._output_layer.add_module("linear", linear_layer)

        if output_nonlinearity:
            self._output_layer.add_module("non_linearity",
                                    output_nonlinearity)
        self.apply(rezero_weights)

    def forward(self, input_val):
        """Forward method.

        Args:
            input_val (torch.Tensor): Input values with (N, *, input_dim)
                shape.

        Returns:
            List[torch.Tensor]: Output values

        """
        if len(input_val.shape) > 2:
            original_shape = input_val.shape
            input_val = torch.flatten(input_val, start_dim=0, end_dim=1)
            x = self._hidden_base(input_val)
            x = self._output_layer(x)
            x = x.view(original_shape[0], original_shape[1], x.shape[-1])
            return x
        else:
            x = self._hidden_base(input_val)
            return self._output_layer(x)


class FlattenSparseMLP(SparseMLP):
    def forward(self, x):
        x = torch.cat(x, dim=-1)
        return super().forward(x)