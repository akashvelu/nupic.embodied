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

        if all([num_seg == 1 for num_seg in num_segments]):
            dendritic_layer_class = OneSegmentDendriticLayer
        assert dendritic_layer_class in {AbsoluteMaxGatingDendriticLayer,
                                         DendriticAbsoluteMaxGate1d,
                                         DendriticGate1d,
                                         OneSegmentDendriticLayer}

        # The nonlinearity can either be k-Winners or ReLU, but not both
        assert not (k_winners and relu)

        super().__init__()

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

        # Final multiheaded layer
        self._output_layer = nn.Sequential()
        linear_layer = nn.Linear(prev_dim, output_dim)
        self._output_layer.add_module("linear", linear_layer)

        if output_nonlinearity:
            self._output_layer.add_module("non_linearity",
                                    output_nonlinearity)

    def forward(self, x, context):
        for layer, activation in zip(self._layers, self._activations):
            x = activation(layer(x, context))

        return self._output_layer(x)

class FlattenDendriticMLP(DendriticMLP):
    def forward(self, x, context):
        x = torch.cat(x, dim=-1)
        return super().forward(x, context)
    