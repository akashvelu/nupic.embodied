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
import torch
from nupic.research.frameworks.dendrites.modules import DendriticMLP
from nupic.research.frameworks.pytorch.models.common_models import SparseMLP


class FlattenSparseMLP(SparseMLP):
    def forward(self, x):
        x = torch.cat(x, dim=-1)
        return super().forward(x)


class FlattenDendriticMLP(DendriticMLP):
    def forward(self, x, context):
        x = torch.cat(x, dim=-1)
        return super().forward(x, context)
