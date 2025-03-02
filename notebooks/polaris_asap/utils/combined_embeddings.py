from typing import Iterable, OrderedDict

import torch
import numpy as np
from chemprop.data.molgraph import MolGraph
from chemprop.data.collate import BatchMolGraph

from utils.chemprop_wrappers import ChemPropEmbedder


# read chemprop_wrappers.collate_fn before looking at this
# type hint should help understand what it does
# this somewhat awkward data structure is used to dramatically simplify the process of augmentation
def collate_fn_combined(batch: Iterable[tuple[tuple[MolGraph, torch.Tensor], tuple[MolGraph, torch.Tensor], float]]):
    mgs_1, feats_1, mgs_2, feats_2, ys = [], [], [], [], []
    for item in batch:
        mgs_1.append(item[0][0])
        feats_1.append(item[0][1])
        mgs_2.append(item[1][0])
        feats_2.append(item[1][1])
        ys.append(item[2])
    return ((BatchMolGraph(mgs_1), torch.stack(feats_1, dim=0)), (BatchMolGraph(mgs_2), torch.stack(feats_2, dim=0)), torch.tensor(np.array(ys), dtype=torch.float32))

# uses our ChemProp-based embedder, but concatenates our extra features on after the learnable embedding part
class MordredChemPropEmbedder(ChemPropEmbedder):
    def forward(self, batch):
        bmg, feats = batch
        Z = super().forward(bmg)
        return torch.cat((Z, feats), dim=1)

class ArbitrarilyMoreComplicatedEmbedder(torch.nn.Module):
    def __init__(self, mp, agg, mordred_size, mordred_layers):
        super().__init__()
        # chemprop
        self.mp = mp
        self.agg = agg

        # mordred
        _modules = OrderedDict()
        _modules["dropout"] = torch.nn.Dropout(p=mordred_size / 1_613)
        activation = torch.nn.ReLU
        for i in range(mordred_layers):
            _modules[f"hidden_{i}"] = torch.nn.Linear(1_613 if i == 0 else mordred_size, mordred_size)
            _modules[f"{activation.__name__.lower()}_{i}"] = activation()
        self.fnn = torch.nn.Sequential(_modules)

        # shared
        self.bn = torch.nn.BatchNorm1d(mp.output_dim + mordred_size)

    def forward(self, batch):
        bmg, feats = batch
        H = self.mp(bmg)
        Z = self.agg(H, bmg.batch)
        f = self.fnn(feats)
        return self.bn(torch.cat((Z, f), dim=1))