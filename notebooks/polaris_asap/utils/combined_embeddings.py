from typing import Iterable

import torch
import numpy as np
from rdkit.Chem import MolFromSmiles
from chemprop.data.molgraph import MolGraph
from chemprop.data.collate import BatchMolGraph
from chemprop.featurizers import MolGraphCache, SimpleMoleculeMolGraphFeaturizer

from utils.chemprop_wrappers import ChemPropEmbedder

def collate_fn_combined(batch: Iterable[tuple[tuple[MolGraph, torch.Tensor], tuple[MolGraph, torch.Tensor], float]]):
    mgs_1, feats_1, mgs_2, feats_2, ys = [], [], [], [], []
    for item in batch:
        mgs_1.append(item[0][0])
        feats_1.append(item[0][1])
        mgs_2.append(item[1][0])
        feats_2.append(item[1][1])
        ys.append(item[2])
    return ((BatchMolGraph(mgs_1), torch.stack(feats_1, dim=0)), (BatchMolGraph(mgs_2), torch.stack(feats_2, dim=0)), torch.tensor(np.array(ys), dtype=torch.float32))

class MordredChemPropEmbedder(ChemPropEmbedder):
    def forward(self, batch):
        bmg, feats = batch
        Z = super().forward(bmg)
        return torch.cat((Z, feats), dim=1)
