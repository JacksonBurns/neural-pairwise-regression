from typing import Iterable

import torch
import numpy as np
from rdkit.Chem import MolFromSmiles
from chemprop.data.molgraph import MolGraph
from chemprop.data.collate import BatchMolGraph
from chemprop.featurizers import MolGraphCache, SimpleMoleculeMolGraphFeaturizer


# uses ChemProp's BatchMolGraph class but not its TrainingBatch class
def collate_fn(batch: Iterable[tuple[MolGraph, MolGraph, float]]):
    mgs_1, mgs_2, ys = zip(*batch)  #  now need to convert y back into a tensor
    return BatchMolGraph(mgs_1), BatchMolGraph(mgs_2), torch.tensor(np.array(ys), dtype=torch.float32)


# stripped-down version of the chemprop.models.MPNN class
class ChemPropEmbedder(torch.nn.Module):
    def __init__(self, mp, agg):
        super().__init__()
        self.mp = mp
        self.agg = agg
        self.bn = torch.nn.BatchNorm1d(mp.output_dim)

    def forward(self, bmg):
        H = self.mp(bmg)
        Z = self.agg(H, bmg.batch)
        return Z


# convenience function to map list[SMILES] -> list[chemprop MolGraphs]
# we use MolGraphCache here (so that one could easily substitute MolGraphCacheOnTheFly)
# but we would instead just ust mgc = list(map(featurizer, mols))
def smiles2molgraphcache(smiles: list[str]):
    mols = list(map(MolFromSmiles, smiles))
    featurizer = SimpleMoleculeMolGraphFeaturizer()
    mgc = MolGraphCache(mols, [None] * len(mols), [None] * len(mols), featurizer)
    return mgc