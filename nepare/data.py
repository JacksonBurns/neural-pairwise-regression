from collections.abc import Sequence
from collections import namedtuple
from typing import Literal
from itertools import combinations, combinations_with_replacement, product, chain
from random import Random

import torch

class PairwiseAugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, X: Sequence, y: Sequence, *, how: Literal['full','ut','sut'] = 'full'):
        super().__init__()
        self.X = X
        self.y = y
        match how:
            case 'full':
                self.idxs = list(product(range(len(X)), repeat=2))
            case 'ut':
                self.idxs = list(combinations_with_replacement(range(len(X)), 2))
            case 'sut':
                self.idxs = list(combinations(range(len(X)), 2))
            case _:
                raise TypeError(f"Invalid configuration {how=}.")
    
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        i, j = self.idxs[index]
        return self.X[i], self.X[j], self.y[i] - self.y[j]
    
    def downsample_(self, n:int, random_seed:int=1701):
        rng = Random(random_seed)
        self.idxs = rng.sample(self.idxs, k=n)

class PairwiseAnchoredDataset(torch.utils.data.Dataset):
    def __init__(self, X_anchors: Sequence, y_anchors: Sequence, X: Sequence, y: Sequence, *, how: Literal['full','half'] = 'full'):
        super().__init__()
        self.Xs = (X_anchors, X)
        self.ys = (y_anchors, y)
        Pair = namedtuple('Pair', ['src_1', 'idx_1', 'src_2', 'idx_2'])
        pairs = []
        for i in range(len(X_anchors)):
            for j in range(len(X)):
                pairs.append(Pair(0, i, 1, j))
                if how == 'full':
                    pairs.append(Pair(1, j, 0, i))
        self.pairs = pairs
    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        p = self.pairs[index]
        return self.Xs[p.src_1][p.idx_1], self.Xs[p.src_2][p.idx_2], self.ys[p.src_1][p.idx_1] - self.ys[p.src_2][p.idx_2]

class PairwiseInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, X_anchors: Sequence, y_anchors: Sequence, X: Sequence, *, how: Literal['full','half'] = 'full'):
        super().__init__()
        self.Xs = (X_anchors, X)
        self.y_anchors = y_anchors
        Pair = namedtuple('Pair', ['src_1', 'idx_1', 'src_2', 'idx_2'])
        pairs = []
        for i in range(len(X_anchors)):
            for j in range(len(X)):
                pairs.append(Pair(0, i, 1, j))
                if how == 'full':
                    pairs.append(Pair(1, j, 0, i))
        self.pairs = pairs
    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        src_1, idx_1, src_2, idx_2 = self.pairs[index]
        return self.Xs[src_1][idx_1], self.Xs[src_2][idx_2], self.y_anchors[idx_1 if src_1 == 0 else idx_2]
