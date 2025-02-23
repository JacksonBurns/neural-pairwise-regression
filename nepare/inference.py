# functions for running inference with neural pairwise regression models
from typing import Sequence, Literal
from statistics import mean, stdev

import torch
import lightning
import numpy as np

from nepare.nn import NeuralPairwiseRegressor
from nepare.data import PairwiseInferenceDataset


def predict(npr: NeuralPairwiseRegressor, pid: torch.utils.data.DataLoader, *, how: Literal['all', 'half'] = 'all', quiet: bool=False) -> torch.Tensor:
    """Run inference with a Neural Pairwise Regressor, recasting results back to absolute.

    Args:
        npr (NeuralPairwiseRegressor): Trained model
        pid (torch.utils.data.DataLoader): Dataloader of a PairwiseInferenceDataset
    """
    trainer = lightning.Trainer(logger=False, enable_progress_bar=not quiet)  # TODO: ensure that we only ever run inference on 1 GPU (or none)
    pred = torch.vstack(trainer.predict(npr, pid))
    match how:
        case 'all' | 'half':
            return _all_anchor(pred, pid)
        case _:
            raise TypeError(f"Unsupported inference method {how=}")

def _all_anchor(pred, pid):  # TODO: clean up this mess of a function - so many casts...
    # do the collation in Python-land for simplicity, sacrificing speed for now
    absolute_predictions = {idx: [] for idx in range(len(pid.dataset.Xs[1]))}
    for pair, prediction in zip(pid.dataset.pairs, pred):  # TODO: add a progress bar that says "De-anchoring" or something
        # for a pair of inputs i,j the network predicts delta_i,j in that order
        # map back to the actual values here
        if pair.src_2 == 1:  # inference point is in position two
            # y_1 - y_2 = f(x_1,x_2) -> y_1 - f(x_1,x_2) = y_2
            _pred = pid.dataset.y_anchors[pair.idx_1] - prediction
            absolute_predictions[pair.idx_2].append(_pred)
        else:
            # y_1 - y_2 = f(x_1,x_2) -> y_1 = f(x_1,x_2) + y_2
            _pred = prediction + pid.dataset.y_anchors[pair.idx_2]
            absolute_predictions[pair.idx_1].append(_pred)
    y_pred = []
    y_stdev = []
    for idx, preds in absolute_predictions.items():
        preds = torch.tensor(np.array(preds))
        y_pred.append(torch.mean(preds, dim=0))
        y_stdev.append(torch.std(preds, dim=0))
    return torch.tensor(np.array(y_pred)), torch.tensor(np.array(y_stdev))