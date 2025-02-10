# functions for running inference with neural pairwise regression models
from typing import Sequence, Literal

import torch
import lightning

from nepare.nn import NeuralPairwiseRegressor
from nepare.data import PairwiseInferenceDataset


def predict(npr: NeuralPairwiseRegressor, pid: torch.utils.data.DataLoader, how: Literal['all'] = 'all'):
    """Run inference with a Neural Pairwise Regressor, recasting results back to absolute.

    Args:
        npr (NeuralPairwiseRegressor): Trained model
        pid (torch.utils.data.DataLoader): Dataloader of a PairwiseInferenceDataset
    """
    trainer = lightning.Trainer(logger=False)
    pred = trainer.predict(npr, pid)
    
