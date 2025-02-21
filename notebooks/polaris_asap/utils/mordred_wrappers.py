from mordred import Calculator, descriptors
import torch
import numpy as np
from rdkit.Chem import MolFromSmiles
from fastprop.data import standard_scale


def smi2features(smis, feature_means=None, feature_vars=None):
    calc = Calculator(descriptors, ignore_3D=True)
    train_features = calc.pandas(map(MolFromSmiles, smis), nmols=len(smis), quiet=True).fill_missing()
    X = torch.tensor(train_features.to_numpy(dtype=np.float32), dtype=torch.float32)
    if feature_means is None or feature_vars is None:
        X, feature_means, feature_vars = standard_scale(X)
    else:
        X = standard_scale(X, feature_means, feature_vars)
    X.clamp_(-3, 3)
    return X, feature_means, feature_vars