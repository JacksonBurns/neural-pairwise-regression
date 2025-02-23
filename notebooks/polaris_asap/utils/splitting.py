from random import Random

from rdkit.Chem import DataStructs, rdFingerprintGenerator
from rdkit import Chem


# I miss the old rdkit syntax, so I use convenience functions like this ¯\_(ツ)_/¯
def _smi2fp(smi):
    fpg = rdFingerprintGenerator.GetMorganGenerator(radius=4)
    return fpg.GetFingerprint(Chem.MolFromSmiles(smi))


# this is basically like a leetcode question but for cheminformatics
# given the training and testing smiles, selects a validation set from the training smiles
# which has 50% randomly selected molecules and 50% molecules that are highly similar
# to the test set
def split(train_smis, test_smis):
    train_fps = list(map(_smi2fp, train_smis))
    test_fps = list(map(_smi2fp, test_smis))

    val_idxs = set()  # specifically use a set here to avoid duplicates
    for fp in test_fps:
        sims = [(i, DataStructs.FingerprintSimilarity(fp, _fp)) for i, _fp in enumerate(train_fps)]
        while 1:  # continue selecting the most similar molecule until one is found which is not already selected
            most_similar_idx = max(sims, key=lambda t: t[1])[0]
            if most_similar_idx in val_idxs:
                sims[most_similar_idx] = (most_similar_idx, 0.0)
            else:
                val_idxs.add(most_similar_idx)
                break
    val_idxs = list(val_idxs)

    train_idxs = [i for i in range(len(train_smis)) if i not in val_idxs]

    # mutate our sets
    rng = Random(42)
    val_idxs_swap = rng.sample(val_idxs, int(len(val_idxs) * 0.5))
    train_idxs_swap = rng.sample(train_idxs, int(len(val_idxs) * 0.5))
    train_idxs = [i for i in train_idxs if i not in train_idxs_swap] + val_idxs_swap
    val_idxs = [i for i in val_idxs if i not in val_idxs_swap] + train_idxs_swap
    return train_idxs, val_idxs