# this script was used to identify good hyperparameters for the combined embedding,
# since I had no intuition on what they should be
#
# all of the logic for model setup, embeddings, etc. is the same as in the `main` notebook
# the only difference is the ray and optuna setup at the bottom of the file
from pathlib import Path

import polaris as po
import pandas as pd
import torch
import lightning
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from chemprop.nn.agg import MeanAggregation, NormAggregation
from chemprop.nn.message_passing import BondMessagePassing
from fastprop.data import standard_scale
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch

from nepare.data import PairwiseAugmentedDataset, PairwiseAnchoredDataset, PairwiseInferenceDataset
from nepare.nn import LearnedEmbeddingNeuralPairwiseRegressor
from nepare.inference import predict

from utils.splitting import split
from utils.chemprop_wrappers import smiles2molgraphcache
from utils.mordred_wrappers import smi2features
from utils.metrics import evaluate_predictions
from utils.combined_embeddings import MordredChemPropEmbedder, collate_fn_combined


def fit(train_df, test_df, tasks, *, chemprop_dim, chemprop_layers, chemprop_aggregation, fnn_dim, fnn_layers, augmentation_strategy):
    val_score = []
    output_dir = Path("lightning_logs")
    for task_n, task_name in enumerate(tasks):
        dataset_kwargs = dict(batch_size=64)
        task_df = train_df[["CXSMILES", task_name]].copy()
        task_df.dropna(inplace=True)
        task_df.reset_index(inplace=True)
        train_idxs, val_idxs = split(task_df["CXSMILES"], test_df["CXSMILES"])
        train_targets = torch.tensor(task_df[task_name].iloc[train_idxs].to_numpy(), dtype=torch.float32).reshape(-1, 1)  # 2d!
        train_targets, target_means, target_vars = standard_scale(train_targets)
        val_targets = torch.tensor(task_df[task_name].iloc[val_idxs].to_numpy(), dtype=torch.float32).reshape(-1, 1)  # 2d!
        val_targets = standard_scale(val_targets, target_means, target_vars)
        # featurize the data
        train_features, feature_means, feature_vars = smi2features(task_df["CXSMILES"][train_idxs])
        val_features, _, _ = smi2features(task_df["CXSMILES"][val_idxs], feature_means, feature_vars)
        train_mgc = smiles2molgraphcache(task_df["CXSMILES"][train_idxs])
        val_mgc = smiles2molgraphcache(task_df["CXSMILES"][val_idxs])
        train_tuples = [(train_mgc[i], train_features[i, :]) for i in range(len(train_targets))]
        val_tuples = [(val_mgc[i], val_features[i, :]) for i in range(len(val_targets))]
        # setup the datasets
        train_dataset = PairwiseAugmentedDataset(train_tuples, train_targets, how='full' if augmentation_strategy == 'full' else 'ut')
        val_dataset = PairwiseAnchoredDataset(train_tuples, train_targets, val_tuples, val_targets, how=augmentation_strategy)
        val_absolute_dataset = PairwiseInferenceDataset(train_tuples, train_targets, val_tuples, how=augmentation_strategy)
        dataset_kwargs["collate_fn"] = collate_fn_combined
        # build the model
        mp = BondMessagePassing(d_h=chemprop_dim, depth=chemprop_layers, activation="leakyrelu")
        agg = MeanAggregation() if chemprop_aggregation == "mean" else NormAggregation()
        embedder = MordredChemPropEmbedder(mp, agg)
        npr = LearnedEmbeddingNeuralPairwiseRegressor(embedder, mp.output_dim + train_features.shape[1], fnn_dim, fnn_layers, lr=5e-5, activation=torch.nn.LeakyReLU)

        # classic lightning training, inference
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **dataset_kwargs)
        val_loader = torch.utils.data.DataLoader(val_dataset, **dataset_kwargs)
        val_absolute_loader = torch.utils.data.DataLoader(val_absolute_dataset, **dataset_kwargs)
        early_stopping = EarlyStopping(monitor="validation/loss", patience=3)
        name = "_".join(["combined", task_name, "hopt"])
        model_checkpoint = ModelCheckpoint(dirpath=output_dir / name, monitor="validation/loss")
        logger = TensorBoardLogger(save_dir=output_dir, name=name)
        trainer = lightning.Trainer(max_epochs=50, log_every_n_steps=1, callbacks=[early_stopping, model_checkpoint], logger=logger, enable_progress_bar=False, enable_model_summary=False)
        trainer.fit(npr, train_loader, val_loader)
        npr = LearnedEmbeddingNeuralPairwiseRegressor.load_from_checkpoint(model_checkpoint.best_model_path)
        y_pred, y_stdev = predict(npr, val_absolute_loader, quiet=True)
        # leave in the scaled space so that the average is weighted equally among the targets
        results_dict = evaluate_predictions(val_targets.flatten().numpy(), y_pred.flatten().numpy())
        val_score.append((len(val_targets), results_dict["MAE"]))
    return {"validation_wavg_mae": sum(n*s for n, s in val_score) / sum(n for n, _ in val_score)}

def _obj(train_df_ref, test_df_ref, tasks_ref, trial):
    train_df = ray.get(train_df_ref)
    test_df = ray.get(test_df_ref)
    tasks = ray.get(tasks_ref)
    return fit(train_df, test_df, tasks, **trial)

if __name__ == "__main__":
    # initial run
    # Current best trial: d2275c7f with validation_wavg_mae=55.0637944161892 and params={'chemprop_dim': 400, 'chemprop_layers': 2, 'chemprop_aggregation': 'mean', 'fnn_dim': 1000, 'fnn_layers': 4, 'augmentation_strategy': 'half'}
    for comp_name in ("asap-discovery/antiviral-potency-2025", "asap-discovery/antiviral-admet-2025"):
        competition = po.load_competition(comp_name)
        train, test = competition.get_train_test_split()
        train_df: pd.DataFrame = train.as_dataframe()
        test_df: pd.DataFrame = test.as_dataframe()
        tasks = list(competition.target_cols)
        # driver code of optimization
        search_space = dict(
            chemprop_dim=tune.choice(range(100, 1001, 100)),
            chemprop_layers=tune.choice(range(1, 5, 1)),
            chemprop_aggregation=tune.choice(("mean", "norm")),
            fnn_dim=tune.choice(range(100, 2101, 200)),
            fnn_layers=tune.choice(range(1, 5, 1)),
            augmentation_strategy=tune.choice(("full", "half"))
        )
        algo = OptunaSearch()
        train_df_ref = ray.put(train_df)
        test_df_ref = ray.put(test_df)
        tasks_ref = ray.put(tasks)
        metric = "validation_wavg_mae"
        resources = {"cpu": 8, "gpu": 1}
        if torch.cuda.is_available():
            resources["gpu"] = 1
        tuner = tune.Tuner(
            tune.with_resources(
                lambda trial: _obj(
                    train_df_ref,
                    test_df_ref,
                    tasks_ref,
                    trial,
                ),
                resources=resources,
            ),
            tune_config=tune.TuneConfig(
                metric=metric,
                mode="min",
                search_alg=algo,
                max_concurrent_trials=1,
                num_samples=16,
            ),
            param_space=search_space,
        )
        results = tuner.fit()
        best = results.get_best_result().config
        print(f"Best hyperparameters identified: {', '.join([key + ': ' + str(val) for key, val in best.items()])}")
