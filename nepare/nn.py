from typing import OrderedDict

import torch
import lightning

class FeedforwardNeuralNetwork(lightning.LightningModule):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, activation: torch.nn.Module = torch.nn.ReLU, lr: float = 1e-3, n_targets: int = 1):
        super().__init__()
        _modules = OrderedDict()
        for i in range(num_layers):
            _modules[f"hidden_{i}"] = torch.nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
            _modules[f"{activation.__name__.lower()}_{i}"] = activation()
        _modules["readout"] = torch.nn.Linear(hidden_size, n_targets)
        self.fnn = torch.nn.Sequential(_modules)
        self.lr = lr
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), self.lr)

    def forward(self, x: torch.Tensor):
        return self.fnn(x)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], name: str):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log(f"{name}/loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        return self._step(batch, "training")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        return self._step(batch, "validation")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        return self._step(batch, "testing")

    def predict_step(self, X: torch.Tensor):
        return self(X[0])

class NeuralPairwiseRegressor(FeedforwardNeuralNetwork):
    def __init__(self, input_size, hidden_size, num_layers, activation = torch.nn.ReLU, lr: float = 1e-4, n_targets: int = 1):
        super().__init__(2*input_size, hidden_size, num_layers, activation, lr, n_targets)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], name: str):
        x_1, x_2, y = batch
        x = torch.cat((x_1, x_2), dim=1)
        return super()._step((x, y), name)

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        x_1, x_2, _ = batch
        x = torch.cat((x_1, x_2), dim=1)
        return self(x)

class LearnedEmbeddingNeuralPairwiseRegressor(NeuralPairwiseRegressor):
    def __init__(self, embedding_module: torch.nn.Module, embedding_size, hidden_size, num_layers, activation=torch.nn.ReLU, lr: float = 1e-4, n_targets: int = 1):
        super().__init__(embedding_size, hidden_size, num_layers, activation, lr, n_targets)
        # must be a learnable module that takes two inputs of some arbitrary type and generates a vector representation
        self.embedding_module = embedding_module
        self.save_hyperparameters(ignore=['embedding_module'])

    def forward(self, batch: tuple[object, object]):
        embedding_1 = self.embedding_module(batch[0])
        embedding_2 = self.embedding_module(batch[1])
        embedding = torch.cat((embedding_1, embedding_2), dim=1)
        return self.fnn(embedding)

    def _step(self, batch: tuple[object, object, torch.Tensor], name: str):
        x_1, x_2, y = batch
        y_hat = self((x_1, x_2))
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log(f"{name}/loss", loss, prog_bar=True, batch_size=y.shape[0])  # first object in batch can be hard to infer batch size
        return loss

    def predict_step(self, batch: tuple[object, object]):
        return self(batch)
