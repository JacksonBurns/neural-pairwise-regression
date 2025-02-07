from typing import OrderedDict, Sequence

import torch
import lightning

class FeedforwardNeuralNetwork(lightning.LightningModule):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, activation: torch.nn.Module = torch.nn.ReLU):
        super().__init__()
        _modules = OrderedDict()
        for i in range(num_layers):
            _modules[f"hidden_{i}"] = torch.nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
            _modules[f"{activation.__name__.lower()}_{i}"] = activation()
        _modules["readout"] = torch.nn.Linear(hidden_size, 1)
        self.fnn = torch.nn.Sequential(_modules)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

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

    def testing_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        return self._step(batch, "testing")

class NeuralPairwiseRegressor(FeedforwardNeuralNetwork):
    def __init__(self, input_size, hidden_size, num_layers, activation = torch.nn.ReLU):
        super().__init__(2*input_size, hidden_size, num_layers, activation)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], name: str):
        x_1, x_2, y = batch
        x = torch.cat(x_1, x_2, dim=1)
        return super()._step((x, y), name)
