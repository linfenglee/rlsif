from typing import Optional, Tuple

import torch


LayerSizes = Tuple[int, ...]


class ValueNet(torch.nn.Module):
    """Simple fully connected value network used by the RL engine."""

    def __init__(
        self,
        fc_layers: LayerSizes = (2, 21, 21, 1),
        activation: torch.nn.Module = torch.nn.LeakyReLU(),
        output_fn: torch.nn.Module = torch.nn.Hardtanh(),
    ) -> None:
        super().__init__()
        self.n_neuron = fc_layers
        self.activation = activation
        self.output_fn = output_fn

        self.layers = self.value_net()

    def value_net(self) -> torch.nn.ModuleList:
        layers = []
        n_layers = len(self.n_neuron) - 1
        for i in range(n_layers):
            if i == n_layers - 1:
                layers.append(
                    self.one_layer(
                        self.n_neuron[i],
                        self.n_neuron[i + 1],
                        self.output_fn,
                        False
                    )
                )
            else:
                layers.append(
                    self.one_layer(
                        self.n_neuron[i],
                        self.n_neuron[i + 1],
                        self.activation,
                        False
                    )
                )
        return torch.nn.ModuleList(layers)

    @staticmethod
    def one_layer(
        input_dim: int,
        output_dim: int,
        activation_fn: Optional[torch.nn.Module] = torch.nn.ReLU(),
        batch_norm: bool = True,
    ) -> torch.nn.Sequential:
        if activation_fn is not None:
            if batch_norm:
                return torch.nn.Sequential(
                    torch.nn.Linear(input_dim, output_dim),
                    torch.nn.BatchNorm1d(output_dim),
                    activation_fn
                )
            else:
                return torch.nn.Sequential(
                    torch.nn.Linear(input_dim, output_dim),
                    activation_fn
                )
        else:
            if batch_norm:
                return torch.nn.Sequential(
                    torch.nn.Linear(input_dim, output_dim),
                    torch.nn.BatchNorm1d(output_dim)
                )
            else:
                return torch.nn.Sequential(
                    torch.nn.Linear(input_dim, output_dim)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out