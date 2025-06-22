from pathlib import Path
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch.nn as nn
import math


class GaussianMixtureModel(pl.LightningModule):
    def __init__(
        self,
        n_components: int,
        dim: int
    ):
        """Inicjalizacja modelu mikstur gausowskich.

        :param n_components: liczba komponentów modelu
        :param n_epochs: liczba epok trenowania modelu
        """
        super().__init__()
        self.n_components = n_components
        self.dim = dim

        self.means = nn.Parameter(torch.rand(
            (self.n_components, self.dim)), requires_grad=False)

        self.covariances = nn.Parameter(torch.eye(self.dim).repeat(
            self.n_components, 1, 1
        ), requires_grad=False)

        self.mixing_coefs = nn.Parameter(torch.full(
            (self.n_components,),
            fill_value=1 / self.n_components
        ), requires_grad=False)

    @torch.no_grad()
    def gamma_top(self, x: torch.Tensor) -> torch.Tensor:
        n_points = x.shape[0]
        # https://stats.stackexchange.com/questions/390532/adding-a-small-constant-to-the-diagonals-of-a-matrix-to-stabilize
        cov_reg = self.covariances + 1e-6 * \
            torch.eye(self.dim, device=x.device).unsqueeze(0)
        xs = torch.reshape(x, (1, n_points, self.dim, 1))
        mus = torch.reshape(self.means, (self.n_components, 1, self.dim, 1))
        Sigmas = torch.reshape(
            cov_reg, (self.n_components, 1, self.dim, self.dim))

        log_py = torch.log(self.mixing_coefs + 1e-10).unsqueeze(1)
        diff = xs - mus
        inv_Sigmas = torch.linalg.inv(Sigmas)

        log_top = torch.matmul(
            torch.matmul(diff.transpose(3, 2), inv_Sigmas),
            diff
        ).squeeze(-1).squeeze(-1)

        log_det = torch.logdet(Sigmas).clamp(min=math.log(1e-10))
        log_bottom = (-0.5 * (self.dim * math.log(2 * math.pi) + log_det))
        log_pxy = log_bottom - 0.5 * log_top
        log_pyx = log_py + log_pxy
        p_yx = torch.exp(log_pyx)
        return p_yx

    @torch.no_grad()
    def expectation_step(self, x: torch.Tensor) -> torch.Tensor:
        """Krok Expectation (obliczenie responsibilities)."""
        g = self.gamma_top(x)
        return g / torch.sum(g, 0)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expectation_step(x)

    @torch.no_grad()
    def maximization_step(
        self, x: torch.Tensor, responsibilities: torch.Tensor
    ):
        """Krok Maximization (obliczenie parametrów)."""
        N_k = torch.sum(responsibilities, 1)
        means = ((responsibilities.unsqueeze(1) @ x).squeeze(1).T / N_k).T
        n_points = x.shape[0]
        mus = torch.reshape(x, (1, n_points, self.dim, 1)) - \
            torch.reshape(self.means, (self.n_components, 1, self.dim, 1))
        covariances = responsibilities.unsqueeze(
            -1).unsqueeze(-1) * (mus @ mus.transpose(3, 2))
        covariances = torch.sum(covariances, 1) / N_k.reshape(-1, 1, 1)
        mixing_coefs = N_k / n_points
        self.means.data.copy_(means)
        self.covariances.data.copy_(covariances)
        self.mixing_coefs.data.copy_(mixing_coefs)

    @torch.no_grad()
    def loglikelihood(self, x: torch.Tensor) -> float:
        """Log-likelihood modelu."""
        g = self.gamma_top(x)
        return (torch.sum(
            torch.log(torch.sum(g, 0))
        ) / x.shape[0]).item()

    @torch.no_grad()
    def training_step(self, x: torch.Tensor):
        responsibilities = self.expectation_step(x)
        self.maximization_step(x, responsibilities)
        log_likelihood = self.loglikelihood(x)
        self.log('log-likelihood', log_likelihood,
                 on_epoch=True, on_step=False)

    def configure_optimizers(self):
        return None


def get_gmm_trainer(model_name: str, max_epochs: int, patience: int = 1, termination_threshold: float = 1e-3, accelerator: str = "gpu"):
    p = str(Path(__file__).resolve().parent.parent / "data" / "gmm-logs")
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        logger=pl.loggers.TensorBoardLogger(
            save_dir=p,
            name=model_name,
        ),
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        callbacks=[
            EarlyStopping(
                monitor="log-likelihood",
                min_delta=termination_threshold,
                patience=patience,
                verbose=True,
                mode="max"
            )]
    )


class WholeTensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx]


class GMMDataModule(pl.LightningDataModule):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor

    def train_dataloader(self):
        dataset = WholeTensorDataset(self.tensor)
        return torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
