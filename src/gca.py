import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.nn import GATConv


def drop_edges_degree_centrality(edge_index: torch.Tensor, deg: torch.Tensor, pe_remove: float, pe_cutoff: float) -> torch.Tensor:
    src, dst = edge_index
    edge_centrality = (deg[src] + deg[dst]) / 2
    probs = torch.log(edge_centrality + 1e-8)
    max_prob = probs.max()
    mean_prob = probs.mean()
    probs = torch.min(
        (max_prob - probs) / (max_prob - mean_prob + 1e-8) * pe_remove,
        torch.full_like(probs, pe_cutoff)
    )
    mask = torch.rand_like(probs) > probs
    return edge_index[:, mask]


def mask_features_degree_centrality(x: torch.Tensor, deg: torch.Tensor, pn_remove: float, pn_cutoff: float) -> torch.Tensor:
    probs = (torch.abs(x) * deg.unsqueeze(-1)).sum(dim=0)
    probs = torch.log(probs + 1e-8)
    max_prob = probs.max()
    mean_prob = probs.mean()
    probs = torch.min(
        (max_prob - probs) / (max_prob - mean_prob + 1e-8) * pn_remove,
        torch.full_like(probs, pn_cutoff)
    )
    mask = torch.rand_like(probs) > probs
    return x * mask


class GCAAugumenter:
    def __init__(self, pe_remove: tuple[float, float], pe_cutoff: tuple[float, float], pn_remove: tuple[float, float], pn_cutoff: tuple[float, float]):
        self.pe_remove = pe_remove
        self.pe_cutoff = pe_cutoff
        self.pn_remove = pn_remove
        self.pn_cutoff = pn_cutoff

    def __call__(self, data: Data):
        deg = degree(data.edge_index[1], dtype=torch.float)

        edge_index_a = drop_edges_degree_centrality(
            data.edge_index, deg, self.pe_remove[0], self.pe_cutoff[0])
        edge_index_b = drop_edges_degree_centrality(
            data.edge_index, deg, self.pe_remove[1], self.pe_cutoff[1])

        x_a = mask_features_degree_centrality(
            data.x, deg, self.pn_remove[0], self.pn_cutoff[0])
        x_b = mask_features_degree_centrality(
            data.x, deg, self.pn_remove[1], self.pn_cutoff[1])

        return (x_a, edge_index_a), (x_b, edge_index_b)


class GCABase(pl.LightningModule):
    def __init__(
            self, encoder: nn.Module,
            temp: float = 0.6, lr: float = 0.001,
            augmenter: GCAAugumenter = GCAAugumenter(
                pe_remove=(0.2, 0.4),
                pe_cutoff=(0.7, 0.7),
                pn_remove=(0.1, 0.1),
                pn_cutoff=(0.7, 0.7)
            )):
        super().__init__()
        self.lr = lr
        self.temp = temp
        self.encoder = encoder
        self.augmenter = augmenter

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def forward_project(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.forward(x, edge_index)

    def info_nce_loss(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.size(0)
        pos_indices_i = torch.arange(batch_size, device=z_i.device)
        pos_indices_j = torch.arange(
            batch_size, 2*batch_size, device=z_i.device)

        sim_full = torch.cat([z_i, z_j]) @  torch.cat([z_i, z_j]).T
        sim_full = torch.exp(sim_full / self.temp)
        positive_sim = sim_full[pos_indices_i, pos_indices_j]
        zero_diag = torch.ones_like(
            sim_full) - torch.eye(sim_full.size(0), device=sim_full.device)
        sim_full = sim_full * zero_diag
        negative_sim = sim_full.sum(dim=1)
        loss = -torch.log(positive_sim / negative_sim)
        return loss.mean()

    def training_step(self, batch: Data, batch_idx: int) -> dict:
        (x_a, edge_index_a), (x_b, edge_index_b) = self.augmenter(batch)
        z_a = self.forward(x_a, edge_index_a)
        z_b = self.forward(x_b, edge_index_b)
        loss = self.info_nce_loss(z_a, z_b)

        self.log('train/loss', loss.item(), on_epoch=True, on_step=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.parameters(),
            lr=self.lr,
            weight_decay=5e-4,
        )


class GATEncoderBN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim, momentum=0.01)
        self.act1 = nn.ReLU()
        self.conv2 = GATConv(hidden_dim, out_dim, heads=1)
        self.act2 = nn.ReLU()

    def forward(self, x, edge_index):
        z = self.act1(self.bn1(self.conv1(x, edge_index)))
        z = self.act2(self.conv2(z, edge_index))
        return z
