from typing import Optional
from torch_geometric.datasets import WikiCS
from torch_geometric.data import Data
from torch_geometric.data.lightning import LightningNodeData
import pytorch_lightning as pl
import torch


def get_statistics(data: Data) -> dict:
    is_directed = data.is_directed() if hasattr(data, 'is_directed') else False

    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(
        1) // 2 if not is_directed else data.edge_index.size(1)

    attr_dim = data.x.size(1) if hasattr(
        data, 'x') and data.x is not None else 0
    num_classes = len(torch.unique(data.y)) if hasattr(
        data, 'y') and data.y is not None else 0

    max_edges = num_nodes * \
        (num_nodes - 1) if is_directed else num_nodes * (num_nodes - 1) // 2
    density = (num_edges / max_edges) * 100 if max_edges > 0 else 0.0

    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'attr_dim': attr_dim,
        'num_classes': num_classes,
        'is_directed': is_directed,
        'graph_density': density,
    }


def load_dataset(name: str, statistics: bool = False) -> Optional[dict]:
    if name == "WikiCS":
        dataset = WikiCS(root='./data/WikiCS')
        data = dataset[0]
    else:
        raise ValueError(name)

    if statistics:
        return get_statistics(data)


def get_dataset_lightning_data(name: str) -> pl.LightningDataModule:
    if name == "WikiCS":
        return LightningNodeData(
            data=WikiCS(root="./data/WikiCS"),
            loader='full'
        )
    else:
        return ValueError(name)
