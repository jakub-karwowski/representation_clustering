import os
import sys
from pathlib import Path
import pytorch_lightning as pl
import torch
import yaml
from torch_geometric.datasets import WikiCS


def main():
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.gca import GATEncoderBN
    params = yaml.safe_load(open("../params.yaml", 'r')
                            )['wikics']['embed_stage']

    encoder = GATEncoderBN(
        in_dim=params['input_dim'],
        hidden_dim=params['hidden_dim'],
        out_dim=params['output_dim']
    )

    model_name = f"gca_{params['output_dim']}_{params['version']}"
    encoder.load_state_dict(torch.load(f"./models/{model_name}.pt"))
    encoder.eval()

    dataset = WikiCS(root='./data/WikiCS')
    data = dataset[0]

    with torch.no_grad():
        embeddings = encoder(data.x, data.edge_index)

    torch.save(
        embeddings, f"./data/embeddings/node_embeddings_{model_name}.pt")
