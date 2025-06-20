import os
import sys
from pathlib import Path
import pytorch_lightning as pl
import torch
from pathlib import Path
import yaml
from torch_geometric.datasets import WikiCS
import argparse

def main():
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.gca import GATEncoderBN, GCNEncoderBN
    params_path = Path(__file__).resolve().parent.parent / "params.yaml"

    parser = argparse.ArgumentParser(description="Provide paramter pack name")
    parser.add_argument('--config', required=True,
                        help="config name", type=str)
    args = parser.parse_args()

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)['wikics']['embed_stage'][args.config]

    if params['encoder'] == 'gate':
        encoder = GATEncoderBN(
            in_dim=params['input_dim'],
            hidden_dim=params['hidden_dim'],
            out_dim=params['output_dim']
        )
    elif params['encoder'] == 'gcn':
        encoder = GCNEncoderBN(
            in_dim=params['input_dim'],
            hidden_dim=params['hidden_dim'],
            out_dim=params['output_dim']
        )
    else:
        raise ValueError(params['encoder'])

    model_name = f"gca_{params['encoder']}_{params['output_dim']}_{params['version']}"
    encoder.load_state_dict(torch.load(f"./models/{model_name}.pt"))
    encoder.eval()

    dataset = WikiCS(root='./data/WikiCS')
    data = dataset[0]

    with torch.no_grad():
        embeddings = encoder(data.x, data.edge_index)

    os.makedirs('./data/embeddings/', exist_ok=True)
    torch.save(
        embeddings, f"./data/embeddings/node_embeddings_{model_name}.pt")


if __name__ == "__main__":
    main()