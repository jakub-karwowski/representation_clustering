import os
import sys
from pathlib import Path
import pytorch_lightning as pl
from sklearn.mixture import GaussianMixture
import torch
import yaml
from pathlib import Path
import numpy as np
import argparse
from torch_geometric.datasets import WikiCS
import pickle


def main():
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.gmm import GaussianMixtureModel, GMMDataModule, get_gmm_trainer

    parser = argparse.ArgumentParser(description="Provide paramter pack name")
    parser.add_argument('--config', required=True,
                        help="config name", type=str)
    args = parser.parse_args()

    params_path = Path(__file__).resolve().parent.parent / "params.yaml"

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)['wikics']['fit_gmm_stage'][args.config]

    SEED = int(params['seed'])
    ACCELERATOR = 'gpu'

    if params['input_type'] == 'raw':
        p = str(
            Path(__file__).resolve().parent.parent / 'data' / 'WikiCS'
        )
        dataset = WikiCS(root=p)
        data = dataset[0]
        node_embedings = data.x
    elif params['input_type'] == 'embeddings':
        embedings_path = Path(__file__).resolve(
        ).parent.parent / params['input']
        with open(embedings_path, 'rb') as f:
            node_embedings = torch.load(f)
    else:
        raise ValueError(params['encoder'])

    if params['model'] == 'sklearn':
        model = GaussianMixture(10, max_iter=int(
            params['max_epochs'], tol=float(params['tol'])), random_state=SEED)
        model.fit(node_embedings.numpy())
        model_file = Path(__file__).resolve().parent.parent / 'models' / 'gmm' / \
            f"gmm_{params['encoder']}_{params['dim']}_{params['version']}.pt"
        with open('gmm_model.pkl', 'wb') as f:
            pickle.dump(model_file, f)
    elif params['model'] == 'diy':
        datamodule = GMMDataModule(node_embedings)
        torch.manual_seed(SEED)
        trainer = get_gmm_trainer('gmm', max_epochs=int(
            params['max_epochs']), termination_threshold=float(params['tol']), accelerator=ACCELERATOR, patience=5)
        model = GaussianMixtureModel(10, int(params['dim']))
        trainer.fit(model, datamodule=datamodule)
        os.makedirs("./models/gmm", exist_ok=True)
        model_file = Path(__file__).resolve().parent.parent / 'models' / 'gmm' / \
            f"gmm_{params['encoder']}_{params['dim']}_{params['version']}.pt"
        torch.save(model.state_dict(), str(model_file))
    else:
        raise ValueError(params['model'])


if __name__ == "__main__":
    main()
