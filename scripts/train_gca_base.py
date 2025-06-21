import os
import sys
from pathlib import Path
import pytorch_lightning as pl
import torch
import yaml
from pathlib import Path
import yaml
import argparse
from pytorch_lightning.callbacks import EarlyStopping

def main():
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.load_data import get_dataset_lightning_data
    from src.gca import GCABase
    from src.gca import GCNEncoderBN, GATEncoderBN

    parser = argparse.ArgumentParser(description="Provide paramter pack name")
    parser.add_argument('--config', required=True,
                        help="config name", type=str)
    args = parser.parse_args()

    params_path = Path(__file__).resolve().parent.parent / "params.yaml"

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)['wikics']['train_stage'][args.config]

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
    model = GCABase(
        encoder=encoder,
        lr=float(params['lr']),
        weight_decay=float(params['weight_decay'])
    )

    trainer = pl.Trainer(
        max_epochs=params['max_epochs'],
        accelerator="gpu",
        logger=pl.loggers.TensorBoardLogger(
            save_dir="./data/logs/",
            name=model_name,
        ),
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        callbacks=[
            EarlyStopping(
                monitor="train/loss",   
                min_delta=1e-8,         
                patience=15,             
                verbose=True,
                mode="min"             
        )]
    )

    datamodule = get_dataset_lightning_data('WikiCS')
    trainer.fit(model, datamodule=datamodule)

    torch.save(model.encoder.state_dict(), f"./models/{model_name}.pt")


if __name__ == "__main__":
    main()
