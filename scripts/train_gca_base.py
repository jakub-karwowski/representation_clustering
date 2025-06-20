import os
import sys
from pathlib import Path
import pytorch_lightning as pl
import torch
import yaml


def main():
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.load_data import get_dataset_lightning_data
    from src.gca import GCABase
    from src.gca import GATEncoderBN

    params = yaml.safe_load(open("../params.yaml", 'r')
                            )['wikics']['train_stage']

    model_name = f"gca_{params['output_dim']}_{params['version']}"
    model = GCABase(
        encoder=GATEncoderBN(
            in_dim=params['input_dim'], hidden_dim=params['hidden_dim'], out_dim=params['output_dim']),
        lr=params['lr'],
        weight_decay=params['weight_decay']
    )

    trainer = pl.Trainer(
        max_epochs=params['max_epochs'],
        accelerator="gpu",
        logger=pl.loggers.TensorBoardLogger(
            save_dir="./models/logs/",
            name=model_name,
        ),
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
    )

    datamodule = get_dataset_lightning_data('WikiCS')
    trainer.fit(model, datamodule=datamodule)

    torch.save(model.encoder.state_dict(), f"./models/{model_name}.pt")


if __name__ == "__main__":
    main()
