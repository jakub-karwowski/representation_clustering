import os
import sys
from pathlib import Path
import pytorch_lightning as pl


def main():
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.load_data import get_dataset_lightning_data
    from src.gca import GCABase
    from src.gca import GATEncoderBN

    model = GCABase(GATEncoderBN(in_dim=300, hidden_dim=256, out_dim=128))

    trainer = pl.Trainer(
        precision=16,
        devices=1,
        max_epochs=5,
        accelerator="cpu",
        logger=pl.loggers.TensorBoardLogger(
            save_dir="./models/logs/",
            name="gca_base",
        ),
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
    )

    datamodule = get_dataset_lightning_data('WikiCS')
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
