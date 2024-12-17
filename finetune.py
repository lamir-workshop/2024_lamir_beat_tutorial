import re

import lightning as L
import mirdata
from sklearn.model_selection import train_test_split

from dataloader import *
from model import *
from train import PLTCN

if __name__ == "__main__":
    num_workers = 1
    brid = mirdata.initialize("brid", version="default")
    # brid.download()

    dataset_tracks = brid.load_tracks()
    dataset_keys = list(dataset_tracks.keys())
    # we do this to avoid loading tracks without annotations
    pattern = r'^\[\d{4}\] M\d+-\d+-[A-Z]+$'
    dataset_keys = [key for key in dataset_keys if bool(re.match(pattern, key))]

    train_keys, test_keys = train_test_split(dataset_keys, test_size=0.2, random_state=42)
    train_keys, val_keys = train_test_split(train_keys, test_size=0.25, random_state=42)
    print(len(train_keys), len(val_keys), len(test_keys))

    train_data = BeatData(dataset_tracks, train_keys, widen=True)
    val_data = BeatData(dataset_tracks, val_keys, widen=True)
    test_data = BeatData(dataset_tracks, test_keys, widen=True)

    train_dataloader = DataLoader(train_data, batch_size=1, num_workers=num_workers)
    val_dataloader = DataLoader(val_data, batch_size=1, num_workers=num_workers)
    test_dataloader = DataLoader(test_data, batch_size=1, num_workers=num_workers)

    learning_rate = 0.005
    n_filters = 20
    kernel_size = 5
    dropout = 0.15
    n_dilations = 11

    ckpt_name = "tcn_202412052128"
    tcn = MultiTracker(n_filters, n_dilations, kernel_size=kernel_size, dropout_rate=dropout)
    model = PLTCN.load_from_checkpoint(f"trained_models/{ckpt_name}.ckpt",
            model=tcn, loss="bce")

    trainer = L.Trainer(
        max_epochs=1
    )

    print("BEFORE FINETUNING")
    trainer.test(model, test_dataloader, verbose=True)

    print("AFTER FINETUNING")
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader, verbose=True)
