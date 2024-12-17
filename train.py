from datetime import datetime

import lightning as L
import mirdata
import mir_eval
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from lightning.pytorch.loggers import WandbLogger

from dataloader import *
from model import *

class PLTCN(L.LightningModule):
    def __init__(self, model, loss):
        super().__init__()
        self.model = model
        self.loss = self._get_loss_fn(loss)
        self.test_fmeasure = []


    def _get_loss_fn(self, loss):
        # for now using only BCE
        return F.binary_cross_entropy


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        x = batch["x"]
        beats_ann = batch["beats"]
        output = self(x)
        beats_det = output["beats"].squeeze(-1)
        loss = self.loss(beats_det, beats_ann)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x = batch["x"]
        beats_ann = batch["beats"]
        output = self(x)
        beats_det = output["beats"].squeeze(-1)
        loss = self.loss(beats_det, beats_ann)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss


    def test_step(self, batch, batch_idx):
        x = batch["x"]
        beats_target = batch["beats_ann"].detach().cpu().numpy().squeeze()
        output = self(x)
        beats_act = output["beats"].squeeze().detach().cpu().numpy()

        beat_dbn = madmom.features.beats.DBNBeatTrackingProcessor(
            min_bpm=55.0, max_bpm=215.0, fps=100, transition_lambda=100,
            online=False
        )
        beats_prediction = beat_dbn(beats_act)

        # add mir_eval call to calculate metrics
        fmeasure = mir_eval.beat.f_measure(beats_target, beats_prediction)
        self.test_fmeasure.append(fmeasure)
        self.log("test_fmeasure", fmeasure, on_step=True)


        return fmeasure

    # Uncomment to see the metrics per item
    # def on_test_epoch_end(self):
    #     for idx, v in enumerate(self.test_fmeasure):
    #         self.log(f"item_{idx}", v)


    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=0.005)
        scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.2, patience=10,
                    threshold=1e-3, cooldown=0, min_lr=1e-7
                ),
                "monitor": "val_loss"
            }
        return [optimizer], [scheduler]


def load_tracks(datasets):
    tracks = {}
    for i in datasets:
        d_tracks = i.load_tracks()
        tracks_to_keep = d_tracks.copy()

        # remove tracks without beat annotations
        for tid, track in d_tracks.items():
            try:
                beat_ann = track.beats.times
            except AttributeError:
                del tracks_to_keep[tid]
                print(f"{tid} has no beat information. skipping\n")

        tracks |= tracks_to_keep
    return list(tracks.keys()), tracks


if __name__ == "__main__":
    NUM_WORKERS = 5
    CKPTS_DIR = "trained_models"
    gtzan_mini = mirdata.initialize("gtzan_genre", version="mini")
    gtzan_mini.download()

    dataset_keys, dataset_tracks = load_tracks([gtzan_mini])

    train_keys, test_keys = train_test_split(dataset_keys, test_size=0.2, random_state=42)
    train_keys, val_keys = train_test_split(train_keys, test_size=0.25, random_state=42)

    print(len(train_keys), len(val_keys), len(test_keys))

    train_data = BeatData(dataset_tracks, train_keys, widen=True)
    val_data = BeatData(dataset_tracks, val_keys, widen=True)
    test_data = BeatData(dataset_tracks, test_keys, widen=True)

    train_dataloader = DataLoader(train_data, batch_size=1, num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(val_data, batch_size=1, num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(test_data, batch_size=1, num_workers=NUM_WORKERS)

    learning_rate = 0.005
    n_filters = 20
    kernel_size = 5
    dropout = 0.15
    n_dilations = 11

    tcn = MultiTracker(n_filters, n_dilations, kernel_size=kernel_size, dropout_rate=dropout)
    model = PLTCN(tcn, "bce")

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    ckpt_name = f"tcn_{timestamp}"
    # ckpt_name = "tcn_202412051834"
    trainer = L.Trainer(
        max_epochs=20,
        gradient_clip_val=0.5,
        callbacks=[
            ModelCheckpoint(
                dirpath=CKPTS_DIR,
                filename=ckpt_name,
                monitor="val_loss",
                auto_insert_metric_name=True,
                save_top_k=1 # save top two best models for this criteron
            )]
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model=model,
                 dataloaders=test_dataloader,
                 ckpt_path=f"{CKPTS_DIR}/{ckpt_name}.ckpt",
                 verbose=True
    )
