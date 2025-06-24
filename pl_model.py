import madmom
import mir_eval
import lightning as L
import torch
import torch.nn.functional as F

import losses


class PLTCN(L.LightningModule):
    def __init__(self, model, params):
        super().__init__()
        self.model = model
        self.loss = self._get_loss_fn(params["LOSS"])
        self.learning_rate = params["LEARNING_RATE"]
        self.test_fmeasure = []

    def _get_loss_fn(self, loss):
        # for now using only BCE
        return losses.masked_binary_cross_entropy

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # get annotations
        x = batch["x"]
        beats_ann = batch["beats"]
        downbeats_ann = batch["downbeats"]

        # get detections
        output = self(x)
        beats_det = output["beats"].squeeze(-1)
        downbeats_det = output["downbeats"].squeeze(-1)

        # calculate losses
        beat_loss = self.loss(beats_det, beats_ann)
        downbeat_loss = self.loss(downbeats_det, downbeats_ann)
        loss = beat_loss + downbeat_loss

        # log them
        self.log("train_beat_loss", beat_loss, prog_bar=True, on_epoch=True)
        self.log("train_downbeat_loss", downbeat_loss, prog_bar=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["x"]
        beats_ann = batch["beats"]
        downbeats_ann = batch["downbeats"]

        output = self(x)
        beats_det = output["beats"].squeeze(-1)
        downbeats_det = output["downbeats"].squeeze(-1)

        beat_loss = self.loss(beats_det, beats_ann)
        downbeat_loss = self.loss(downbeats_det, downbeats_ann)
        loss = beat_loss + downbeat_loss

        self.log("val_beat_loss", beat_loss, prog_bar=True, on_epoch=True)
        self.log("val_downbeat_loss", downbeat_loss, prog_bar=True, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x = batch["x"]
        beats_target = batch["beats_ann"].detach().cpu().numpy().squeeze()
        downbeats_target = batch["downbeats_ann"].detach().cpu().numpy().squeeze()
        output = self(x)

        beats_act = output["beats"].squeeze().detach().cpu().numpy()
        downbeats_act = output["downbeats"].squeeze().detach().cpu().numpy()
        print("beats_act", beats_act.shape)
        print("downbeats_act", downbeats_act.shape)

        beat_dbn = madmom.features.beats.DBNBeatTrackingProcessor(
            min_bpm=55.0, max_bpm=215.0, fps=100, transition_lambda=100, online=False
        )
        beats_prediction = beat_dbn(beats_act)

        downbeat_dbn = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
            beats_per_bar=[2, 3, 4], min_bpm=55.0, max_bpm=215.0, fps=100,
            transition_lambda=100, threshold=0.005
        )
        downbeats_prediction = downbeat_dbn(downbeats_act)
        print("downbeats_activation", downbeats_act)
        print("downbeats_prediction", downbeats_prediction)

        # add mir_eval call to calculate metrics
        # check
        # https://mir-evaluation.github.io/mir_eval/#module-mir_eval.beat
        # if you want to explore other metrics
        beat_fmeasure = mir_eval.beat.f_measure(beats_target, beats_prediction)
        downbeat_fmeasure = mir_eval.beat.f_measure(downbeats_target, downbeats_prediction)
        self.test_fmeasure.append(fmeasure)
        self.log("beat_fmeasure", beat_fmeasure, on_step=True)
        self.log("downbeat_fmeasure", downbeat_fmeasure, on_step=True)

        return fmeasure

    # Uncomment to see the metrics per item
    # def on_test_epoch_end(self):
    #     for idx, v in enumerate(self.test_fmeasure):
    #         self.log(f"item_{idx}", v)

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=0.005)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.2,
                patience=10,
                threshold=1e-3,
                cooldown=0,
                min_lr=1e-7,
            ),
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]
