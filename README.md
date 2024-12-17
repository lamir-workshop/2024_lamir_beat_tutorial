# LAMIR: Beat Tracking Tutorial

Source code for the 2024 LAMIR beat tracking tutorial

---

## file overview

* `config.py`: has hyperparameters for training and finetuning.
* `dataloader.py`: has the dataloader for audio and beat data
* `model.py`: has the torch beat tracking model
* `pl_model.py`: implements pytorch lighning model
* `train.py`: training script
* `finetune.py`: finetuning script (BRID only for now)

## installing dependencies
```bash
pip install -r requirements.txt
```

## training
alter whatever hyperparameters you with on the `PARAMS_TRAIN` inside `config.py`
```bash
python train.py
```

## finetuning
we provide one pre-trained model for the finetuning. to use it, keep the
`PARAMS_FINETUNE` dictionary as is.

if you trained a new model and want to use it, please make sure to update the
`CHECKPOINT_NAME` variable in the `config.py` file accordingly.

```python
PARAMS_FINETUNE = {
    "MAX_NUM_FILES": 5,
    "LEARNING_RATE": 0.005,
    "N_FILTERS": 20,
    "KERNEL_SIZE": 5,
    "DROPOUT": 0.15,
    "N_DILATIONS": 11,
    "N_EPOCHS": 5,
    "LOSS": "BCE",
    "NUM_WORKERS": 7,
    # Change here with your checkpoint filename
    "CHECKPOINT_FILE": <new_filename>
}
```

and then run

```bash
python finetune.py
```

**NOTE**: if you wish to expand the finetuning script to candombe, as shown in the
tutorial, just replace the mirdata dataset from `brid` to `candombe` and remove
the pattern matching. the repo will be soon update to add the candombe
finetuning example.
