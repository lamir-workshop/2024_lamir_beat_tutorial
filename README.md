# LAMIR: Beat Tracking Tutorial

source code for the 2024 LAMIR beat tracking tutorial

authored by [Giovana Morais](https://github.com/giovana-morais)

---

## file overview

* `config.py`: hyperparameters for training and finetuning.
* `dataloader.py`: dataloader for audio and beat data
* `model.py`: torch beat tracking model
* `pl_model.py`: pytorch lightning model
* `train.py`: training script
* `finetune.py`: finetuning script 

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

run the script by specifying which dataset you want to finetune it:

```bash
python finetune.py --dataset=brid
# or
python finetune.py --dataset=candombe
```

if you want to download the datasets, just provide the `--download` flag

```bash
python finetune.py --dataset=brid --download
# or
python finetune.py --dataset=candombe --download
```

**NOTE**: this will run the download with `force_overwrite`.
