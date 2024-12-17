# LAMIR: Beat Tracking Tutorial

* `config.py`: has parameters for training and fine tuning.
* `dataloader.py`: has the dataloader for audio and beat data
* `model.py`: has the torch beat tracking model
* `pl_model.py`: implements pytorch lighning model
* `train.py`: training script
* `finetune.py`: finetuning script (BRID only for now)

## dependencies
```bash
pip install -r requirements.txt
```

## training
```bash
python train.py
```

## finetuning
if you trained a new model, please make sure to update the `CHECKPOINT_NAME`
variable in the `config.py` file accordingly.
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
