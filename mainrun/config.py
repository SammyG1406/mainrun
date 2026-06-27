"""
    Hyperparameter and model configuration dataclasses 
    shared across all modules.
"""

import os
import time
from dataclasses import dataclass

MODE = os.environ.get("MODE", "full").lower()
assert MODE in ("smoke", "validate", "full"), f"Unknown MODE: {MODE}"

_MODE_NUM_TITLES = {"smoke": 5_000, "validate": 25_000, "full": 100_000}
_MODE_EPOCHS     = {"smoke": 1,     "validate": 7,      "full": 7}


# Training run settings; mode-dependent fields resolve at import time from MODE.
@dataclass
class Hyperparameters:
    block_size:      int   = 256
    batch_size:      int   = 32
    vocab_size:      int   = 16_000
    n_layer:         int   = 12
    n_head:          int   = 8
    d_model:         int   = 640
    dropout:         float = 0.1
    lr:              float = 6e-4
    weight_decay:    float = 0.1
    evals_per_epoch: int   = 3

    epochs:     int   = _MODE_EPOCHS[MODE]
    seed:       int   = 1337
    num_titles: int   = _MODE_NUM_TITLES[MODE]
    val_frac:   float = 0.10
    log_file:   str   = (
        f"./logs/mainrun_{MODE}_{time.strftime('%Y-%m-%dT%H-%M-%S')}.log"
        if MODE != "full" else "./logs/mainrun.log"
    )


# Immutable architecture spec passed into GPT and its sub-modules at construction.
@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer:    int
    n_head:     int
    d_model:    int
    dropout:    float
