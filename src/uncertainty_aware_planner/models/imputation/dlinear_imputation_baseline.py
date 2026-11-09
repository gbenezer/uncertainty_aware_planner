# file to define the dlinear baseline model

from pathlib import Path
from typing import Dict

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from pypots.imputation.dlinear import DLinear
from pypots.optim.adam import Adam

def build_model(train_ds: NDArray, cfg: Dict):
    pass

def fit(model, train_ds, cfg, run_dir):
    pass

def predict_quantiles(ckpt, eval_ds, cfg):
    pass