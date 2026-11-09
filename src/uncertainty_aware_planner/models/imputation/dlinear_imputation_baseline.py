# file to define the dlinear baseline model

from pathlib import Path
from typing import Dict, Union

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from pypots.imputation.dlinear import DLinear
from pypots.optim.adam import Adam

def build_model(train_ds: Dict[str, NDArray], cfg: Dict):
    
    model_name = cfg["name"]
    
    if model_name != "DLinear_Baseline":
        raise ValueError(f"Wrong Configuration: expected DLinear_Baseline, got {model_name}")
    
    return DLinear(
        n_steps = dlinear_config["n_steps"],
        n_features = dlinear_config["n_features"],
        moving_avg_window_size = dlinear_config["moving_avg_window_size"],
        individual=False,
        d_model=dlinear_config["d_model"],
        ORT_weight=dlinear_config["ORT_weight"],
        MIT_weight=dlinear_config["MIT_weight"],
        batch_size=dlinear_config["batch_size"],
        epochs=dlinear_config["epochs"],
        patience=dlinear_config["patience"],
        # training_loss=<class 'pypots.nn.modules.loss.MAE'>,
        # validation_metric=<class 'pypots.nn.modules.loss.MSE'>,
        optimizer=Adam(**dlinear_config["optimizer_parameters"]),
        num_workers=0,
        model_saving_strategy='best',
        verbose=True
    )