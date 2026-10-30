# derived from https://github.com/Dingdong-Inc/frn-50k-baseline/blob/main/latent_demand_recovery/exp/app.py
# retrieved 30Oct2026

import random
import numpy as np
import torch

def set_seed(seed_value=1024):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    return