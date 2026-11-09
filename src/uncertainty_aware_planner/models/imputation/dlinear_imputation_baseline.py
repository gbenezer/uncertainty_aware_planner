# file to define the dlinear baseline model

from pathlib import Path

import numpy as np
import pandas as pd

from pypots.imputation.dlinear import DLinear
from pypots.optim.adam import Adam