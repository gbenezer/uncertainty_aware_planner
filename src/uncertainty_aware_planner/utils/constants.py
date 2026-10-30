# File to hold library-wide global constants
from itertools import combinations


# total number of days in traing set
HORIZON = 90

# total number of days in test set
TEST_HORIZON = 7

# raw data covariates
COVARIATES = [
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
]

# raw data primary keys
PRIMARY_KEYS = ["store_id", "product_id", "dt"]

# number of hours the store is open
NUMBER_OPEN_HOURS = 16

# number of non-overlapping windows to use to split the data across all experiments
# needs to evenly divide HORIZON, number of folds/splits, cannot be both random and uncorrelated
WINDOW_NUMBER = 6

# derived constant; number of days per window, needs to be an integer
WINDOW_DURATION = HORIZON // WINDOW_NUMBER

# train-test splits for imputation by window index, with last window reserved for calibration
IMPUTATION_SPLITS = [combination for combination in combinations(range(WINDOW_NUMBER - 1), 2)]