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
# number of hours in a day in general
NUMBER_HOURS_IN_DAY = 24

# number of non-overlapping windows to use to split the data across all experiments
# needs to evenly divide HORIZON, number of folds/splits, cannot be both random and uncorrelated
WINDOW_NUMBER = 6

# Imputation window duration is 28 days training and validation
IMPUTATION_WINDOW_DURATION = 28

