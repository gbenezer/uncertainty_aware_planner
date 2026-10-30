# File to hold library-wide global constants

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
