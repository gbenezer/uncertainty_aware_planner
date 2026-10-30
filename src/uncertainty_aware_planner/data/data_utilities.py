# derived from https://github.com/Dingdong-Inc/frn-50k-baseline/blob/main/latent_demand_recovery/exp/data/generate_data.py
# retrieved 29Oct2026

# Original Code
import numpy as np
from datasets import load_dataset

# relevant constant definitions
HORIZON = 90  # total number of days in traing set
WINDOW_NUMBER = 6  # needs to evenly divide HORIZON, number of folds/splits, cannot be both random and uncorrelated
WINDOW_DURATION = HORIZON // WINDOW_NUMBER  # number of days per window
COVARIATES = [
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
]
PRIMARY_KEYS = ["store_id", "product_id", "dt"]
MAX_MISSING_PATCH = 4
MIN_MISSING_PATCH = 2
NUMBER_OPEN_HOURS = 16

# TODO: change to a more generic directory
DEFAULT_CACHE_DIR = "/home/gil/projects/uncertainty_aware_planner/data"


def mimic_missing(
    patch_ts,
    p=0.5,
    max_missing_patch=MAX_MISSING_PATCH,
    min_missing_patch=MIN_MISSING_PATCH,
):
    """Add synthetic censoring (with known ground truth) to an hourly sales array.

    Real stockouts are already NaN in patch_ts, but their true demand is unknown.
    To get an evaluation signal, this function masks additional entries that were
    actually observed, so the masked values can be used as ground truth.

    Two kinds of artificial missingness are applied, and only to days with no
    stockouts (so the hidden values really are known):
      1. continuous-patch missing: whole consecutive days are hidden
      2. intra-patch missing: part of a single day (a prefix or suffix of hours)
         is hidden

    Args:
        patch_ts: array of shape (batch_size, patch_num, patch_len), i.e.
            (series x windows, days per window, open hours per day). NaN marks
            existing stockout hours.
        p: missing rate; the probability that an eligible day gets intra-day
            masking, and (divided by 10) the chance that a continuous patch is
            started at a given day.
        max_missing_patch: maximum length (in days) of a continuous missing patch,
            also the size of the lookback slice that can be written to.
        min_missing_patch: minimum length (in days) of a continuous missing patch.

    Returns:
        patch_ts_missing: patch_ts with the extra entries set to NaN.
        valid_idx: boolean array, True where a value was observed originally but
            has been masked here (i.e. where ground truth exists).
    """

    # getting the appropriate constants from the input data

    # number of unique series per window times the number of windows
    batch_size = patch_ts.shape[0]

    # the number of days per window
    patch_num = patch_ts.shape[1]

    # number of hours for each day in a window
    patch_len = patch_ts.shape[-1]

    # broadcast range of integer hours to a 1 X 1 X work hours NumPy array
    patch_time = np.arange(patch_len)[None, None, :]  # patch_time which could broadcast
    ## missing mechanism

    # number of already missing (stockout) entries per day
    patch_missing_cnt = np.isnan(patch_ts).sum(axis=-1, keepdims=True)

    # days where there were no stockouts
    non_missing_idx = patch_missing_cnt == 0

    # continuous-patch missing: conti_idx
    # (marks whole days to hide, shape (batch_size, patch_num, 1), broadcast over hours later)

    # running count of consecutive stockout-free days ending at each day
    # (recorded for each day but not used after the loop)
    non_missing_cumsum = np.zeros_like(non_missing_idx.astype(int))

    # current run length of stockout-free days for each series, shape (batch_size, 1)
    sum_vec = np.zeros_like(non_missing_idx[:, 0].astype(int))

    # output mask for days hidden by continuous patches, initially nothing hidden
    conti_idx = np.zeros_like(non_missing_idx)  # (batch_size, day_len, 1)

    # walk through the days of each window in order
    for i in range(patch_num):
        # reset the run length to 0 if day i had any stockout, otherwise keep it
        sum_vec = np.where(non_missing_idx[:, i], sum_vec, 0)

        # extend the run by one day if day i was stockout-free (adds 0 otherwise)
        sum_vec = sum_vec + non_missing_idx[:, i].astype(int)
        non_missing_cumsum[:, i] = sum_vec.copy()

        # only start patches once there are enough preceding days to fill the
        # lookback slice of max_missing_patch days
        if i > max_missing_patch:
            # draw a random patch length in [min_missing_patch, max_missing_patch]
            # for every series
            conti_len = np.random.randint(
                low=min_missing_patch, high=max_missing_patch + 1, size=sum_vec.shape
            )

            # keep the drawn length only if (a) the run of stockout-free days is
            # longer than the patch, so every hidden day has ground truth, and
            # (b) a random draw succeeds with probability p / 10; otherwise 0
            conti_len = np.where(
                (conti_len < sum_vec) & (np.random.rand(*sum_vec.shape) < p / 10),
                conti_len,
                0,
            )

            # build, for every series, the offsets of the last max_missing_patch
            # days ending at day i: the arange is shifted by each row's first
            # entry, so every row becomes 0, 1, ..., max_missing_patch - 1
            conti_tmp = np.arange(batch_size * max_missing_patch).reshape(
                batch_size, max_missing_patch, 1
            )

            # flip the offsets so they count down (max_missing_patch, ..., 1)
            # toward day i, i.e. the "distance" of each day from day i
            conti_tmp = max_missing_patch - (conti_tmp - conti_tmp[:, 0:1])

            # a day is hidden if it lies within conti_len of day i and is not the
            # (distance 1) day i itself, so the patch is conti_len - 1 days long
            # and ends on day i - 1; with conti_len == 0 nothing is hidden
            conti_tmp = (conti_tmp <= conti_len[:, None]) & (conti_tmp > 1)

            # OR the new patch into the lookback slice (days i-max+1 ... i)
            # so patches from different iterations can accumulate
            conti_idx[:, i - max_missing_patch + 1 : i + 1] = (
                conti_idx[:, i - max_missing_patch + 1 : i + 1] | conti_tmp
            )

    # intra-patch missing: intra_idx
    # (hides the start or the end of a day's open hours)

    # choose days to hide part of, with probability p, among stockout-free days only
    intra_rand_idx = (np.random.rand(*patch_missing_cnt.shape) < p) & non_missing_idx

    # random hour at which the hidden stretch begins, in [0, int(patch_len * (1 - p)))
    patch_missing_start_time = np.random.randint(
        low=0, high=int(patch_len * (1 - p)), size=patch_missing_cnt.shape
    )

    # mirrored hour used for the "hide the beginning of the day" variant
    patch_missing_end_time = patch_len - patch_missing_start_time

    # variant A: hide from start_time to the end of the day (a suffix)
    intra_missing_idx_front = patch_time >= patch_missing_start_time

    # variant B: hide from the start of the day up to end_time (a prefix)
    intra_missing_idx_backend = patch_time <= patch_missing_end_time

    # shape (batch_size, patch_num, 1): one coin flip per day to pick the variant
    shape = list(intra_missing_idx_front.shape)
    shape[-1] = 1

    # for each day, use variant A with probability 0.5, otherwise variant B
    intra_missing_idx = np.where(
        np.random.rand(*shape) <= 0.5,
        intra_missing_idx_front,
        intra_missing_idx_backend,
    )

    # only apply the hour mask on the selected days
    intra_idx = intra_rand_idx & intra_missing_idx  # (batch_size, patch_num, patch_len)

    # combine both mechanisms; conti_idx (batch_size, patch_num, 1) broadcasts
    # across the hours of each day
    sample_idx = intra_idx | conti_idx

    # set the selected entries to NaN to create the new censored series
    patch_ts_missing = np.where(sample_idx, np.nan, patch_ts)

    # ground-truth mask: entries that were observed originally (not NaN) but are
    # now missing, so they can be used to score recovered demand
    valid_idx = ~np.isnan(patch_ts) & np.isnan(patch_ts_missing)
    return patch_ts_missing, valid_idx


def load_raw_data_for_imputation(missing_rate=0.5, cache_dir=DEFAULT_CACHE_DIR):
    """Load FreshRetailNet-50K and build windowed, censored hourly sales data.

    Downloads (or reads from cache) the training split of the Hugging Face
    dataset "Dingdong-Inc/FreshRetailNet-50K". Each store-product series spans
    HORIZON (90) days and is split into WINDOW_NUMBER (6) contiguous windows of
    WINDOW_DURATION (15) days. Only the NUMBER_OPEN_HOURS (16) open hours of
    each day (hours 6 to 21) are kept, so each window has 15 * 16 = 240 time
    steps. Hours with a real stockout are set to NaN, and extra artificial
    censoring with known ground truth is added by `mimic_missing`.

    The reshaping assumes that, after sorting by PRIMARY_KEYS, every series
    has exactly HORIZON consecutive daily rows.

    Args:
        missing_rate: probability `p` passed to `mimic_missing`, controlling
            how much additional (artificial) missingness is introduced.
        cache_dir: directory used by `datasets.load_dataset` to cache the
            downloaded data.

    Returns:
        dict with the following keys:
            "train_set": dict with a single key "X", a float array of shape
                (n_windows, n_steps, n_features), where n_windows is
                n_series * WINDOW_NUMBER, n_steps is 240 and n_features is 9.
                The feature axis holds:
                    0: hourly sales (NaN at real stockouts and at artificially
                       masked entries)
                    1-7: the COVARIATES (discount, holiday_flag, activity_flag,
                       precpt, avg_temperature, avg_humidity, avg_wind_level),
                       each normalized by its per-window maximum (+ 0.1) and
                       repeated across the hours of each day
                    8: position within the open day, scaled to [0, 1]
            "ts_origin": array of shape (n_windows, n_steps, 1) with the
                original hourly sales before any NaN masking, used as ground
                truth.
            "valid_idx": boolean array of shape (n_windows, n_steps, 1), True
                where a value was observed originally but masked by
                `mimic_missing` (the entries that can be used for evaluation).
            "params": dict of model configuration values: "n_steps" (240),
                "n_features" (9), "patch_len" (16, hours per day) and "OT"
                (1, the number of target channels, i.e. sales).
    """

    # Load the full dataset
    dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K", cache_dir=cache_dir)

    # Get the training dataset as a Pandas DataFrame
    data = dataset["train"].to_pandas()

    # Sort by store ID, product ID, and dt (string date)
    data = data.sort_values(by=PRIMARY_KEYS)

    # Retrieve the number of unique store-product series within the dataset
    series_num = data.shape[0] // HORIZON

    # define a tuple for repeated application
    data_shape = (series_num * WINDOW_NUMBER, WINDOW_DURATION, 24)

    # convert the nested scaled sales per hour to a NumPy array
    # reshape appropriately, and take only hours that the store is open
    hours_sale = np.array(data["hours_sale"].tolist())
    hours_sale_origin = hours_sale.reshape(*data_shape)[..., 6:22]

    # convert the nested stockout indicator to a NumPy array
    # reshape appropriately, and take only hours that the store is open
    hours_stock_status = np.array(data["hours_stock_status"].tolist())
    hours_stock_status = hours_stock_status.reshape(*data_shape)[..., 6:22]

    # convert zero sales volume where stockout is occurring to NaN values
    hours_sale = np.where(hours_stock_status == 1, np.nan, hours_sale_origin)

    # reshape covariate data
    covariate = data[COVARIATES].values.reshape(
        series_num * WINDOW_NUMBER, WINDOW_DURATION, len(COVARIATES)
    )

    # normalize covariates
    covariate = covariate / (covariate.max(axis=1, keepdims=True) + 0.1)

    # add additional censoring that has ground truth
    hours_sale, valid_idx = mimic_missing(
        hours_sale,
        p=missing_rate,
        max_missing_patch=MAX_MISSING_PATCH,
        min_missing_patch=MIN_MISSING_PATCH,
    )

    # adding necessary axes to input NumPy arrays
    hours_sale_with_feature_axis = hours_sale[..., None]
    covariates_with_hour_axis = covariate[:, :, None, :]

    # broadcasting the covariates across each hour as they are daily values
    broadcasted_covariates = np.broadcast_to(
        covariates_with_hour_axis, hours_sale.shape + (len(COVARIATES),)
    )

    # concatenate along the feature axis
    data = np.concatenate(
        [hours_sale_with_feature_axis, broadcasted_covariates], axis=-1
    )

    # create a feature encoding an hourly position and broadcast it to the correct shape
    hour_position = np.broadcast_to(
        np.arange(NUMBER_OPEN_HOURS)[None, None, :, None] / (NUMBER_OPEN_HOURS - 1),
        hours_sale_with_feature_axis.shape,
    )

    # concatenate to the dataset
    # first dimension on feature axis is scaled sales
    # next dimensions are normalized covariates
    # final dimension is hour position within the work day
    data = np.concatenate([data, hour_position], axis=-1)

    # get that number of features in final dataset
    number_features = data.shape[-1]

    # reshape into hours in window by number of features NumPy array
    data = data.reshape(-1, WINDOW_DURATION * NUMBER_OPEN_HOURS, number_features)

    # reshaping other relevant NumPy arrays
    valid_idx = valid_idx.reshape(-1, WINDOW_DURATION * NUMBER_OPEN_HOURS, 1)
    hours_sale_origin = hours_sale_origin.reshape(
        -1, WINDOW_DURATION * NUMBER_OPEN_HOURS, 1
    )

    train_set = {"X": data}
    data = {
        "train_set": train_set,
        "ts_origin": hours_sale_origin,
        "valid_idx": valid_idx,
        "params": {
            "n_steps": WINDOW_DURATION * NUMBER_OPEN_HOURS,
            "n_features": number_features,
            "patch_len": NUMBER_OPEN_HOURS,
            "OT": 1,
        },
    }
    return data
