# derived from https://github.com/Dingdong-Inc/frn-50k-baseline/blob/main/latent_demand_recovery/exp/data/generate_data.py
# retrieved 29Oct2026

# Original Code
import os
import pandas as pd
import numpy as np
from datasets import load_dataset

# relevant constant definitions
HORIZON = 90 # total number of days in traing set
WINDOW_NUMBER = 6 # needs to evenly divide HORIZON, number of folds/splits, cannot be both random and uncorrelated
WINDOW_DURATION = HORIZON // WINDOW_NUMBER # number of days per window
COVARIATES = ['discount', 'holiday_flag','activity_flag', 'precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level']
PRIMARY_KEYS = ['store_id', 'product_id', 'dt']
MAX_MISSING_PATCH = 7
MIN_MISSING_PATCH = 3
NUMBER_OPEN_HOURS = 16

# TODO: change to a more generic directory
DEFAULT_CACHE_DIR = "/home/gil/projects/uncertainty_aware_planner/data"

def mimic_missing(patch_ts, p=0.5, max_missing_patch=7, min_missing_patch=3):
    
    # getting the appropriate constants from the input data
    
    # number of unique series per window times the number of windows
    batch_size = patch_ts.shape[0]
    
    # the number of days per window
    patch_num = patch_ts.shape[1]
    
    # number of hours for each day in a window
    patch_len = patch_ts.shape[-1]
    
    # broadcast range of integer hours to a 1 X 1 X work hours NumPy array
    patch_time = np.arange(patch_len)[None,None,:] # patch_time which could broadcast
    ## missing mechanism
    
    # number of already missing (stockout) entries per day
    patch_missing_cnt = np.isnan(patch_ts).sum(axis=-1, keepdims=True)
    
    # days where there were no stockouts
    non_missing_idx = patch_missing_cnt==0
    
    # continuous-patch missing: conti_idx 
    non_missing_cumsum = np.zeros_like(non_missing_idx.astype(int))
    sum_vec = np.zeros_like(non_missing_idx[:,0].astype(int))
    conti_idx = np.zeros_like(non_missing_idx) # (batch_size, day_len, 1)
    for i in range(patch_num):
        sum_vec = np.where(non_missing_idx[:,i], sum_vec, 0)
        sum_vec = sum_vec + non_missing_idx[:,i].astype(int)
        non_missing_cumsum[:,i] = sum_vec.copy()
        if i>max_missing_patch:
            conti_len = np.random.randint(low=min_missing_patch, high=max_missing_patch+1, size=sum_vec.shape)
            conti_len = np.where((conti_len < sum_vec) & (np.random.rand(*sum_vec.shape)<p/10), conti_len, 0)
            conti_tmp = np.arange(batch_size * max_missing_patch).reshape(batch_size, max_missing_patch, 1)
            conti_tmp = max_missing_patch - (conti_tmp - conti_tmp[:,0:1])
            conti_tmp = (conti_tmp <= conti_len[:,None]) & (conti_tmp > 1)
            conti_idx[:,i-max_missing_patch+1:i+1] = conti_idx[:,i-max_missing_patch+1:i+1] | conti_tmp
            
    # intra-patch missing: intra_idx
    intra_rand_idx = (np.random.rand(*patch_missing_cnt.shape) < p) & non_missing_idx  
    patch_missing_start_time = np.random.randint(low=0, high=int(patch_len*(1-p)), size=patch_missing_cnt.shape)
    patch_missing_end_time = patch_len - patch_missing_start_time
    intra_missing_idx_front = patch_time>=patch_missing_start_time
    intra_missing_idx_backend = patch_time<=patch_missing_end_time
    shape = list(intra_missing_idx_front.shape)
    shape[-1] = 1
    intra_missing_idx = np.where(np.random.rand(*shape)<=0.5, intra_missing_idx_front, intra_missing_idx_backend)
    intra_idx = intra_rand_idx & intra_missing_idx # (batch_size, patch_num, patch_len)


    # valid_idx = np.squeeze(intra_rand_idx | conti_idx)
    sample_idx = intra_idx | conti_idx

    patch_ts_missing = np.where(sample_idx, np.nan, patch_ts)
    valid_idx = ~np.isnan(patch_ts)&np.isnan(patch_ts_missing)
    return patch_ts_missing, valid_idx
        
        
def load_data(missing_rate=0.5, cache_dir = DEFAULT_CACHE_DIR):
    
    # Load the full dataset
    dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K", cache_dir=cache_dir)
    
    # Get the training dataset as a Pandas DataFrame
    data = dataset['train'].to_pandas()

    # Sort by store ID, product ID, and dt (string date)
    data = data.sort_values(by=PRIMARY_KEYS)
    
    # Retrieve the number of unique store-product series within the dataset
    series_num = data.shape[0]//HORIZON
    
    # define a tuple for repeated application
    data_shape = (series_num*WINDOW_NUMBER, WINDOW_DURATION, 24)

    # convert the nested scaled sales per hour to a NumPy array
    # reshape appropriately, and take only hours that the store is open
    hours_sale = np.array(data['hours_sale'].tolist())
    hours_sale_origin = hours_sale.reshape(*data_shape)[...,6:22]
    
    # convert the nested stockout indicator to a NumPy array
    # reshape appropriately, and take only hours that the store is open
    hours_stock_status = np.array(data['hours_stock_status'].tolist())
    hours_stock_status = hours_stock_status.reshape(*data_shape)[...,6:22]
    
    # convert zero sales volume where stockout is occurring to NaN values
    hours_sale = np.where(hours_stock_status==1, np.nan, hours_sale_origin)
    
    # reshape covariate data
    covariate = data[COVARIATES].values.reshape(series_num*WINDOW_NUMBER, WINDOW_DURATION, len(COVARIATES))
    
    # normalize covariates
    covariate = covariate/(covariate.max(axis=1, keepdims=True)+0.1)

    # add additional censoring that has ground truth
    hours_sale, valid_idx = mimic_missing(hours_sale, p=missing_rate, max_missing_patch=MAX_MISSING_PATCH, min_missing_patch=MIN_MISSING_PATCH)
    
    # adding necessary axes to input NumPy arrays
    hours_sale_with_feature_axis = hours_sale[...,None]
    covariates_with_hour_axis = covariate[:,:,None,:]
    
    # broadcasting the covariates across each hour as they are daily values
    broadcasted_covariates = np.broadcast_to(covariates_with_hour_axis, hours_sale.shape + (len(COVARIATES),))
    
    # concatenate along the feature axis
    data = np.concatenate([hours_sale_with_feature_axis, broadcasted_covariates], axis=-1)
    
    # create a feature encoding an hourly position and broadcast it to the correct shape
    hour_position = np.broadcast_to(np.arange(NUMBER_OPEN_HOURS)[None,None,:,None]/(NUMBER_OPEN_HOURS - 1), hours_sale_with_feature_axis.shape)

    # concatenate to the dataset
    # first dimension on feature axis is scaled sales
    # next dimensions are normalized covariates
    # final dimension is hour position within the work day
    data = np.concatenate([data, hour_position], axis=-1)
    
    # get that number of features in final dataset
    number_features = data.shape[-1]
    
    # reshape into hours in window by number of features NumPy array
    data = data.reshape(-1, WINDOW_DURATION*NUMBER_OPEN_HOURS, number_features)
    
    # reshaping other relevant 
    valid_idx = valid_idx.reshape(-1, WINDOW_DURATION*NUMBER_OPEN_HOURS, 1)
    hours_sale_origin = hours_sale_origin.reshape(-1, WINDOW_DURATION*NUMBER_OPEN_HOURS, 1)
    
    train_set={'X': data}
    data = {
        'train_set': train_set,
        'ts_origin': hours_sale_origin,
        'valid_idx': valid_idx,
        'params': {
            'n_steps': WINDOW_DURATION*NUMBER_OPEN_HOURS,
            'n_features': number_features,
            'patch_len': NUMBER_OPEN_HOURS,
            'OT':1
        }
    }
    return data