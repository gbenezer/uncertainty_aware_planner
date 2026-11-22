from pathlib import Path
from typing import Union
import pandas as pd
import uncertainty_aware_planner.data.helper_functions as help_fun

INFERENCE_DATA_DIR = Path("results/imputation/pretrained_model_eval/model_outputs")
EXAMPLE_FILE = "pretrained_model_inference_output_seed_86400_DLinear_Imputation_Quantile_3_quantiles.csv"
example_file_path = INFERENCE_DATA_DIR / EXAMPLE_FILE

test_df = help_fun.import_inference_table(example_file_path)
series_hourly = help_fun.get_series_hourly()
test_df_with_data = help_fun.add_series_metadata(
    imputed=test_df, series_hourly=series_hourly
)

print(test_df_with_data.head())
for col in test_df_with_data.columns:
    print(col)
    print(test_df_with_data[col].dtype)
