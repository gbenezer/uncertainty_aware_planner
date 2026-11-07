from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
ORIGINAL_DATASET_PATH = DATA_DIR / "second_dataset_authors_experiment.csv"
FIFTH_EXPERIMENT_PATH = DATA_DIR / "fifth_dlinear_baseline_experiment.csv"

author_run_df = pd.read_csv(ORIGINAL_DATASET_PATH)
fifth_experiment_df = pd.read_csv(FIFTH_EXPERIMENT_PATH)
fifth_experiment_df = fifth_experiment_df[fifth_experiment_df["daily_pool"]]

summary_author_wape_df = author_run_df.groupby(["environment","new_fn","n_covar"]).agg(
    {
        "wape": ["min", "max", "median", "mean", "std", "count"]
    }
)
summary_author_wpe_df = author_run_df.groupby(["environment","new_fn","n_covar"]).agg(
    {
        "wpe": ["min", "max", "median", "mean", "std", "count"]
    }
)
summary_experiment_wape_df = fifth_experiment_df.groupby(["fold"]).agg(
    {
        "wape": ["min", "max", "median", "mean", "std", "count"]
    }
)
summary_experiment_wpe_df = fifth_experiment_df.groupby(["fold"]).agg(
    {
        "wpe": ["min", "max", "median", "mean", "std", "count"]
    }
)
print("Author WAPE")
print(summary_author_wape_df)
print("Author WPE")
print(summary_author_wpe_df)
print("Experimental WAPE")
print(summary_experiment_wape_df)
print("Experimental WPE")
print(summary_experiment_wpe_df)