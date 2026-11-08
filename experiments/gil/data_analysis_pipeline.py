from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
ORIGINAL_DATASET_PATH = DATA_DIR / "second_dataset_authors_experiment.csv"
FIFTH_EXPERIMENT_PATH = DATA_DIR / "fifth_dlinear_baseline_experiment.csv"
SIXTH_EXPERIMENT_PATH = DATA_DIR / "sixth_dlinear_baseline_experiment.csv"
SEVENTH_EXPERIMENT_PATH = DATA_DIR / "seventh_dlinear_baseline_experiment.csv"

author_run_df = pd.read_csv(ORIGINAL_DATASET_PATH)
fifth_experiment_df = pd.read_csv(FIFTH_EXPERIMENT_PATH)
sixth_experiment_df = pd.read_csv(SIXTH_EXPERIMENT_PATH)
seventh_experiment_df = pd.read_csv(SEVENTH_EXPERIMENT_PATH)

summary_author_wape_df = author_run_df.groupby(["new_fn","pooled"]).agg(
    {
        "wape": ["min", "max", "median", "mean", "std", "count"]
    }
)
summary_author_wpe_df = author_run_df.groupby(["new_fn"]).agg(
    {
        "wpe": ["min", "max", "median", "mean", "std", "count"]
    }
)
summary_experiment_wape_df = seventh_experiment_df.groupby(["leaked", "daily_pool"]).agg(
    {
        "wape": ["min", "max", "median", "mean", "std", "count"]
    }
)
summary_experiment_wpe_df = seventh_experiment_df.groupby(["leaked"]).agg(
    {
        "wpe": ["min", "max", "median", "mean", "std", "count"]
    }
)
print("Author WAPE")
print(summary_author_wape_df)
print("Author WPE")
print(summary_author_wpe_df)
print("Experimental WAPE, 7")
print(summary_experiment_wape_df)
print("Experimental WPE, 7")
print(summary_experiment_wpe_df)