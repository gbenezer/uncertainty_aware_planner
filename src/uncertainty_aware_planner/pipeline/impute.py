"""Imputation stage: train one model on one fold (or the final split) and write CONTRACTS.md 2.5 tables.

    python -m uncertainty_aware_planner.pipeline.impute \
        --config configs/forecasting_models/temporal_fusion_transformer.yaml \
        --demand-daily data/processed/demand_daily/<imputation-run-id>.parquet \
        --seed 2025 [--fold 3 | --eval-parts calib test] [--smoke]

Fold k trains on the fold's train window and forecasts its val window (rolling 7-day origins). Without
--fold the final model trains on 03-28..06-18 and forecasts the calibration and test weeks from the same
checkpoint, so conformal offsets fitted on calib apply to the model that made the test forecasts.

Outputs: <output_dir>/<run_id>/{fold<k>,final_<part>}.parquet and <run_dir>/<run_id>/{fold<k>,final}/ with
the checkpoint, training logs, all predicted quantiles and run_log.json.
"""

# Each architecture module provides build_model(train_ds, cfg), fit(model, train_ds, cfg, run_dir) -> ckpt
# and predict_quantiles(ckpt, eval_ds, cfg) -> (quantiles, index, decoder_lengths).
ARCHITECTURES = {"TimesNet_Quantile": None, "DLinear_Baseline": None, "DLinear_Quantile": None}