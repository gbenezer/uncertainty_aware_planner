
from pathlib import Path
from datetime import datetime, timezone
import csv

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# Page setup
st.set_page_config(
    page_title="Uncertainty-Aware Demand Planning",
    layout="wide",
)

st.title("Uncertainty-Aware Demand Planning")
st.caption("FreshRetailNet-50K | Processed Daily Demand")


# File paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demand_daily"
    / "imp-none-censored.parquet"
)

OVERRIDES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "planner_edits.csv"
)

RECOVERED_PATH = (
    PROJECT_ROOT
    / "results"
    / "dashboard"
    / "recovered_demand.csv"
)

FORECAST_PATH = (
    PROJECT_ROOT
    / "results"
    / "dashboard"
    / "forecast_7day.csv"
)

OVERRIDE_COLUMNS = [
    "timestamp",
    "store_id",
    "product_id",
    "date",
    "old_value",
    "new_value",
    "corrected",
    "reason",
]


# Load processed demand data
@st.cache_data(show_spinner="Loading processed demand data...")
def load_data():
    data = pd.read_parquet(
        DATA_PATH,
        columns=[
            "store_id",
            "product_id",
            "date",
            "demand_observed",
            "demand_recovered",
            "demand_total",
            "stock_hour6_22_cnt",
            "discount",
            "holiday_flag",
            "activity_flag",
            "avg_temperature",
            "avg_humidity",
        ],
    )

    data["date"] = pd.to_datetime(data["date"])
    return data


# Load saved corrections
def load_overrides():
    if (
        not OVERRIDES_PATH.exists()
        or OVERRIDES_PATH.stat().st_size == 0
    ):
        return pd.DataFrame(columns=OVERRIDE_COLUMNS)

    data = pd.read_csv(
        OVERRIDES_PATH,
        dtype={
            "store_id": str,
            "product_id": str,
            "date": str,
        },
    )

    if "corrected" not in data.columns:
        data["corrected"] = True
    else:
        data["corrected"] = (
            data["corrected"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )

    for column in OVERRIDE_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA

    return data[OVERRIDE_COLUMNS]


# Save correction
def save_override(correction):
    OVERRIDES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        OVERRIDES_PATH.exists()
        and OVERRIDES_PATH.stat().st_size > 0
    ):
        with OVERRIDES_PATH.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            existing_columns = next(
                csv.reader(file),
                [],
            )

        if existing_columns != OVERRIDE_COLUMNS:
            existing = load_overrides()
            existing.to_csv(
                OVERRIDES_PATH,
                index=False,
            )

    write_header = (
        not OVERRIDES_PATH.exists()
        or OVERRIDES_PATH.stat().st_size == 0
    )

    with OVERRIDES_PATH.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=OVERRIDE_COLUMNS,
        )

        if write_header:
            writer.writeheader()

        writer.writerow(correction)


# Load model output
@st.cache_data(show_spinner=False)
def load_model_output(path, modified_time):
    data = pd.read_csv(
        path,
        dtype={
            "store_id": str,
            "product_id": str,
        },
    )

    return data


# Filter selected model output
def get_model_output(path, required_columns):
    if not path.exists():
        return None, "Model output file is not available yet."

    try:
        data = load_model_output(
            str(path),
            path.stat().st_mtime_ns,
        )

        missing = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing:
            return None, (
                "Model output is missing columns: "
                + ", ".join(missing)
            )

        data = data[
            (data["store_id"] == str(store_id))
            & (data["product_id"] == str(product_id))
        ].copy()

        if data.empty:
            return None, (
                "No model predictions are available "
                "for this store and product."
            )

        return data, None

    except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
        return None, f"Could not read model output: {error}"


# Verify dataset
if not DATA_PATH.exists():
    st.error(
        f"Processed dataset not found: {DATA_PATH}"
    )
    st.stop()

df = load_data()

if df.empty:
    st.warning("The processed dataset is empty.")
    st.stop()


# Store and product selection
col1, col2 = st.columns(2)

with col1:
    store_id = st.selectbox(
        "Select Store",
        sorted(df["store_id"].dropna().unique()),
    )

store_df = df[
    df["store_id"] == store_id
]

with col2:
    product_id = st.selectbox(
        "Select Product",
        sorted(
            store_df["product_id"].dropna().unique()
        ),
    )


# Filter selected product
filtered = store_df[
    store_df["product_id"] == product_id
].copy()

filtered = filtered.sort_values("date")

if filtered.empty:
    st.warning("No data available for this product.")
    st.stop()


# Product information
st.divider()
st.subheader(
    f"Store {store_id} | Product {product_id}"
)

total_days = filtered["date"].nunique()
average_demand = filtered["demand_total"].mean()
zero_demand_days = (
    filtered["demand_observed"] == 0
).sum()


# Summary metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Days of Data",
        f"{total_days:,}",
    )

with col2:
    st.metric(
        "Average Daily Demand (Scaled)",
        f"{average_demand:.2f}",
    )

with col3:
    st.metric(
        "Zero Observed-Demand Days",
        f"{zero_demand_days:,}",
    )


# Dashboard tabs
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Demand Chart",
        "7-Day Forecast",
        "Data Overrides",
        "Covariates",
    ]
)


# Tab 1: Demand chart
with tab1:
    st.subheader("Historical Daily Demand")

    chart_data = filtered[
        [
            "date",
            "demand_observed",
            "demand_total",
        ]
    ].copy()

    chart_data = chart_data.rename(
        columns={
            "demand_observed": "Observed Demand",
            "demand_total": "No-Recovery Baseline",
        }
    )

    # Load recovered demand
    recovered, recovered_error = get_model_output(
        RECOVERED_PATH,
        [
            "store_id",
            "product_id",
            "date",
            "recovered_demand",
        ],
    )

    if recovered is not None:
        recovered["date"] = pd.to_datetime(
            recovered["date"],
            errors="coerce",
        )

        recovered["recovered_demand"] = pd.to_numeric(
            recovered["recovered_demand"],
            errors="coerce",
        )

        recovered = recovered.dropna(
            subset=["date", "recovered_demand"]
        )

        recovered = recovered.drop_duplicates(
            subset=["date"],
            keep="last",
        )

        chart_data = chart_data.merge(
            recovered[["date", "recovered_demand"]],
            on="date",
            how="left",
            validate="many_to_one",
        )

        if chart_data["recovered_demand"].notna().any():
            chart_data = chart_data.rename(
                columns={
                    "recovered_demand": "Model-Recovered Demand",
                }
            )

            st.line_chart(
                chart_data,
                x="date",
                y=[
                    "Observed Demand",
                    "No-Recovery Baseline",
                    "Model-Recovered Demand",
                ],
                x_label="Date",
                y_label="Scaled Demand",
                height=400,
            )

            st.caption(
                "Model-Recovered Demand shows the "
                "imputation model's estimated total demand."
            )

        else:
            recovered_error = (
                "No recovered-demand values match "
                "the historical dates."
            )

    if recovered is None or recovered_error is not None:
        st.line_chart(
            chart_data,
            x="date",
            y=[
                "Observed Demand",
                "No-Recovery Baseline",
            ],
            x_label="Date",
            y_label="Scaled Demand",
            height=400,
        )

        st.info(
            "Recovered demand is not available yet. "
            f"{recovered_error or ''} "
            "The chart currently shows the "
            "no-recovery baseline."
        )


# Tab 2: Seven-day forecast
with tab2:
    st.subheader("7-Day Demand Forecast")

    st.caption(
        "Forecasted daily demand with 95% "
        "prediction intervals."
    )

    # Load forecast results
    forecast, forecast_error = get_model_output(
        FORECAST_PATH,
        [
            "store_id",
            "product_id",
            "forecast_date",
            "forecast",
            "lower",
            "upper",
        ],
    )

    if forecast is None:
        st.info(
            "Seven-day forecast is not available yet. "
            f"{forecast_error}"
        )

    else:
        forecast["forecast_date"] = pd.to_datetime(
            forecast["forecast_date"],
            errors="coerce",
        )

        for column in ["forecast", "lower", "upper"]:
            forecast[column] = pd.to_numeric(
                forecast[column],
                errors="coerce",
            )

        forecast = forecast.dropna(
            subset=[
                "forecast_date",
                "forecast",
                "lower",
                "upper",
            ]
        )

        forecast = forecast.sort_values(
            "forecast_date"
        )

        forecast = forecast.drop_duplicates(
            subset=["forecast_date"],
            keep="last",
        )

        # Validate prediction intervals
        valid_intervals = (
            (forecast["lower"] <= forecast["forecast"])
            & (forecast["forecast"] <= forecast["upper"])
        )

        forecast = forecast[
            valid_intervals
        ].copy()

        # Display the first seven forecast dates
        forecast = forecast.head(7)

        if forecast.empty:
            st.warning(
                "No valid forecast predictions "
                "were found for this product."
            )

        else:
            if len(forecast) != 7:
                st.warning(
                    "Only "
                    f"{len(forecast)} forecast days "
                    "are currently available."
                )

            # Forecast summary metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Average Forecasted Demand",
                    f"{forecast['forecast'].mean():.2f}",
                )

            with col2:
                st.metric(
                    "Minimum Forecasted Demand",
                    f"{forecast['forecast'].min():.2f}",
                )

            with col3:
                st.metric(
                    "Maximum Forecasted Demand",
                    f"{forecast['forecast'].max():.2f}",
                )

            # Forecast chart
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=forecast["forecast_date"],
                    y=forecast["upper"],
                    mode="lines",
                    line=dict(
                        width=0,
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=forecast["forecast_date"],
                    y=forecast["lower"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(50, 120, 220, 0.18)",
                    line=dict(
                        width=0,
                    ),
                    name="95% Prediction Interval",
                    hoverinfo="skip",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=forecast["forecast_date"],
                    y=forecast["forecast"],
                    mode="lines+markers",
                    name="Forecasted Demand",
                    line=dict(
                        color="#2563EB",
                        width=3,
                    ),
                )
            )

            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Scaled Demand",
                height=430,
                hovermode="x unified",
                margin=dict(
                    l=20,
                    r=20,
                    t=30,
                    b=20,
                ),
                legend=dict(
                    orientation="h",
                    y=1.12,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            # Forecast table
            st.subheader("Daily Forecast Details")

            forecast_table = forecast[
                [
                    "forecast_date",
                    "forecast",
                    "lower",
                    "upper",
                ]
            ].copy()

            forecast_table.columns = [
                "Date",
                "Forecasted Demand",
                "Lower Bound (95%)",
                "Upper Bound (95%)",
            ]

            st.dataframe(
                forecast_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Date": st.column_config.DateColumn(
                        "Date",
                        format="YYYY-MM-DD",
                    ),
                    "Forecasted Demand":
                        st.column_config.NumberColumn(
                            "Forecasted Demand",
                            format="%.3f",
                        ),
                    "Lower Bound (95%)":
                        st.column_config.NumberColumn(
                            "Lower Bound (95%)",
                            format="%.3f",
                        ),
                    "Upper Bound (95%)":
                        st.column_config.NumberColumn(
                            "Upper Bound (95%)",
                            format="%.3f",
                        ),
                },
            )

            st.info(
                "The shaded area represents the model's "
                "95% prediction interval. Wider intervals "
                "indicate greater uncertainty. "
                "The interval must come from the team's "
                "evaluated forecasting model."
            )


# Tab 3: Data overrides
with tab3:
    st.subheader("Manual Demand Corrections")

    st.caption(
        "Correct a daily demand value and mark the "
        "record as corrected. All changes are logged "
        "without modifying the original dataset."
    )

    # Load corrections
    try:
        overrides = load_overrides()
    except (
        OSError,
        ValueError,
        pd.errors.ParserError,
    ) as error:
        st.error(
            f"Could not load corrections: {error}"
        )
        st.stop()

    # Select date
    available_dates = (
        filtered["date"]
        .dt.date
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    selected_date = st.selectbox(
        "Select Date",
        available_dates,
        format_func=lambda date: date.strftime(
            "%B %d, %Y"
        ),
    )

    selected_row = filtered[
        filtered["date"].dt.date == selected_date
    ].iloc[0]

    original_demand = float(
        selected_row["demand_total"]
    )

    # Find existing corrections
    product_overrides = overrides[
        (overrides["store_id"] == str(store_id))
        & (overrides["product_id"] == str(product_id))
    ].copy()

    previous_corrections = product_overrides[
        product_overrides["date"]
        == selected_date.isoformat()
    ]

    # Determine correction status
    if previous_corrections.empty:
        current_demand = original_demand
        already_corrected = False

    else:
        latest_correction = (
            previous_corrections.iloc[-1]
        )

        current_demand = float(
            latest_correction["new_value"]
        )

        already_corrected = bool(
            latest_correction["corrected"]
        )

    # Display correction status
    if already_corrected:
        st.success(
            "Corrected: This record has a saved "
            "manual correction."
        )
    else:
        st.info(
            "Not corrected: No confirmed manual "
            "correction is recorded for this date."
        )

    # Display demand values
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Original Demand (Scaled)",
            f"{original_demand:.2f}",
        )

    with col2:
        st.metric(
            "Current Corrected Demand",
            f"{current_demand:.2f}",
        )

    # Correction form
    with st.form(
        f"demand_override_form_{store_id}_{product_id}_{selected_date}"
    ):
        corrected_demand = st.number_input(
            "Corrected Demand (Scaled)",
            min_value=0.0,
            value=current_demand,
            step=0.1,
            format="%.3f",
        )

        is_corrected = st.checkbox(
            "Mark this record as corrected",
            value=already_corrected,
            help=(
                "Check this box to confirm that the "
                "demand record has been manually corrected."
            ),
        )

        reason = st.text_area(
            "Reason for Correction",
            placeholder=(
                "Example: Stockout was incorrectly "
                "recorded as in stock."
            ),
        )

        save = st.form_submit_button(
            "Save Correction",
            type="primary",
        )

    # Save correction
    if save:
        if not is_corrected:
            st.warning(
                "Please check 'Mark this record "
                "as corrected' before saving."
            )

        elif not reason.strip():
            st.error(
                "Please provide a reason "
                "for the correction."
            )

        elif (
            abs(
                corrected_demand - current_demand
            ) < 1e-9
            and already_corrected
        ):
            st.warning(
                "This record is already corrected "
                "and the demand value is unchanged."
            )

        else:
            correction = {
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
                "store_id": str(store_id),
                "product_id": str(product_id),
                "date": selected_date.isoformat(),
                "old_value": current_demand,
                "new_value": corrected_demand,
                "corrected": True,
                "reason": reason.strip(),
            }

            try:
                save_override(correction)

                st.success(
                    "Correction saved successfully!"
                )

                st.rerun()

            except OSError as error:
                st.error(
                    f"Could not save correction: {error}"
                )

    # Correction history
    st.divider()
    st.subheader("Correction History")

    if product_overrides.empty:
        st.info(
            "No corrections recorded "
            "for this product."
        )

    else:
        history = product_overrides[
            [
                "date",
                "old_value",
                "new_value",
                "corrected",
                "reason",
                "timestamp",
            ]
        ].copy()

        history = history.sort_values(
            "timestamp",
            ascending=False,
        )

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
            column_config={
                "date": "Date",
                "old_value": st.column_config.NumberColumn(
                    "Previous Value",
                    format="%.3f",
                ),
                "new_value": st.column_config.NumberColumn(
                    "Corrected Value",
                    format="%.3f",
                ),
                "corrected": st.column_config.CheckboxColumn(
                    "Corrected",
                ),
                "reason": "Reason",
                "timestamp": "Saved At (UTC)",
            },
        )

    st.info(
        "Corrections are stored in planner_edits.csv. "
        "The original Parquet data remains unchanged. "
        "The forecasting pipeline must apply saved "
        "corrections in a future run; models are "
        "not automatically retrained."
    )


# Tab 4: Covariates
with tab4:
    st.subheader("Discount History")

    st.line_chart(
        filtered,
        x="date",
        y="discount",
        x_label="Date",
        y_label="Discount",
    )

    st.subheader("Temperature History")

    st.line_chart(
        filtered,
        x="date",
        y="avg_temperature",
        x_label="Date",
        y_label="Average Temperature",
    )

    st.subheader("Humidity History")

    st.line_chart(
        filtered,
        x="date",
        y="avg_humidity",
        x_label="Date",
        y_label="Average Humidity",
    )

    st.subheader(
        "Holiday and Activity Indicators"
    )

    st.line_chart(
        filtered,
        x="date",
        y=[
            "holiday_flag",
            "activity_flag",
        ],
    )
