"""Test helpers shared by the data-stage tests."""

from datetime import date

import numpy as np
import pandas as pd


def fake_imputed_hourly(
    cube, start: date, end: date, value: float = 0.5, masks: pd.DataFrame | None = None
):
    """An imputed_hourly-shaped table that fills every real stockout hour from start to end with `value`.

    Masked evaluation rows (non-null actual_demand) are appended when `masks` is given.
    """
    a, b = cube.day_index(start), cube.day_index(end) + 1
    series, day, hour = np.nonzero(cube.stockout[:, a:b])
    stamps = cube.dates[a + day].astype("datetime64[h]") + hour.astype("timedelta64[h]")
    rows = pd.DataFrame(
        {
            "store_id": cube.series["store_id"].to_numpy()[series],
            "product_id": cube.series["product_id"].to_numpy()[series],
            "datetime": stamps.astype("datetime64[s]"),
            "estimated_demand": value,
            "actual_demand": np.nan,
        }
    )
    if masks is not None:
        masked = masks[["store_id", "product_id", "datetime"]].assign(
            estimated_demand=99.0, actual_demand=1.0
        )
        rows = pd.concat([rows, masked], ignore_index=True)
    return rows
