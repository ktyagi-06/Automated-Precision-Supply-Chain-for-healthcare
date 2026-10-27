"""
Recursive multi-day forecasting, batched across products.

Each step predicts one day for *every* product in a single model call, then
feeds the predictions back in as lag features for the next day. Serving and
the training-time backtest both use this function, so the accuracy reported
in training is the accuracy of the forecasts the API produces.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from features import CAT_COLS, inference_row


def recursive_forecast(model, encoder, feature_names: list[str],
                       series: dict[str, np.ndarray], last_date,
                       meta: dict[str, tuple[str, float]], horizon: int) -> dict[str, np.ndarray]:
    """Forecast `horizon` days after `last_date`.

    series : product_id -> daily units (oldest -> newest, ending on last_date)
    meta   : product_id -> (category, unit_cost)
    Returns product_id -> array of `horizon` non-negative predictions.
    """
    pids = list(series)
    # inference only needs the last MIN_HISTORY..few weeks; keep a short tail
    hist = {pid: list(map(float, series[pid][-60:])) for pid in pids}

    cat_df = pd.DataFrame({"product_id": pids, "category": [meta[p][0] for p in pids]})
    cat_encoded = encoder.transform(cat_df[CAT_COLS])
    cat_names = encoder.get_feature_names_out(CAT_COLS)
    cat_frame = pd.DataFrame(cat_encoded, columns=cat_names)

    out = {pid: np.zeros(horizon) for pid in pids}
    last_date = pd.Timestamp(last_date)
    for step in range(horizon):
        target = last_date + timedelta(days=step + 1)
        rows = [inference_row(np.array(hist[pid]), target, meta[pid][1]) for pid in pids]
        X = pd.concat([pd.DataFrame(rows), cat_frame], axis=1)[feature_names]
        preds = np.clip(model.predict(X), 0, None)
        for pid, p in zip(pids, preds):
            out[pid][step] = p
            hist[pid].append(float(p))
    return out
