"""
Model store: loads the trained demand model and produces recursive multi-day
forecasts (batched across products). The model can be hot-swapped after a
retrain without restarting the API.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import joblib
import numpy as np
import sklearn
import xgboost

from forecast_core import recursive_forecast


class ModelIncompatible(RuntimeError):
    """Saved artifacts are missing, unreadable or built with different library versions."""


class ModelStore:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self._lock = threading.RLock()
        self.model = None
        self.encoder = None
        self.metadata: dict = {}

    # ----------------------------------------------------------------- state
    @property
    def version(self) -> str:
        return self.metadata.get("model_version", "none")

    @property
    def loaded(self) -> bool:
        return self.model is not None

    # --------------------------------------------------------------- loading
    def load(self) -> None:
        """Load artifacts from disk. Raises ModelIncompatible if they can't be trusted."""
        try:
            meta = json.loads((self.model_dir / "model_metadata.json").read_text())
            versions = meta.get("library_versions") or {}
            for lib, current in (("scikit-learn", sklearn.__version__), ("xgboost", xgboost.__version__)):
                saved = versions.get(lib)
                # pickles are only safe within the same major.minor release
                if saved is None or saved.split(".")[:2] != current.split(".")[:2]:
                    raise ModelIncompatible(f"model was built with {lib} {saved}, running {current}")
            model = joblib.load(self.model_dir / "demand_model.joblib")
            encoder = joblib.load(self.model_dir / "category_encoder.joblib")
        except ModelIncompatible:
            raise
        except Exception as e:  # missing files, corrupt pickle, ...
            raise ModelIncompatible(f"cannot load model artifacts: {e}") from e
        with self._lock:
            self.model, self.encoder, self.metadata = model, encoder, meta

    def swap(self, result) -> None:
        """Install a freshly trained `TrainResult`."""
        with self._lock:
            self.model, self.encoder, self.metadata = result.model, result.encoder, result.metadata

    # ------------------------------------------------------------ forecasting
    def forecast(self, series: dict[str, np.ndarray], last_date, meta: dict, horizon: int) -> dict[str, np.ndarray]:
        with self._lock:
            model, encoder, md = self.model, self.encoder, self.metadata
        if model is None:
            raise ModelIncompatible("no model loaded")
        return recursive_forecast(model, encoder, md["all_feature_names"], series, last_date, meta, horizon)
