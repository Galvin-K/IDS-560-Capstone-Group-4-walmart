"""Optional adapter to the team's *actual* v4 ordering implementation.

The five supplied notebooks run:
  rc.setup(m5=True, forecasts=True, library=True)
  fc = baseline_forecast(item, store, as_of)
  recommend(item, store, as_of, on_hand=..., forecast=fc, call_llm=False)

Do not run this automatically on each Streamlit rerun. It can load historical
M5 arrays, a 14-day forecast cube and the v4 library, and may exceed free-tier
RAM. Enable only after all required data is deployed and resource-tested.
"""
import os
from pathlib import Path
from functools import lru_cache
import numpy as np

ROOT = Path(__file__).resolve().parent


def policy_status():
    if os.getenv('V4_POLICY_ENABLED', '0') != '1':
        return False, 'Not enabled. Enable only after installing the shared v4 code and real 14-day forecast artifacts.'
    if not (ROOT / 'rag_common.py').exists() and not (ROOT / 'RAG' / 'rag_common.py').exists():
        return False, 'rag_common.py has not been provided.'
    return True, 'v4 policy adapter is enabled; required forecast files will be checked when executed.'


@lru_cache(maxsize=1)
def _load_v4_policy():
    import importlib
    import sys
    for path in (str(ROOT), str(ROOT / 'RAG')):
        if path not in sys.path:
            sys.path.insert(0, path)
    rc = importlib.import_module('rag_common')
    rc.setup(m5=True, forecasts=True, library=True)
    for name in ('baseline_forecast', 'recommend'):
        if not callable(getattr(rc, name, None)):
            raise RuntimeError(f'rag_common.{name} unavailable')
    return rc


def v4_recommendation(item, store, decision_date, on_hand):
    """Return validated-shape outputs, not a fabricated 14-day prediction."""
    enabled, reason = policy_status()
    if not enabled:
        raise RuntimeError(reason)
    rc = _load_v4_policy()
    forecast = rc.baseline_forecast(item, store, str(decision_date))
    daily = np.asarray(forecast.get('mean', []), dtype=float)
    if len(daily) < 14 or not np.isfinite(daily[:14]).all():
        raise RuntimeError('Exact 14-day forecast unavailable for this product/date')
    outcome = rc.recommend(item, store, str(decision_date), on_hand=int(on_hand),
                           forecast=forecast, call_llm=False)
    if not isinstance(outcome, dict) or not isinstance(outcome.get('order'), dict):
        raise TypeError('Unexpected response from official v4 policy')
    return {
        'forecast_14_days': float(daily[:14].sum()),
        'forecast_source': str(forecast.get('model', 'v4 policy forecast')),
        'order': outcome['order'],
        'flags': outcome.get('flags', []),
        'confidence': outcome.get('confidence', {}),
        'segment': outcome.get('segment', {}),
    }


# Lightweight OFFLINE v4 policy table. This path does not initialize M5 history,
# the large forecast cube, an LLM, or the retrieval cross-encoder.
OFFLINE_POLICY_FILE = ROOT / "forecasting" / "v4_inventory_policy_2015_12_31.csv"
OFFLINE_DECISION_DATE = "2015-12-31"


def offline_policy_status():
    """Return (ready, explanation) independently of V4_POLICY_ENABLED."""
    if not OFFLINE_POLICY_FILE.is_file():
        return False, f"Precomputed v4 policy file missing: {OFFLINE_POLICY_FILE.name}"
    return True, "Historical v4 inventory policy table available (decision date 2015-12-31)."


@lru_cache(maxsize=1)
def load_offline_policy():
    """Load and verify the fixed-date, item-store-level v4 output once."""
    import pandas as pd
    from inventory_engine import apply_precomputed_policy
    ok, message = offline_policy_status()
    if not ok:
        raise FileNotFoundError(message)
    data = pd.read_csv(OFFLINE_POLICY_FILE, dtype={"item_id": str, "store_id": str, "series_id": str, "as_of": str})
    required = {
        "as_of", "item_id", "store_id", "demand_14d", "safety_stock",
        "order_up_to", "confidence", "hold", "hold_reasons", "flags",
        "new_item_if_confirmed_target", "model", "model_days",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Precomputed v4 policy missing columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("Precomputed v4 policy CSV is empty")
    if not data["as_of"].eq(OFFLINE_DECISION_DATE).all():
        raise ValueError("The precomputed policy date does not match the 2015-12-31 decision date")
    if data.duplicated(["item_id", "store_id"]).any():
        raise ValueError("Duplicate item-store records in v4 policy output")
    if not data["model"].fillna("").str.startswith("LightGBM").all():
        raise ValueError("Some precomputed rows do not use LightGBM")
    import numpy as np
    for col in ("demand_14d", "safety_stock", "order_up_to", "model_days"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
        if data[col].isna().any() or not np.isfinite(data[col]).all():
            raise ValueError(f"Invalid or missing numeric values in {col}")
    if (data[["demand_14d", "safety_stock", "order_up_to"]] < 0).any().any():
        raise ValueError("Negative targets or demands in v4 policy output")
    if not data["model_days"].between(14,14).all():
        raise ValueError("The CSV does not contain 14 actual forecast days for every series")
    # Fail early if a representative row cannot be processed by the stock helper.
    apply_precomputed_policy(data.iloc[0], on_hand=0)
    return data.set_index(["store_id", "item_id"], drop=False)


def offline_v4_recommendation(item, store, on_hand, on_order=0):
    """Get one stock-dependent decision from a cached historic policy row."""
    from inventory_engine import apply_precomputed_policy
    data = load_offline_policy()
    key = (str(store), str(item))
    if key not in data.index:
        raise KeyError(f"No precomputed 14-day policy for {item} at {store}")
    row = data.loc[key]
    if hasattr(row, "ndim") and row.ndim != 1:
        raise ValueError("Unexpected duplicate policy rows")
    result = apply_precomputed_policy(row, on_hand=on_hand, on_order=on_order)
    return {**row.to_dict(), **result}
