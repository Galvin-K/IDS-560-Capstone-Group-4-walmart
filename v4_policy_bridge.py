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
