"""Shared inventory source for the Walmart M5 Streamlit prototype.

No uploaded inventory is persisted to the repository or a database.
"""
import hashlib
import math
import pandas as pd

KEYS = ['store_id', 'item_id']


def parse_inventory_upload(file):
    df = pd.read_csv(file, dtype={'store_id': str, 'item_id': str})
    if not set(KEYS + ['on_hand']).issubset(df.columns):
        raise ValueError('CSV must contain store_id, item_id, on_hand columns.')
    df = df[KEYS + ['on_hand']].copy()
    for k in KEYS:
        if df[k].isna().any() or df[k].astype(str).str.strip().eq('').any():
            raise ValueError(f'Empty {k} values are not permitted.')
        df[k] = df[k].astype(str).str.strip()
    if df.duplicated(KEYS).any():
        raise ValueError('Duplicate store_id + item_id pairs found. Use one row per pair.')
    raw = df['on_hand'].astype('string').str.strip().replace('', pd.NA)
    value = pd.to_numeric(raw, errors='coerce')
    bad = raw.notna() & (value.isna() | ~value.ge(0) | (value % 1 != 0))
    if bad.any():
        raise ValueError('on_hand must be a nonnegative whole number or blank for missing inventory.')
    df['on_hand'] = value.astype('Int64')
    return df


def simulated_stock(item_id, demand, p90_sum):
    scenario = hashlib.sha256(str(item_id).encode('utf8')).digest()[0] % 3
    demand = max(0.0, float(demand))
    upper = max(demand, float(p90_sum))
    if scenario == 0:
        return max(0, math.floor(demand * .65))
    if scenario == 1:
        return max(0, math.ceil((demand + upper) / 2))
    return max(0, math.ceil(upper * 1.15) + 1)


def apply_inventory(items, uploaded, mode, overrides=None):
    """Returns item-store rows with starting_stock and inventory_source."""
    out = items.copy()
    if 'store_id' not in out.columns or not set(KEYS).issubset(out.columns):
        raise ValueError('Item summaries need store_id and item_id.')
    out['starting_stock'] = out.apply(
        lambda r: simulated_stock(r.item_id, r.demand, r.daily_p90_sum), axis=1
    ).astype('int64')
    out['inventory_source'] = 'Simulated'
    if mode == 'Uploaded inventory' and uploaded is not None:
        out = out.merge(uploaded, on=KEYS, how='left', validate='many_to_one')
        matched = out['on_hand'].notna()
        out.loc[matched, 'starting_stock'] = out.loc[matched, 'on_hand'].astype('int64')
        out.loc[matched, 'inventory_source'] = 'Uploaded'
        out = out.drop(columns='on_hand')
    if overrides:
        for (store, item), value in overrides.items():
            mask = out['store_id'].eq(store) & out['item_id'].eq(item)
            out.loc[mask, 'starting_stock'] = int(value)
            out.loc[mask, 'inventory_source'] = 'Manual override'
    out['starting_stock'] = out['starting_stock'].astype('int64')
    return out


def forecast_inventory_answer(question, store, item, on_hand, forecast, stockout_date, source):
    """A grounded numeric summary, not an LLM inference or purchase order."""
    gap = max(0.0, float(forecast) - int(on_hand))
    return (f'**Selected store/product:** {store} / {item}\n\n'
            f'- **7-day LightGBM forecast:** {forecast:.2f} units\n'
            f'- **Starting stock ({source}):** {int(on_hand):,} units\n'
            f'- **Forecast shortfall:** {gap:.2f} units\n'
            f'- **Rounded-up gap:** {math.ceil(gap):,} units (illustrative; not a validated order quantity)\n'
            f'- **Projected stockout:** {stockout_date}\n\n'
            'Calculated from the selected forecast and stock inputs, assuming no replenishment. '
            'No live Walmart inventory data or actual lead-time validation is available.')
