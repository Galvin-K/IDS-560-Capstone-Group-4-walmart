"""Produce a compact, historical v4 inventory decision table outside Streamlit.

Run from the repository root: python precompute_v4_inventory.py --limit 5
Then: python precompute_v4_inventory.py

This script uses the team's own rag_common.py and its M5/forecast inputs.
It does NOT use the RAG evidence library or LLM and does NOT assume that
M5 has observed on-hand stock. The resulting table is a planning scenario
at a historical decision date, not an operational purchase-order feed.
"""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIELDS = [
    'as_of','item_id','store_id','series_id','model','model_days',
    'demand_14d','p10_sum_14d','p90_sum_14d','safety_stock',
    'safety_stock_method','order_up_to','new_item_if_confirmed_target',
    'confidence','confidence_reasons','hold','hold_reasons','flags',
    'status'
]


def apply_stock_override(row, on_hand: float, on_order: float = 0) -> dict:
    """Apply user stock values to saved policy targets without loading M5.

    Explicitly held items remain order_qty=0; show a conditional quantity only.
    """
    if not (math.isfinite(float(on_hand)) and math.isfinite(float(on_order))):
        raise ValueError('Stock and on-order inputs must be finite')
    if on_hand < 0 or on_order < 0:
        raise ValueError('Stock and on-order inputs cannot be negative')
    target = float(row['order_up_to'])
    if not math.isfinite(target):
        raise ValueError('Missing valid order-up-to target')
    hold = str(row['hold']).strip().lower() in ('true', '1', 'yes')
    recommended = max(0, math.ceil(target - on_hand - on_order))
    conditional_target = row.get('new_item_if_confirmed_target', '')
    if conditional_target is not None and str(conditional_target).strip() not in ('','nan'):
        conditional = max(0, math.ceil(float(conditional_target) - on_hand - on_order))
    else:
        conditional = recommended
    return {'order_qty': 0 if hold else recommended,
            'hold': hold,
            'if_confirmed_order_qty': conditional if hold else None}


def collect_row(rc, sid: str, as_of: str) -> dict:
    item, store = rc.split_series_id(sid)
    fc = rc.baseline_forecast(item, store, as_of, horizon=14, quiet=True)
    # Policy computation uses zero stock only to determine stock-independent
    # target. End users provide estimated or manually overridden stock later.
    res = rc.place_order(item, store, as_of, on_hand=0, on_order=0)
    order = res['order']
    flag_rows = res.get('flags', [])
    conf = res.get('confidence', {})
    new_item_target = ''
    if 'if_confirmed_order_qty' in order and any(f.get('code') == 'NEW_ITEM' for f in flag_rows):
        new_item_target = order['if_confirmed_order_qty']
    means = np.asarray(fc['mean'], dtype=float)[:14]
    p10 = np.asarray(fc['p10'], dtype=float)[:14]
    p90 = np.asarray(fc['p90'], dtype=float)[:14]
    if len(means) != 14 or not np.isfinite(means).all():
        raise ValueError('Invalid 14-day forecast')
    if not str(fc.get('model', '')).startswith('LightGBM'):
        raise ValueError('Not a LightGBM forecast at this date; refusing silent Recent Mean substitution')
    return {
        'as_of': as_of, 'item_id': item, 'store_id': store, 'series_id': sid,
        'model': fc.get('model',''), 'model_days': fc.get('model_days', 14),
        'demand_14d': round(float(means.sum()),4),
        'p10_sum_14d': round(float(p10.sum()),4),
        'p90_sum_14d': round(float(p90.sum()),4),
        'safety_stock': order['safety_stock'],
        'safety_stock_method': order.get('safety_stock_method',''),
        'order_up_to': order['order_up_to'],
        'new_item_if_confirmed_target': new_item_target,
        'confidence': conf.get('level','Not evaluated'),
        'confidence_reasons': ' | '.join(map(str,conf.get('reasons',[]))),
        'hold': bool(order.get('hold',False)),
        'hold_reasons': ' | '.join(order.get('hold_reasons',[])),
        'flags': ' | '.join(str(f.get('code','')) for f in flag_rows),
        'status': 'historical_plan_estimated_inventory_required'
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--as-of', default='2015-12-31', help='Decision date; matches dashboard Jan 1-7 2016 forecast')
    parser.add_argument('--limit',type=int,default=None,help='First N item-store series; use 5 for a smoke test')
    parser.add_argument('--output',default='forecasting/v4_inventory_policy_2015_12_31.csv')
    args=parser.parse_args()
    import rag_common as rc
    rc.setup(m5=True,forecasts=True,library=False)
    if rc.CUBE is None:
        raise SystemExit('Missing LightGBM forecast cube')
    day=pd.Timestamp(args.as_of)
    if not (rc.CUBE['origins'] == day).any():
        raise SystemExit(f'No exact LightGBM forecast origin for {args.as_of}; choose a weekly origin to avoid forecast extension')
    ids=list(rc.CUBE['ids'])
    if args.limit is not None:
        ids=ids[:args.limit]
    out=ROOT / args.output
    out.parent.mkdir(parents=True,exist_ok=True)
    ok=0; errors=[]
    with out.open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=FIELDS)
        writer.writeheader()
        for i,sid in enumerate(ids,1):
            try:
                writer.writerow(collect_row(rc,str(sid),args.as_of))
                ok+=1
            except Exception as exc:
                errors.append((str(sid),str(exc)))
            if i%1000==0:
                print(f'Processed {i:,}/{len(ids):,}: {ok:,} complete; {len(errors):,} skipped',flush=True)
    print(f'Output: {out} | rows: {ok:,} | errors: {len(errors):,}')
    if errors:
        print('First failures:',errors[:5]);raise SystemExit(2)


if __name__=='__main__':
    main()
