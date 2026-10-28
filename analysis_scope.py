"""Grounded, read-only multi-level summaries for the Walmart M5 assistant."""
import math
import re
import pandas as pd
from inventory_bridge import apply_inventory

SCOPES = ["Automatic (from question)", "Selected product", "Selected department", "Selected store", "All stores"]

def choose_scope(choice, question):
    """Conservative rule-based routing; store-to-store questions need all stores."""
    if choice != SCOPES[0]:
        return choice
    q = question.lower().strip()
    # Questions about differences, rankings, or multiple stores must include all stores.
    if re.search(r"\b(all stores|across (?:all )?stores|compare stores|stores compared|store comparison|between stores|which stores|what stores|top \d+ stores|highest[- ](?:shortage|demand).*stores|stores with|store rankings?|rank(?:ing)? stores|overall|companywide|entire dataset|all locations|all products|across locations|each store|every store)\b", q):
        return "All stores"
    if re.search(r"\bstores\b", q):
        return "All stores"  # plural implies more than one store
    if re.search(r"\b(across|compare|rank|ranking|top|highest|lowest|which)\b.*\b(departments|categories)\b", q):
        return "All stores"  # broad department rankings are not one selected department
    if re.search(r"\b(department|category)\b", q):
        return "Selected department"
    if re.search(r"\b(store|location)\b", q):
        return "Selected store"
    return "Selected product"

def scope_summary(forecasts, store, dept, item, scope, uploaded, mode, overrides):
    """Aggregate distinct item-store forecast rows, without using external RAG text."""
    scope_df = forecasts
    if scope == "Selected product":
        scope_df = scope_df[(scope_df.store_id == store) & (scope_df.item_id == item)]
    elif scope == "Selected department":
        scope_df = scope_df[(scope_df.store_id == store) & (scope_df.dept_id == dept)]
    elif scope == "Selected store":
        scope_df = scope_df[scope_df.store_id == store]
    source = scope_df.groupby(['store_id', 'dept_id', 'item_id'], as_index=False).agg(
        demand=('forecast', 'sum'), daily_p90_sum=('p90', 'sum'))
    source['demand'] = source['demand'].fillna(0).clip(lower=0)
    source['daily_p90_sum'] = source['daily_p90_sum'].fillna(0).clip(lower=0)
    source = apply_inventory(source, uploaded, mode, overrides)
    source['gap'] = (source.demand - source.starting_stock).clip(lower=0)
    source['order_gap'] = source.gap.map(math.ceil)
    by_store = source.groupby('store_id', as_index=False).agg(
        forecast=('demand','sum'), stock=('starting_stock','sum'),
        gap=('gap','sum'), units_to_cover=('order_gap','sum'), items=('item_id','count'))
    by_store = by_store.sort_values('gap', ascending=False)
    by_dept = source.groupby('dept_id', as_index=False).agg(
        forecast=('demand','sum'), gap=('gap','sum'), items=('item_id','count'))
    by_dept = by_dept.sort_values('forecast', ascending=False)
    top = source.sort_values('gap', ascending=False).head(8)
    label = {'Selected product': f'{store}/{item}', 'Selected department':f'{store}/{dept}',
             'Selected store':store, 'All stores':'All forecasted stores'}[scope]
    lines = [f'Analysis scope: {scope} ({label})',
             f'Included item-store series: {len(source):,}',
             f'7-day predicted demand: {source.demand.sum():,.2f} units',
             f'Starting stock across included series: {source.starting_stock.sum():,} units',
             f'Sum of item-level positive forecast gaps: {source.gap.sum():,.2f} units',
             f'Sum of rounded-up item-level gaps (illustrative): {source.order_gap.sum():,} units',
             f'Items with a positive gap: {(source.gap > 0).sum():,}',
             'Stock origins: '+', '.join(f'{k}: {v}' for k,v in source.inventory_source.value_counts().items()),
             'Largest store-level shortages (forecast gap units):']
    lines += [f'- {r.store_id}: forecast {r.forecast:,.2f}, gap {r.gap:,.2f}, items {r.items:,}' for r in by_store.head(10).itertuples()]
    lines += ['Top departments by forecast:']
    lines += [f'- {r.dept_id}: forecast {r.forecast:,.2f}, gap {r.gap:,.2f}' for r in by_dept.head(8).itertuples()]
    lines += ['Largest individual shortages:']
    lines += [f'- {r.store_id}/{r.item_id}: 7-day forecast {r.demand:.2f}, stock {r.starting_stock}, gap {r.gap:.2f}' for r in top.itertuples()]
    lines += ['Scope is limited to the included M5 forecasting sample, NOT all Walmart products.',
              'These are forecast-based calculations, not validated purchase orders or actual stockouts.',
              'Sum of positive item-level gaps is NOT the same as overall aggregate demand minus aggregate stock.']
    return '\n'.join(lines), source
