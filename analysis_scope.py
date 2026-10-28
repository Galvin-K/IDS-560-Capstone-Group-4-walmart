"""Question-driven, read-only forecast scope for the Walmart M5 assistant.

The assistant never inherits filters from the other dashboard tabs.
"""
import math
import re
import pandas as pd
from inventory_bridge import apply_inventory

SCOPES = ["Automatic (from question)", "Selected product", "Selected department", "Selected store", "All stores"]
STATE_WORDS = {'california':'CA','texas':'TX','wisconsin':'WI'}
CATEGORY_WORDS = {'foods':'FOODS','food':'FOODS','groceries':'FOODS',
                  'hobbies':'HOBBIES','hobby':'HOBBIES',
                  'household':'HOUSEHOLD'}

def resolve_question(question, forecasts, selection="Automatic (from question)"):
    """Return requested store/department/product filters; None means unrestricted.

    Explicit IDs in a question always take priority over page sidebar defaults.
    """
    q = str(question).lower()
    available_stores = set(forecasts['store_id'].astype(str).unique())
    available_depts = set(forecasts['dept_id'].astype(str).unique())
    available_items = set(forecasts['item_id'].astype(str).unique())
    stores = sorted({m.upper() for m in re.findall(r'\b(?:CA|TX|WI)_\d+\b', q, re.I)} & available_stores)
    items = sorted({m.upper() for m in re.findall(r'\b(?:FOODS|HOBBIES|HOUSEHOLD)_\d+_\d+\b', q, re.I)} & available_items)
    depts = sorted({m.upper() for m in re.findall(r'\b(?:FOODS|HOBBIES|HOUSEHOLD)_\d+\b(?!_\d)', q, re.I)} & available_depts)
    states = {v for word,v in STATE_WORDS.items() if re.search(r'\b'+word+r'\b', q)}
    # State abbreviations are recognized when written as stand-alone uppercase tokens.
    states |= {s for s in ('CA','TX','WI') if re.search(r'\b'+s+r'\b', str(question))}
    if not stores and states:
        stores = sorted(s for s in available_stores if s.split('_')[0] in states)
    cats = {v for word,v in CATEGORY_WORDS.items() if re.search(r'\b'+word+r'\b', q)}
    if not depts and not items and cats:
        depts = sorted(d for d in available_depts if d.split('_')[0] in cats)
    # A manual choice determines dimension, never a hidden sidebar filter.
    # If no identifier was supplied, use all available records rather than guessing.
    if selection == 'Selected store' and not stores:
        hint = 'No store ID was given; showing all stores.'
    elif selection == 'Selected department' and not depts:
        hint = 'No department ID was given; showing all departments.'
    elif selection == 'Selected product' and not items:
        hint = 'No product ID was given; showing all products.'
    else:
        hint = ''
    return {'stores':stores, 'departments':depts, 'items':items, 'hint':hint}

def choose_scope(choice, question):
    """Keep existing API compatible; actual filtering uses resolve_question."""
    if choice != SCOPES[0]:
        return choice
    return 'Question-based scope'

def scope_summary(forecasts, store=None, dept=None, item=None, scope=None,
                  uploaded=None, mode='Simulated inventory', overrides=None,
                  missing_policy='Use simulated inventory', question='', filters=None):
    """Aggregate exactly the records requested in the question, not sidebar defaults."""
    criteria = filters if filters is not None else resolve_question(question, forecasts, scope or SCOPES[0])
    subset = forecasts
    if criteria['stores']:
        subset = subset[subset['store_id'].isin(criteria['stores'])]
    if criteria['departments']:
        subset = subset[subset['dept_id'].isin(criteria['departments'])]
    if criteria['items']:
        subset = subset[subset['item_id'].isin(criteria['items'])]
    if subset.empty:
        raise ValueError('No forecast records match the requested combination of stores and products.')
    source = subset.groupby(['store_id','dept_id','item_id'],as_index=False).agg(
        demand=('forecast','sum'),daily_p90_sum=('p90','sum'))
    source['demand'] = source['demand'].fillna(0).clip(lower=0)
    source['daily_p90_sum'] = source['daily_p90_sum'].fillna(0).clip(lower=0)
    source = apply_inventory(source, uploaded, mode, overrides or {}, missing_policy=missing_policy)
    source['gap'] = (source.demand - source.starting_stock).clip(lower=0)
    source['order_gap'] = source.gap.map(math.ceil)
    by_store = source.groupby('store_id',as_index=False).agg(
        forecast=('demand','sum'),gap=('gap','sum'),
        total_item_series=('item_id','count'),
        item_series_with_shortage=('gap',lambda x:int((x>0).sum())))
    by_store = by_store.sort_values('gap',ascending=False)
    by_dept = source.groupby('dept_id',as_index=False).agg(
        forecast=('demand','sum'),gap=('gap','sum'),items=('item_id','count'))
    by_dept = by_dept.sort_values('gap',ascending=False)
    names = [', '.join(criteria[k]) if criteria[k] else 'all' for k in ('stores','departments','items')]
    label = f'Stores: {names[0]} | Departments: {names[1]} | Products: {names[2]}'
    lines = [f'Analysis scope: {label}', f'Included item-store series: {len(source):,}',
             f'7-day predicted demand: {source.demand.sum():,.2f} units',
             f'Starting stock across included series: {source.starting_stock.sum():,.0f} units',
             f'Sum of item-level positive forecast gaps: {source.gap.sum():,.2f} units',
             f'Sum of rounded-up item-level gaps (illustrative): {source.order_gap.sum():,.0f} units',
             f'Item-store series with a positive gap: {(source.gap>0).sum():,}',
             'Stock origins: '+', '.join(f'{k}: {v}' for k,v in source.inventory_source.value_counts().items()),
             'Store-level comparisons:']
    lines += [f'- {r.store_id}: forecast {r.forecast:,.2f}, gap {r.gap:,.2f}, '
              f'total_item_series {r.total_item_series}, item_series_with_shortage {r.item_series_with_shortage}'
              for r in by_store.itertuples()]
    lines += ['Department comparisons:']
    lines += [f'- {r.dept_id}: forecast {r.forecast:,.2f}, gap {r.gap:,.2f}, item series {r.items}'
              for r in by_dept.itertuples()]
    lines += ['Largest individual positive gaps:']
    lines += [f'- {r.store_id}/{r.item_id}: demand {r.demand:.2f}, stock {r.starting_stock}, gap {r.gap:.2f}'
              for r in source.sort_values('gap',ascending=False).head(8).itertuples()]
    lines += ['These are historical M5 forecasts on the included sample, not live Walmart inventory.',
              'Forecast gaps and order quantities are illustrative and not validated purchasing decisions.']
    return '\n'.join(lines), source
