"""Inventory planning calculations, independent of Streamlit.

No M5 on-hand or supplier lead-time data are assumed to be observed.
Exact 14-day forecasts and historical forecast errors must be supplied to
reproduce a calibrated protection-period ordering policy.
"""
from dataclasses import dataclass
from math import ceil, isfinite
from typing import Optional, Sequence
import numpy as np


@dataclass(frozen=True)
class OrderPlan:
    expected_demand: float
    safety_stock: Optional[float]
    target_stock: Optional[float]
    inventory_position: float
    computed_order_qty: Optional[int]
    recommended_order_qty: int
    hold: bool
    flags: tuple[str, ...]
    confidence: str
    forecast_source: str
    safety_method: str


def protection_period_order(
    forecast_daily: Sequence[float],
    on_hand: float,
    *,
    on_order: float = 0,
    backorders: float = 0,
    lead_days: int = 7,
    review_days: int = 7,
    historical_positive_errors: Optional[Sequence[float]] = None,
    manual_safety_stock: Optional[float] = None,
    hold_flags: Sequence[str] = (),
    confidence: str = "Not evaluated",
    forecast_source: str = "Provided forecast",
) -> OrderPlan:
    """Compute an order-up-to quantity for lead + review days.

    `historical_positive_errors`: historic protection-period UNDERFORECAST
    amounts (actual demand - forecast demand), measured for the same horizon.
    The positive 95th percentile is an empirical safety stock estimate.
    When no historic errors or explicit buffer exist, *do not* invent a safety
    stock or definitive order quantity.

    Operational hold flags are passed from independently validated checks; this
    module never infers NEW_ITEM / POSSIBLE_STOCKOUT / DISCONTINUED from the
    seven-day forecast alone.
    """
    if lead_days < 0 or review_days < 1:
        raise ValueError("Lead days must be nonnegative and review days positive")
    horizon = lead_days + review_days
    values = np.asarray(forecast_daily, dtype=float).reshape(-1)
    if len(values) < horizon:
        raise ValueError(f"Need {horizon} daily forecasts; received {len(values)}")
    if not np.isfinite(values[:horizon]).all() or (values[:horizon] < 0).any():
        raise ValueError("Daily forecasts must be finite and nonnegative")
    if any(not isfinite(float(v)) or float(v) < 0 for v in (on_hand, on_order, backorders)):
        raise ValueError("Inventory quantities must be finite and nonnegative")

    stock = None
    method = "Unavailable: historical errors or an explicit planning buffer required"
    if historical_positive_errors is not None:
        errors = np.asarray(historical_positive_errors, dtype=float).reshape(-1)
        if len(errors) and np.isfinite(errors).all():
            stock = max(0.0, float(np.quantile(errors, 0.95)))
            method = "Empirical 95th percentile of historical protection-period underforecast errors"
    if manual_safety_stock is not None:
        if not isfinite(float(manual_safety_stock)) or float(manual_safety_stock) < 0:
            raise ValueError("Manual safety stock must be finite and nonnegative")
        stock = float(manual_safety_stock)
        method = "User-entered planning buffer (not calibrated from historical errors)"

    demand = float(values[:horizon].sum())
    position = float(on_hand + on_order - backorders)
    target = (demand + stock) if stock is not None else None
    computed = max(0, ceil(target - position)) if target is not None else None
    holds = tuple(dict.fromkeys(str(f) for f in hold_flags if str(f).strip()))
    # A rule is provisional when the safety-stock input is unavailable.
    all_flags = holds + (("SAFETY_STOCK_UNAVAILABLE",) if stock is None else ())
    hold = bool(holds) or stock is None
    return OrderPlan(
        expected_demand=demand,
        safety_stock=stock,
        target_stock=target,
        inventory_position=position,
        computed_order_qty=computed,
        recommended_order_qty=0 if hold else int(computed),
        hold=hold,
        flags=all_flags,
        confidence=confidence,
        forecast_source=forecast_source,
        safety_method=method,
    )


def repeat_week_scenario(seven_day_forecast: Sequence[float]) -> np.ndarray:
    """Explicit, unvalidated 14-day extrapolation for UI demonstrations only."""
    values = np.asarray(seven_day_forecast, dtype=float).reshape(-1)
    if len(values) != 7 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Exactly seven finite nonnegative forecast values required")
    return np.tile(values, 2)


def apply_precomputed_policy(row, on_hand: float, on_order: float = 0) -> dict:
    """Recalculate order quantities from stock-independent v4 offline targets.

    The original policy was evaluated at a fixed decision date; changing stock
    affects only inventory position and order quantity, not forecasts/flags.
    Held items remain on hold, regardless of starting inventory.
    """
    from math import ceil, isfinite
    if any(not isfinite(float(x)) or float(x) < 0 for x in (on_hand, on_order)):
        raise ValueError("On-hand and on-order quantities must be nonnegative and finite")
    target = float(row["order_up_to"])
    if not isfinite(target) or target < 0:
        raise ValueError("Missing or invalid precomputed order-up-to target")
    raw_hold = row["hold"]
    if isinstance(raw_hold, bool):
        held = raw_hold
    else:
        held = str(raw_hold).strip().lower() in {"true", "1", "yes"}
    computed = max(0, ceil(target - float(on_hand) - float(on_order)))
    alt_target = row.get("new_item_if_confirmed_target", None)
    if alt_target is not None and str(alt_target).strip().lower() not in {"", "nan", "none", "<na>"}:
        alternative = float(alt_target)
        if not isfinite(alternative) or alternative < 0:
            raise ValueError("Invalid new-item conditional target")
        conditional = max(0, ceil(alternative - float(on_hand) - float(on_order)))
    else:
        conditional = computed
    return {
        "on_hand": float(on_hand),
        "on_order": float(on_order),
        "inventory_position": float(on_hand) + float(on_order),
        "computed_order_qty": computed,
        "order_qty": 0 if held else computed,
        "hold": held,
        "if_confirmed_order_qty": conditional if held else None,
    }
