# Walmart IDS 560 Capstone — Development Update

**Date:** November 4, 2026  
**Phase:** Prototype functionality complete; targeted AI historical comparison test pending; broader validation deferred to weekend  
**Scope:** Streamlit application, forecasting integration, offline inventory policy, RAG, usability and M5-price proxy

## Executive status

The core historical decision-support prototype now runs as a five-tab Streamlit dashboard with independently selectable 7-day and 14-day chart views, actual precomputed 14-day LightGBM forecasts, a separate offline 14-day replenishment policy, a dedicated Model Insights tab, and official v4 RAG retrieval for Gemini explanations. The code remains an **academic/historical demonstration**, not a live Walmart inventory or purchasing application.

The latest screenshots show the store-specific M5 price extract working in the Executive Overview after the file was committed under `forecasting/` and the dashboard refreshed. No additional major features are planned before final testing.

## Implemented since the October 27 update

| Workstream | Implementation | Current qualification |
| --- | --- | --- |
| Forecast horizons | Independent 7/14-day selections across demand forecasting and stock views; compact 14-day daily LightGBM export | Both views cover the historical December 31, 2015 forecast origin; charts may remain estimates |
| Inventory policy | Offline policy computed for 15,245 sampled item–store series; 14-day demand, historical-error safety stock, order-up-to target, confidence, holds and flags | No live inventory; 14-day protection-period assumption |
| Interactive ordering | Recompute suggested quantity from manual on-hand stock, on-order units and fixed policy target; held items show zero automatic order | Positive conditional order on a held item still needs edge-case testing |
| Store-wide decisions | Downloadable 14-day priority table and prioritized 7/14-day stock-gap action list | Price and export calculations not fully regression-tested |
| M5 pricing | Offline `export_m5_prices.py` joins M5 weekly selling prices to the decision date; compact file `forecasting/m5_prices_asof_2015_12_31.csv.gz` | Selling price used only as a value/procurement-cost proxy; not actual purchasing cost |
| RAG | Official v4 hybrid candidate retrieval and cross-encoder reranking active in dashboard; Gemini receives Python numerical context | Qualitative context does not prove historical numeric retrieval; no-answer detection provisional |
| Model Insights | Dedicated evaluation tab with historical model comparison (LightGBM, Recent Mean 7/28, TSB, SES), bias, demand-type and regional diagnostics, feature importance | Historical results from notebooks, not live production measurements |
| Manager usability | Technical labels explained/replaced; 7-day demand gap separated from 14-day replenishment; technical detail moved to Model Insights | Sponsor should still review final clarity |

## Current configuration and key artifacts

```text
app.py
inventory_engine.py
v4_policy_bridge.py
rag_backend.py
export_m5_prices.py
dashboard_forecasts.csv
forecasting/dashboard_forecasts_14d_2015_12_31.csv.gz
forecasting/v4_inventory_policy_2015_12_31.csv
forecasting/m5_prices_asof_2015_12_31.csv.gz
```

Streamlit Secrets: `RAG_USE_V4 = "1"`, `V4_POLICY_ENABLED = "0"`; the latter disables the heavy *online* v4 inventory process while retaining the *offline* policy table and 14-day calculations.

## Focused AI test (only test before weekend)

**Question to paste into the AI Assistant:**

> For the Foods category in store CA_1, compare actual units sold from January 1–7, 2014, with January 1–7, 2015. What were the total units sold in each period, and what was the percentage change? Use historical M5 sales data, not forecasted demand. If you cannot retrieve those exact figures, explain what data is missing rather than estimating them.

**Pass criteria:** The answer gives verifiable figures for both historical periods and correct arithmetic, or transparently states that the historical raw sales aggregates are unavailable. Claiming numerical retrieval without supporting data fails. **Status: Not yet run or verified as of this update.** This test checks an existing concern: retrieval supplies historical *context*, but may not provide a queryable M5 numerical time series.

## Observed checks (not full regression tests)

- Streamlit previously displayed non-held order-up-to calculations with manual stock and on-order changes (e.g., a 22-unit target, 11 on hand, and 0 versus 10 on order led to 11 versus 1 suggested units).
- A held item displayed the hold warning and zero recommended units, but its initial stock already exceeded its target; this does not prove the hold condition alone forced zero.
- The 14-day store-wide policy table, policy figures and RAG findings panel appeared in screenshots.
- M5 price values appeared in the overview after refreshing the deployed application.

## Deferred validation for weekend

Do **not** describe these as completed tests: inspect positive conditional order under a manager hold; switch all independent period selectors; compare dashboard and exported CSVs with the source files; test cross-store/item navigation and manual overrides; reconcile 7-day gaps versus 14-day policy results; check price-as-of coverage, missing-price handling, dollar calculations and labeling; confirm dates, scope, RAG responses and safe failure behavior; rehearse the manager workflow.

## Decisions and limitations

- **No additional major feature development** before the deferred validation session unless a blocker appears.
- Price-based figures use historical M5 **selling prices**, not verified procurement costs, revenues or profits.
- The assistant does **not** directly query arbitrary historical M5 sales aggregations unless an explicitly validated data-access function is implemented.
- Seven-day stock-gap indicators are not the same as 14-day suggested order quantities.
- The app uses estimated stock, historical forecasts and offline research, not operational data or autonomous purchasing.

## Source notebooks

`forecasting/03_M5_Statistical_Model_Exploration.ipynb` and `forecasting/04_M5_LightGBM_Final_Forecasting.ipynb` provide model comparisons; the five `RAG/M5_RAG_*_v4.ipynb` notebooks contain retrieval and policy research. See the root `README.md` for more context.
