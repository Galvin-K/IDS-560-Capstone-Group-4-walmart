# Walmart M5 | Demand & Inventory Planner

**UIC IDS 560 Capstone — Group 4**  
**Project status:** Functional academic prototype; sponsor and faculty feedback pending  
**Updated:** October 27, 2026

**Live dashboard:** https://ids-560-capstone-group-4-walmart.streamlit.app/

## Project overview

This project explores how historical demand forecasts, inventory inputs, and an AI assistant can support retail inventory planning. It uses the public Walmart M5 competition dataset for historical forecasting demonstrations. However, we do not have access to Walmart's internal data and the dataset **is not** connected to Walmart's current inventory, purchasing, or operational systems.

The dashboard combines precomputed LightGBM item-store-day forecasts, illustrative inventory calculations, historical evidence retrieval (RAG), and Gemini-generated explanations. Calculations are performed in Python; Gemini interprets the supplied context rather than generating the forecast or executing orders.

## Dashboard pages

| Page | What it provides |
| --- | --- |
| **Executive Overview** | Selected-store demand summary, inventory action KPIs, simulation assumptions, comparison charts, and a ranked **Top Priorities** table. The table offers 5, 10, 20, or 50 rows and includes estimated order cost. |
| **Demand Forecasting** | Department-level forecast table and chart, page-specific department/product selection, product KPIs, and daily forecasts with illustrative P10/P90 ranges. |
| **Inventory Planning** | Prioritized action and illustrative order lists, page-specific product selection and stock overrides, assumptions, projected inventory KPIs, product stock trajectories, user-selected multi-product stock runway, and shortage visualizations. |
| **AI Assistant** | Question-based analysis scope (selected item, department, store, or all available stores/products), calculated forecast/inventory context, relevant historical evidence, and Gemini explanations. |

The AI assistant can compare stores and departments using calculated quantities from the dashboard's included forecasting sample. It also retrieves dated historical evidence; these documents provide contextual information, **not** the source of numerical inventory calculations.

## Data and modeling

- **Data:** Public M5 Forecasting Accuracy/Uncertainty data (historical sales, product/store hierarchy, calendar and historical selling-price inputs in the original M5 package). The deployed dashboard uses a derived forecast file rather than querying the raw M5 files at runtime.
- **Unit of forecast:** Item–store–day; seven-day forecasts displayed for a historical January 2016 demonstration.
- **Forecast model:** LightGBM with precomputed point estimates and daily lower/upper scenarios (`p10`, `p90`). Model work and evaluation artifacts are located under `forecasting/` and `evaluation/`.
- **Scope:** The included forecast dataset is a **50% forecasting sample**, not all Walmart products. The number of item-store series displayed depends on the active analysis scope.
- **Forecast evaluation:** Project evaluations are documented separately; avoid interpreting the app as a live accuracy or replenishment validation system.

## Inventory inputs and calculations

The sidebar provides two inventory modes:

1. **Simulated inventory:** Generates illustrative starting stock for forecasted item-store pairs.
2. **Uploaded inventory:** Accepts a CSV with the columns `store_id,item_id,on_hand`. For missing item-store records or missing `on_hand` values, the user chooses **use simulated inventory** or **assume 0 units**. An explicitly entered `0` is treated as zero stock.

Inventory Planning also supports a manual override for a chosen product. No inventory is fetched from Walmart systems. The uploaded values are supplied by the user for the historical demonstration and should not be assumed accurate or current.

For each product, an illustrative point-forecast gap is calculated from predicted seven-day demand and starting stock. Summing positive item-level gaps is **not** equivalent to subtracting total stock from total demand. Units-to-order and stockout estimates exclude real lead-time constraints, inbound shipments, substitution effects, and validated ordering policies.

## Prices versus costs: pending business clarification

**The original M5 dataset includes historical selling prices, but it does not provide Walmart's actual product procurement costs.** These are different quantities.

- The dashboard currently uses an **adjustable assumed unit cost** (default **$5**) to estimate illustrative ordering expense: `estimated order cost = illustrative units to order × assumed unit cost`.
- Historical **selling price** could support a separate estimated sales-value or revenue-at-risk metric after the appropriate product, store, and week join, but it is **not** a substitute for procurement cost.
- The dashboard's current assumed selling-price field is likewise illustrative; neither field should be presented as actual transaction prices or purchasing costs.
- **Decision pending:** Retain the cost assumption until the sponsor and professor advise whether to incorporate historical selling prices, user-supplied costs, or another explicitly defined metric. No cost-model change is planned before that feedback.

## AI Assistant architecture

1. Load precomputed forecast data and the chosen inventory values.
2. Determine analysis scope from the question (or let the user override it manually).
3. Aggregate forecast and gap metrics deterministically in Python and display tables/KPIs.
4. Retrieve eligible historical evidence from the local RAG corpus (`rag_backend.py`).
5. Send the question, structured numeric context, and retrieved passages to Gemini for an explanation.

**AI interpretation is not a decision engine.** Gemini does not change inventory, place orders, or re-run LightGBM. Outputs should be checked against the displayed tables. References to store/department shortages mean model-calculated gaps under the selected stock assumptions, not confirmed real-world stockouts.

## Repository structure

```text
.
├── app.py                     # Streamlit dashboard (four pages)
├── analysis_scope.py          # Scope selection and numeric summaries
├── inventory_bridge.py       # CSV inventory parsing and stock-source logic
├── rag_backend.py             # Retrieval of historical evidence
├── dashboard_forecasts.csv    # Precomputed forecast output
├── requirements.txt          # Python dependencies
├── forecasting/              # Forecast development notebook(s)
├── RAG/                      # Retrieval development and indexed evidence
├── evaluation/               # Forecast / retrieval evaluation artifacts
├── README.md
└── .gitignore
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Configure a Google Gemini API key in Streamlit Secrets (or an appropriate local secrets/environment setup):

```toml
GEMINI_API_KEY = "YOUR_KEY_HERE"
```

Do **not** commit real API keys or uploaded inventory containing confidential information to GitHub. If Gemini is unavailable, forecast displays and local RAG retrieval can still work independently.

## Current limitations

- Data are historical (2011–2016), and the deployed forecasts are precomputed, not generated by real-time inference.
- Sales are an imperfect proxy for unconstrained customer demand, especially during out-of-stock periods.
- Forecasting coverage is limited to the included sample; extrapolation to all products or current Walmart operations is not supported.
- Inventory is simulated or user-uploaded; observed stockouts, supplier lead times, inbound orders, and actual procurement costs are unavailable.
- Default cost and price assumptions are illustrative. **Actual procurement cost is unknown; historical M5 selling prices are not procurement costs.**
- Daily P10/P90 scenarios should not be interpreted as validated seven-day prediction intervals when summed across days.
- Positive forecast gaps, stockout dates, suggested order quantities, and cost estimates have not been operationally validated.
- RAG evidence may be broader than a specific item or location; retrieved similarity does not establish causality. Gemini explanations can be mistaken and must be verified against the calculated tables and original sources.
- The prototype does not autonomously order, change stock, or connect to Walmart's operational systems.

## Status and next review

The current code supports store/department comparisons, product forecasts, inventory inputs and overrides, stock-runway views, and forecast-aware Gemini + RAG explanations. Next changes will be considered **after sponsor and professor feedback**, particularly concerning historical prices, assumed cost values, and decision-support validation.

See **`PROJECT_UPDATE_2026-10-27.md`** for the current handoff notes and verification checklist.
