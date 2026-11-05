# Walmart M5 | Demand & Inventory Decision Support

**UIC IDS 560 Capstone — Group 4**  
**Team:** Keta Patel, Galvin Kao, Tyler Yuen, Fan Mengxi, Mengyi Wang  
**Sponsor:** Anjali Garg (Walmart) | **Faculty:** Prof. Fatemeh Sarayloo  
**Last documentation update:** November 4, 2026 (after deployed pricing and RAG checks)  
**Status:** Functional historical decision-support prototype; final regression checks and documentation review pending

**[Open the Streamlit dashboard](https://ids-560-capstone-group-4-walmart.streamlit.app/)**

## 1. Project overview

**Business question:** *How can a store manager order enough inventory to meet customer demand without ordering too much?*

The project uses the public Walmart M5 data to study sales behavior, forecast demand by product and store, and turn the numerical results into a dashboard that supports inventory decisions. The approach has four parts:

1. **Understand past demand:** Explore sales patterns across products, stores, dates, prices, and SNAP benefit days.
2. **Forecast demand:** Compare simple baselines, intermittent-demand models, and LightGBM using future periods the models have not seen.
3. **Explain the results:** Retrieve historical evidence and use a language model to explain Python-calculated forecasts and inventory gaps.
4. **Present possible actions:** Display 7-day demand gaps, 14-day replenishment suggestions with historical-error safety stock and review holds, estimated inventory, and limitations for a manager to review.

This is **not a live Walmart application**. The project has no access to Walmart's internal stock, supplier, purchasing, or order-management systems. The dashboard does **not** place orders or independently determine an operationally optimal ordering policy.

## 2. Data

### M5 sources and granularity

The analysis uses three primary files from the public M5 forecasting dataset:

| Source | Contents | How it is used |
| --- | --- | --- |
| Sales (`sales_train_*`) | Daily unit sales for each item–store series | Historical outcomes and lag/rolling-demand features |
| Calendar (`calendar.csv`) | Dates, weekdays, events/holidays, SNAP flags by state | Calendar and event features |
| Selling prices (`sell_prices.csv`) | Historical weekly selling prices by item, store, and Walmart week | Price-related analysis and forecast features |

The **modeling unit is an item–store–day**. The original data cover **3,049 items, seven departments, three broad categories (Foods, Hobbies, Household), and 10 stores** across California (4), Texas (3), and Wisconsin (3). The dataset contains approximately **59.2 million daily sales observations**, **30,490 item–store series**, and **1,941 daily observations per series** (January 2011–May 2016). These are dataset-level counts, *not* the number of records loaded into the Streamlit application.

Sales are joined to the calendar by day and to prices by item, store, and Walmart week. The team conducted nine integrity checks, removed pre-launch periods before an item's first sale from relevant analysis, and handled incomplete historical price coverage. The source price table covers approximately **79% of item–store weeks**; the project does not infer that a missing price represents a sale price of zero.

### Sampling strategy

The team sampled **complete item–store series**, not isolated daily rows, using proportional quotas across **nine state × category groups** and a fixed random ordering:

| Dataset | Item–store series | Share of full M5 series |
| --- | ---: | ---: |
| Full M5 set | 30,490 | 100% |
| Larger modeling sample | 15,245 | 50% |
| Smaller modeling sample | 6,098 | 20% |
| Initial prototype subset | 100 | Exploratory subset |

The 20% sample is nested within the 50% sample. The **deployed dashboard uses the 50% sample's precomputed forecasting output**. Results should not be described as performance for all of Walmart's products.

## 3. Exploratory data analysis

The following findings summarize our exploratory analysis. They describe observed relationships, **not established causes**.

| Area | Finding | Why it matters |
| --- | --- | --- |
| Sparse demand | **68.0%** of item–store days had zero recorded sales | A single overall accuracy score can hide poor performance on sporadic series |
| Demand types | **72.7% intermittent**, **18.3% lumpy**, **6.2% smooth**, **2.8% erratic** | Motivates intermittent-demand baselines and segmented evaluation |
| Sales trend | Average daily units increased **14.7%** from 2012 to 2015 | Older training data can produce systematic bias as sales levels change |
| Category mix | Foods accounted for **68.6% of units** but approximately **58.0% of estimated revenue** | Volume and financial importance are different rankings |
| Revenue concentration | Approximately **37%** of item–store series accounted for **80% of estimated revenue** | Reviewing items in priority order is more practical than reviewing all 30,490 series |
| Weekdays | Saturday sales were about **21% above** an average day; Wednesdays about **13% below** | Weekday features and planning cycles matter |
| Holidays | Labor Day was about **24% above** a local baseline; Thanksgiving **29% below** | Event effects need explicit calendar treatment |
| SNAP | Wisconsin Foods sales were approximately **30.4% higher on SNAP days**; Texas **15.9%**, California **10.6%** | State/category-specific calendar information helps describe variation |
| Price association | Sales were about **1.24× normal** for weeks priced 10–20% below the item's typical price | Useful predictive association, **not** proof of discount causality |

**Important limitation:** M5 provides sales, not unconstrained demand. A zero-sale day may reflect no customer purchases, a stockout, a product not being offered, or other factors. Actual inventory and lost sales are not observed.

## 4. Forecasting methodology

### Evaluation design

The project used **chronological evaluation**, keeping training and testing in their correct time order:

- **Training:** 2011–2014.
- **Validation:** Rolling-origin evaluation during 2015, with roughly 50 weekly forecast origins.
- **Final holdout:** January–May 2016, evaluated after model selection.
- **Forecast horizons:** 7, 14, and 28 days.
- **Unit of comparison:** Daily item–store forecasts, using comparable series, forecast dates, and error definitions.

Demand-type classifications were calculated from the training period, rather than using future observations. A leakage audit was also conducted to ensure forecasts did not use future sales as input.

### Demand segmentation

Demand patterns were classified with **Average Demand Interval (ADI)** and the **squared coefficient of variation (CV²)** of nonzero demand. The analysis used cutoffs of **ADI = 1.32** and **CV² = 0.49** to distinguish smooth, intermittent, erratic, and lumpy series. A separate no-sales group captured series without positive sales in the training history.

This segmentation was used both to understand error patterns and to guide forecasting-range calibration.

### Forecast candidates

We evaluated several model families under a common protocol:

- **Simple baselines:** Zero Naive, Weekly Seasonal Naive, Recent Mean (7-day), and a 28-day Recent Mean control.
- **Intermittent-demand methods:** Croston, Syntetos–Boylan Approximation (SBA), and Teunter–Syntetos–Babai (TSB).
- **Machine learning:** **LightGBM**, with a Tweedie objective suited to nonnegative, zero-heavy sales.

The final LightGBM implementation used **28 features in six groups**, including historical rolling sales levels and variability, price information, item/store attributes, calendar/SNAP/event signals, same-weekday history, and forecast lead time. Model-gain analysis assigned about **89.3% of gain to recent sales features**; that reflects what the model uses, **not** a causal estimate of added accuracy.

### Bias handling and model selection

The first LightGBM version systematically under-forecast as total sales increased over time. The validation results were:

1. The initial 2015 signed bias was approximately **−4.4%**.
2. Quarterly retraining reduced it to approximately **−3.1%**.
3. A correction based on the previous 13 weeks of known errors, by demand type and capped at **0.8–1.25×**, reduced it to approximately **−0.3%**.

The team predefined four model-selection checks related to improvement on sporadic demand, performance relative to the 28-day control, not harming smooth/erratic demand, and avoiding excessive aggregate bias. The evaluation showed that the selected LightGBM configuration passed those checks in the evaluated samples and horizons.

### Metrics

- **WAPE:** `sum(abs(actual - forecast)) / sum(actual) × 100%` — primary relative absolute-error metric; lower is better.
- **Signed bias:** `sum(forecast - actual) / sum(actual) × 100%` — indicates net over- or under-forecasting.
- **RMSE:** Emphasizes larger misses.
- **Pinball loss and empirical coverage:** Evaluate daily P10/P90 ranges.

**WAPE depends on the evaluation granularity.** Daily item–store WAPE is not directly comparable to a score computed on weekly totals or aggregate store sales.

## 5. Forecasting results

### Final 2016 holdout: WAPE by horizon

The following results are from the 2016 holdout evaluation using the 50% stratified sample.

| Forecast horizon | LightGBM WAPE | Recent Mean 7d WAPE | Recent Mean 28d WAPE |
| --- | ---: | ---: | ---: |
| **7 days** | **71.6%** | 74.6% | 73.9% |
| **14 days** | **72.6%** | 75.8% | 74.7% |
| **28 days** | **73.9%** | 77.3% | 75.9% |

At the seven-day horizon, the evaluation also found **RMSE 2.040** for LightGBM versus **2.219** for Recent Mean, and signed bias of **−0.7%** versus **−1.8%**. These are results on historical holdout sales, not live-order performance.

For 2015 validation, the final LightGBM achieved approximately **75.0% seven-day WAPE**, compared with **77.5%** for Recent Mean on the validation set.

### Practical historical example: January 1–7, 2016

The week after Christmas provides a useful example to illustrate the value of recognizing calendar effects:

- **Actual sales:** 130,915 units across the 15,245-series sample.
- **LightGBM forecast:** approximately 136,272 units (**+4.1%** relative to actual).
- **Recent Mean 7-day forecast:** approximately **−19.6%** relative to actual, partly because the previous week included Christmas closure.

This is the **historical week used in the deployed dashboard**, not a prediction for the current calendar week. A close forecast is useful, but it does not establish avoided stockouts or optimal orders.

### Where errors remain

Baseline diagnostics found that:

- Intermittent and lumpy series contributed roughly **71.9% of the baseline's total absolute error**.
- Foods contributed approximately **59.4% of absolute error**, while Hobbies had the largest **category-level WAPE (104.7%)**.
- High-volume series contributed approximately **62.4%** of the baseline absolute-error total, even though low-volume products often had worse *relative* WAPE.

These figures describe **Recent Mean baseline error analysis**; they should not be presented as a segmented error decomposition for the final LightGBM model.

### Prediction ranges

The forecasting work also generated **daily P10/P90 ranges**. The 2016 evaluation found **77.7% empirical coverage on 2016 daily observations**, slightly below the **80% target**, with demand-type-based calibration proposed/selected because a single pooled range performed unevenly across demand types.

**Dashboard-specific caution:** The deployed `dashboard_forecasts.csv` contains daily `p10` and `p90` columns. Summing daily bounds produces *illustrative seven-day scenarios*, **not** a calibrated prediction interval for the seven-day total. The displayed file's range-calibration variant should be checked against the saved forecasting notebook before claiming that the deployed bounds use the selected demand-type calibration.

## 6. AI assistant and retrieval methodology

The dashboard uses **Python** to calculate forecast and inventory summaries; **Gemini** generates explanations. Historical documents are retrieved by `rag_backend.py` and supplied as context, not used as a source of truth for numeric KPIs.

### Retrieval system (official v4)

The active configuration tested in Streamlit uses **hybrid evidence retrieval with cross-encoder reranking**, using the v4 RAG notebooks and shared `rag_common.py` implementation. The live configuration uses:

```toml
RAG_USE_V4 = "1"
V4_POLICY_ENABLED = "0"
```

`V4_POLICY_ENABLED = "0"` leaves the **memory-heavy online policy engine disabled**. The dashboard instead reads the smaller, **offline-computed** policy table. It does **not** disable the separate 14-day replenishment section. Retrieval filters supporting passages by decision date, location, category and a provisional relevance threshold. The relevance threshold is **not** a validated method for detecting every unanswerable question.

Offline RAG benchmarks reported precision@5 **0.376** and hit@5 **0.848** for hybrid + reranking, compared with **0.315** and **0.758** for the dense baseline. These are notebook benchmarks, **not measured live accuracy** for the deployed assistant. Original retrieval materials include government sources, economic and weather context, corporate filings and M5-derived evidence notes; evidence does not establish causal explanations.

### Historical-sales comparison limitation

A retrieved document or qualitative M5 evidence passage is **not the same as a queryable item–store historical sales table**. As of this update, the assistant has not been verified to retrieve exact historical units for arbitrary category/store/date comparisons. On November 4, a focused test asked the AI Assistant to compare **actual Foods units sold in CA_1 during January 1–7, 2014 versus January 1–7, 2015** and calculate the year-over-year change. The assistant retrieved **five historical context passages** but explicitly said it could not access the exact historical totals. It **did not invent numbers**, correctly distinguishing forecast snapshots from actual sales. **Result:** the answer-safety behavior passed; arbitrary date/store/category numerical querying remains **not implemented**. The relevant underlying M5 sales exist in the research dataset, but they are not exposed to the deployed assistant as a structured query interface.

The assistant does not purchase inventory, alter stock, calculate new LightGBM forecasts, or verify its own historical claims against raw M5 tables.

## 7. Deployed dashboard — November 2026

**Application:** https://ids-560-capstone-group-4-walmart.streamlit.app/

The application presents **five tabs**:

| Tab | Manager-facing purpose | Data/horizon |
| --- | --- | --- |
| **Executive Overview** | Review demand gaps, estimated price-based exposure, and a separate replenishment summary | 7-day stock-gap view; 14-day ordering summary |
| **Demand Forecasting** | Compare departments and examine daily item forecasts, with **independent 7/14-day selectors** by chart/section | Precomputed daily LightGBM forecasts |
| **Inventory Planning** | Review a prioritized stock-gap list, individual stock projections, product comparisons, and the 14-day safety-stock/order plan | Independent 7/14-day stock checks; separate fixed 14-day policy |
| **Model Insights** | Inspect historical model comparisons, forecast error and bias, demand-pattern differences, regional diagnostics and feature importance | 7/14/28-day held-out research evaluations |
| **AI Assistant** | Ask scope-specific questions and get Gemini explanations informed by precomputed numeric summaries and retrieved historical context | Forecast context + v4 RAG |

### Dates and forecasting inputs

- **Decision date:** December 31, 2015.
- **7-day demand views:** January 1–7, 2016; baseline file `dashboard_forecasts.csv`.
- **14-day demand views:** January 1–14, 2016; compact daily export `forecasting/dashboard_forecasts_14d_2015_12_31.csv.gz` from the historical 14-day LightGBM forecast cube.
- **14-day replenishment policy:** `forecasting/v4_inventory_policy_2015_12_31.csv` with approximately **15,245 item–store policy rows**. For a selected item, the dashboard displays the forecast total, historical-error safety stock, target inventory, recommended quantity, confidence and review/hold flags.

**Horizon controls are intentionally independent by section.** Selecting a 14-day stock projection does **not** change the fixed 14-day replenishment policy. Stock-projection graphs assume **no additional deliveries** unless explicitly stated; they are not predictions of actual on-hand inventory.

### Inventory and ordering assumptions

Historical M5 **does not contain verified current on-hand stock**, order-in-transit quantities, or supplier lead-time records. The dashboard estimates starting stock and permits **manual overrides** for the selected item. It also permits a user-entered quantity already on order for the 14-day policy display.

For a non-held item, suggested order quantity is recomputed from the offline target:

```text
inventory_position = estimated_or_entered_on_hand + already_on_order
suggested_units = ceil(max(0, 14_day_target - inventory_position))
```

An item flagged **hold for manager review** shows **zero automatically recommended units**; any conditional quantity requires a separate management decision. The demand forecast, safety stock, confidence and hold flag **do not change** when the user changes starting inventory. The offline policy assumes a **14-day protection period** (7-day lead-time assumption plus 7-day review interval). Altering that period would require regenerating policy outputs.

The 7-day or 14-day **stock-gap action list** is a diagnostic, not the same as the separate 14-day order recommendation. The latter includes a safety buffer and review holds. The application does not connect to procurement or execute orders.

### M5 price-based dollar figures

Prices now come from a **store/item-specific historical M5 selling-price extract**, generated by `export_m5_prices.py` from M5 calendar and sell-price records, and stored at `forecasting/m5_prices_asof_2015_12_31.csv.gz`. The price is selected as of the December 31, 2015 decision date. Missing prices remain missing; monetary totals should disclose when they include only products with available prices.

These are **price-based values**, not Walmart supplier or procurement costs, verified revenues, or profits. M5's selling price is being used as a **proxy for illustrative order value**, not as a validated purchase price. The earlier fixed **$5 assumed cost / $8 selling price inputs** are no longer the active price source.

### Uncertainty and historical evaluation

The dashboard may display daily lower/higher demand scenarios derived from P10/P90 estimates. Summed daily bounds **are not a calibrated multi-day prediction interval**. The historical prediction-range evaluation reported approximately **77.7% empirical daily coverage** versus an 80% goal.

Model Insights compares **Recent Mean 7**, **Recent Mean 28**, **TSB**, **SES** and selected **LightGBM** on held-out data, with 7-, 14-, and 28-day research results. Separate charts break down forecasting error and bias by demand pattern, category, state and important features. **Croston and SBA were also explored in the statistical notebook**, but were not deployed as alternative daily forecast feeds. LightGBM remains the selected forecasting model; exploratory statistical methods do not silently replace it.

The evaluation uses historical item–store samples, not live Walmart outcomes. The current dashboard computes forecast KPIs from pre-exported artifacts; it does not retrain or evaluate its model live.

## 8. Repository structure

```text
.
├── app.py                                  # Streamlit five-tab interface
├── dashboard_forecasts.csv                 # Original 7-day daily forecasts
├── analysis_scope.py                       # AI question scope helpers
├── inventory_bridge.py                    # Estimated inventory / legacy calculations
├── inventory_engine.py                    # Policy and stock scenario calculations
├── v4_policy_bridge.py                     # Compact offline v4 policy access
├── rag_backend.py                          # RAG retrieval integration
├── rag_common.py                           # Shared v4 routines (offline/retrieval)
├── export_m5_prices.py                     # Offline historical price extraction
├── precompute_v4_inventory.py              # Offline 14-day policy generation
├── forecasting/
│   ├── dashboard_forecasts_14d_2015_12_31.csv.gz
│   ├── v4_inventory_policy_2015_12_31.csv
│   ├── m5_prices_asof_2015_12_31.csv.gz
│   ├── forecast_cube_lightgbm_50.npz
│   ├── series_segments_50.csv
│   ├── lightgbm_error_by_segment_50.csv
│   ├── 03_M5_Statistical_Model_Exploration.ipynb
│   └── 04_M5_LightGBM_Final_Forecasting.ipynb
├── M5/                                     # Public M5 source data used offline
├── RAG/                                    # Evidence files and five v4 notebooks
├── evaluation/                             # Historical research/evaluation
├── test/                                   # Python checks
├── Development Updates/                   # Dated implementation notes
├── requirements.txt
└── README.md
```

Some large research artifacts use Git LFS. The Streamlit dashboard uses **small precomputed files** for daily forecast views, prices and policy decisions; it does not load the full forecast cube or raw M5 sales tables on every page refresh.

## 9. Running the dashboard

From the repository root:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Set `GEMINI_API_KEY` in Streamlit Secrets and enable v4 retrieval only when its dependencies are available:

```toml
GEMINI_API_KEY = "YOUR_KEY_HERE"
RAG_USE_V4 = "1"
V4_POLICY_ENABLED = "0"
```

**Never commit API keys.** The original `M5/datasets/calendar.csv` and `M5/datasets/sell_prices.csv` are used to **generate** the compact price file offline. The deployed app reads the extract instead of opening the full source price dataset.

## 10. Validation status and limitations

### What has been checked

- The live app loads and shows the separate seven-day and 14-day planning outputs.
- User-entered stock and units already on order changed a non-held item's suggested quantity as expected in observed UI checks.
- A held-item example displayed its warning, low confidence and zero suggested units. A **held item with a positive conditional quantity** remains an important edge case to confirm.
- The store-wide 14-day policy table appeared and showed a download control; a downloaded CSV has **not** been independently compared row-by-row with the policy file.
- The dashboard displayed M5-price-based dollar figures after the price file upload and deployment/cache refresh, instead of “Price file needed” (for example, TX_2 seven-day gap value $8,782 and sales value at risk $181 in the observed UI). The **calculation formulas and incomplete-price behavior still require final validation**.
- The active AI screen reported official v4 retrieval. The November 4 historical Foods/CA_1 comparison test retrieved five passages and correctly declined to invent unavailable actual sales totals; **structured historical numerical querying is still unsupported**.

### Confirmed RAG limitation and possible next step

The AI Assistant currently retrieves relevant historical *summaries*, not arbitrary raw M5 daily-sales aggregates. A future enhancement could export a compact, keyed historical sales table and use a deterministic Python/SQL calculation tool for store/category/date totals, then pass verified results to Gemini for explanation. This capability was **not** implemented during the November 4 update.

### Remaining limitations

- M5 is historical (2011–2016), not live store demand; only the **50% series sample** is modeled in the deployed policy.
- Sales are observed transactions, not necessarily full demand when products were out of stock.
- Inventory, supplier timing and pricing-cost equivalence are assumptions or proxies, not live Walmart measurements.
- Forecast uncertainty ranges and review/hold logic have research/backtest results, **not production validation**.
- Gemini can miss evidence or generate unsupported figures; retrieved passages do not guarantee exact historical aggregations.
- The application is research decision support, not a connected enterprise ordering service.


## 11. Research sources and documentation

- `forecasting/01_M5_Data_Preparation.ipynb` — preparation and sampling.
- `forecasting/02_Baseline_Forecasting_and_Error_Analysis.ipynb` — baseline forecasts and error analysis.
- `forecasting/03_M5_Statistical_Model_Exploration.ipynb` — Croston-family, TSB, SES exploratory evaluation and screening.
- `forecasting/04_M5_LightGBM_Final_Forecasting.ipynb` — selected model, held-out evaluation and diagnostics.
- `RAG/M5_RAG_1_evidence_library_v4.ipynb` through `RAG/M5_RAG_5_benchmark_v4.ipynb` — v4 retrieval and decision-support research.
- `Development Updates/Update 2.md` — November 4 implementation log, including Streamlit integration and testing; see [`Update 2.md`](Development%20Updates/Update%202.md).

All claims about model accuracy, retrieval quality or order-policy coverage must be labeled **offline research results** unless independently reproduced in the deployed app.
