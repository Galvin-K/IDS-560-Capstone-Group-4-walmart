# Walmart M5 | Demand & Inventory Decision Support

**UIC IDS 560 Capstone — Group 4**  
**Team:** Keta Patel, Galvin Kao, Tyler Yuen, Fan Mengxi, Mengyi Wang  
**Sponsor:** Anjali Garg (Walmart) | **Faculty:** Prof. Fatemeh Sarayloo  
**Last documentation update:** October 27, 2026  
**Status:** Working academic prototype; sponsor and faculty feedback pending

**[Open the Streamlit dashboard](https://ids-560-capstone-group-4-walmart.streamlit.app/)**

## 1. Project overview

**Business question:** *How can a store manager order enough inventory to meet customer demand without ordering too much?*

The project uses the public Walmart M5 data to study sales behavior, forecast demand by product and store, and turn the numerical results into a dashboard that supports inventory decisions. The approach has four parts:

1. **Understand past demand:** Explore sales patterns across products, stores, dates, prices, and SNAP benefit days.
2. **Forecast demand:** Compare simple baselines, intermittent-demand models, and LightGBM using future periods the models have not seen.
3. **Explain the results:** Retrieve historical evidence and use a language model to explain Python-calculated forecasts and inventory gaps.
4. **Present possible actions:** Display forecasts, illustrative inventory levels, prioritized items, order-gap calculations, and limitations for a manager to review.

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

## 6. RAG and AI explanation methodology

### Historical evidence retrieval

The RAG work assembled public historical sources, including:

- Federal Reserve Beige Book materials;
- NOAA Storm Events records;
- SNAP-related government information;
- Walmart SEC filings; and
- M5-derived historical evidence cards.

The retrieval index contained of approximately **7,333 dated passages from 4,796 documents**, embedded using a BGE-small sentence embedding model. Passages are tagged with dates and relevant geographic context; retrieval filters out documents published **after the forecast decision date** to reduce future-information leakage.

The deployed application calls `rag_backend.py` to retrieve supporting passages, and Google Gemini produces the readable explanation. **Historical evidence is context, not a substitute for numerical forecast or inventory calculations.** The dashboard's main numeric summaries are calculated in Python before they are passed to Gemini.

### What the RAG tests found

Retrieval testing found:

- **13 of 15** questions requiring a relevant source returned the expected kind of evidence.
- **No future-dated documents** appeared in the reported date-filter checks (16 queries plus 200 random checks).
- A question with no reasonable historical answer (COVID-19 demand in a pre-2020 library) still scored **0.68** in similarity, overlapping the scores of valid matches.

The last finding matters: **a similarity threshold alone is not a reliable no-answer detector**. The assistant must still distinguish supported claims from missing evidence.

An early comparison across three test orders also found that RAG + LLM cited sources in **84% of its reasons**, compared with **0%** for an LLM without retrieval. A LangChain variant accessed more kinds of evidence but cited less consistently and needed more model calls. These are **small-sample experiments**, not validated production benchmarks or claims about the current Gemini deployment.

**Current implementation:** The working Streamlit assistant now combines deterministic Python forecast/inventory summaries with retrieved RAG passages and Gemini explanations, rather than restricting the assistant to historical retrieval alone. Tyler's newer `M5_RAG_agent_v3.ipynb` was shared separately but had **not yet been merged/tested** in the working dashboard as of this update.

## 7. Dashboard — current implementation

**Live application:** https://ids-560-capstone-group-4-walmart.streamlit.app/

The Streamlit interface reads **precomputed** LightGBM results from `dashboard_forecasts.csv` and presents four pages. It is a decision-support **demonstration**, not an inventory management system.

| Page | Current functionality |
| --- | --- |
| **Executive Overview** | Selected-store forecast overview, inventory KPIs, adjustable assumed costs/lead time, store comparisons and priority list; select **5, 10, 20, or 50** rows and view illustrative ordering costs. |
| **Demand Forecasting** | Department-level summary table and chart first; then page-specific department/product selection, item KPIs, daily point forecasts and P10/P90 displays, and evaluation details. |
| **Inventory Planning** | Prioritized action list, simulated order list, page-specific product/department choice and stock override, inventory KPIs, product stock trajectories, user-selected multi-product stock runway, and potential-shortage views. |
| **AI Assistant** | Automatic or manual analysis scope, deterministic forecast and inventory context, comparisons at product/department/store/all-store level, historical RAG retrieval, and Gemini explanations. |

### Inventory input rules

The sidebar supports **Simulated inventory** and **Uploaded inventory**. The uploaded CSV format is:

```csv
store_id,item_id,on_hand
CA_1,FOODS_1_001,25
CA_1,FOODS_1_002,0
TX_1,HOUSEHOLD_1_001,12
```

If an item–store pair is missing from the uploaded file, or its `on_hand` value is blank, the user can select **simulated fallback** or **assume 0 units**. An explicit zero remains zero. The inventory planning page also supports a **manual stock override** for a chosen product.

**Missing inventory is not the same as confirmed zero inventory.** Assuming zero is a user-selected demonstration scenario, not a verified Walmart stock reading.

### What the calculations mean

For each item–store series, the dashboard compares seven-day predicted units with starting stock. In simplified form:

```text
illustrative_positive_gap = max(0, predicted_7_day_units - starting_stock)
illustrative_units_to_cover = ceil(illustrative_positive_gap)
estimated_order_cost = illustrative_units_to_cover × assumed_unit_cost
```

The dashboard also displays estimated runoff/stockout timing under the scenario of **no new replenishment**. These are *scenario calculations*, not verified order quantities or observed stockouts. **Sum of positive item-level gaps is not the same as total forecast demand minus total starting stock**: excess stock for one item cannot automatically cover a different item's shortage.

The default **assumed unit procurement cost is $5**, and users can change it. The M5 price file provides **historical selling prices**, not Walmart's actual procurement costs. Estimated order expense therefore uses an assumption, not observed purchase costs. Historical selling prices could be incorporated later for a separately defined estimated sales-value metric, subject to sponsor/faculty input.

### How Gemini and RAG fit into the dashboard

```text
M5 historical sales + calendar + prices
                │
                ▼
    Offline forecast training / evaluation
                │
                ▼
       dashboard_forecasts.csv
                │
     user stock input / simulation
                │
                ▼
 Python forecast + inventory calculations ──► displayed KPIs / tables
                │
                ├──► structured context ───────┐
                │                              │
Historical documents / M5 evidence cards      │
                │                              │
       dated RAG retrieval ────────────────────┤
                                               ▼
                                        Gemini explanation
                                               │
                                               ▼
                                       Manager reviews
```

The **Automatic** scope selector uses rules based on question wording; it is not an independently validated semantic planner. Users can explicitly choose a product, department, selected store, or all stores. The assistant does not modify the forecasts, change stock, or place orders.

### Example demo questions

- `Which stores have the largest predicted inventory shortages over the next seven days?`
- `Which departments have the highest predicted inventory gaps across all stores?`
- `What is the forecast and starting stock for this selected product?`
- `What historical evidence could help explain Foods demand in Wisconsin?`

For the first two examples, the team manually checked that Gemini's displayed rankings aligned with the Python-generated numerical tables. That is **limited UI verification**, not comprehensive system validation.

## 8. Repository layout

```text
.
├── app.py                   # Streamlit UI and page logic
├── analysis_scope.py        # Question-scope routing and aggregate summaries
├── inventory_bridge.py     # Uploaded inventory and fallback calculations
├── rag_backend.py           # Historical evidence retrieval used by the app
├── dashboard_forecasts.csv  # Precomputed 7-day demo forecasts
├── requirements.txt        # Runtime Python dependencies
├── forecasting/            # Forecast notebooks and modeling work
├── RAG/                    # RAG development and evidence artifacts
├── evaluation/             # Evaluation work
├── Development Updates/    # Dated progress notes
├── README.md
└── .gitignore
```

*Keep the Python entrypoint and relative data paths together when reorganizing files. The working Streamlit app currently expects `app.py` at the repository root.*

## 9. Running the dashboard

From the repository root:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The Gemini API key belongs in **Streamlit Secrets** for the deployed app (or in a securely configured local environment), never in the committed repository:

```toml
GEMINI_API_KEY = "YOUR_KEY_HERE"
```

The app retrieves forecasts from its checked-in CSV; it does **not** retrain LightGBM on each page load. If Gemini is unavailable, the forecasting and calculation portions are separate from the API-generated explanation. The app may need the local RAG index files for evidence retrieval.

## 10. Limitations and open questions

1. **Historic, limited scope.** M5 covers 2011–2016. The live UI shows a historical January 2016 case, using only the 15,245-series sample, not current Walmart demand.
2. **Sales are not unconstrained demand.** Out-of-stock periods and missed sales cannot be observed directly.
3. **No verified inventory or purchasing inputs.** Stock, replenishment timing, procurement costs, and operational order constraints are user-provided or assumed. Order suggestions remain illustrative.
4. **Cost ≠ selling price.** The adjustable $5 assumed unit cost is not Walmart procurement cost; M5's historical selling prices cannot simply replace it.
5. **Imperfect forecasts and uncertainty coverage.** Daily WAPE remains substantial for sparse series; historical P10/P90 coverage fell below the nominal 80% target. Summing daily bounds does not give a calibrated seven-day interval.
6. **RAG/LLM limitations.** A high similarity score does not guarantee useful evidence. Retrieval can miss relevant context, and Gemini can overstate or misread statistics; original sources and Python tables take priority.
7. **No production integration.** There is no live database connection, validated stockout record, warehouse synchronization, production forecasting endpoint, or autonomous ordering.
8. **Not yet fully regression-tested.** Selected cross-store and department questions worked during manual testing, but all stock-upload scenarios, overrides, page combinations, and explanation claims still need systematic verification.



## 12. Project documentation and evaluation

This README combines:

- **Research and evaluation:** The EDA, model-selection results, holdout metrics, uncertainty evaluation, and initial RAG tests are documented in the project research materials and relevant notebooks.
- **Forecasting work:** See the notebook(s) under `forecasting/` for implementation details and outputs.
- **Current dashboard:** The code in `app.py`, `analysis_scope.py`, `inventory_bridge.py`, `rag_backend.py`, and the checked-in `dashboard_forecasts.csv` describes the deployed demonstration.
- **Change history:** See dated Markdown files under `Development Updates/` for subsequent implementation changes.

**Interpretation rule:** Offline model-evaluation results, manually checked dashboard examples, and proposed future work are different kinds of evidence. They should not be presented as if they were all independently validated current-system results.
