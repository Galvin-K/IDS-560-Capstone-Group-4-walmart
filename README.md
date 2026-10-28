
# Walmart M5 | Demand & Inventory Decision Support

**UIC IDS 560 Capstone — Group 4**  
**Team:** Keta Patel, Galvin Kao, Tyler Yuen, Fan Mengxi, Mengyi Wang  
**Sponsor:** Anjali Garg (Walmart) | **Faculty:** Prof. Fatemeh Sarayloo  
**Last documentation update:** October 27, 2026  
**Status:** Working academic prototype; sponsor and faculty feedback pending

**[Open the Streamlit dashboard](https://ids-560-capstone-group-4-walmart.streamlit.app/)**

## 1. Project Overview

**Business question:** *How can a store manager order enough inventory to meet customer demand without ordering too much?*

The project uses the public Walmart M5 data to study sales behavior, forecast demand by product and store, and turn the numerical results into a dashboard that supports inventory decisions.

Our approach has four parts:

1. **Understand past demand:** Explore sales patterns across products, stores, dates, prices, and SNAP benefit days.
2. **Forecast demand:** Compare simple baselines, intermittent-demand models, and LightGBM using future periods the models have not seen.
3. **Explain the results:** Retrieve historical evidence and use a language model to explain Python-calculated forecasts and inventory gaps.
4. **Present possible actions:** Display forecasts, illustrative inventory levels, prioritized items, order-gap calculations, and limitations for a manager to review.

This is **not a live Walmart application**. The project has no access to Walmart's internal stock, supplier, purchasing, or order-management systems. The dashboard does not place orders or independently determine an operationally optimal ordering policy.

## 2. Dataset and Data Preparation

### 2.1 M5 Data Sources

The midterm analysis uses three primary files from the public M5 forecasting dataset:

| Source | Contents | How it is used |
| --- | --- | --- |
| Sales (`sales_train_*`) | Daily unit sales for each item–store series | Historical outcomes and lag/rolling-demand features |
| Calendar (`calendar.csv`) | Dates, weekdays, events/holidays, SNAP flags by state | Calendar and event features |
| Selling prices (`sell_prices.csv`) | Historical weekly selling prices by item, store, and Walmart week | Price-related analysis and forecast features |

The **modeling unit is an item–store–day**.

The original data cover:

- **3,049 products**
- **7 departments**
- **3 categories:** Foods, Hobbies, Household
- **10 stores:** California (4), Texas (3), Wisconsin (3)
- **30,490 item–store series**
- **Approximately 59.2 million daily sales observations**
- **1,941 daily observations per series**, covering January 2011–May 2016

These are dataset-level counts, not the number of records loaded into the Streamlit application.

### 2.2 Data Cleaning and Joining

Sales are joined to the calendar by day and to prices by item, store, and Walmart week.

As described in our midterm presentation, we:

1. Joined sales, calendar, and selling-price data using their shared identifiers.
2. Conducted nine data-integrity checks.
3. Removed periods before a product's first sale from relevant analyses.
4. Checked incomplete price coverage rather than interpreting missing prices as zero.
5. Created stratified samples for model development and evaluation.

The historical price table covers approximately **79% of item–store weeks**.

### 2.3 Sampling Strategy

We sampled **complete item–store series**, rather than individual daily records, using proportional sampling across nine state × category groups.

| Dataset | Item–store series | Share of full data |
| --- | ---: | ---: |
| Full M5 dataset | 30,490 | 100% |
| Main modeling sample | 15,245 | 50% |
| Smaller modeling sample | 6,098 | 20% |
| Initial prototype sample | 100 | Exploratory subset |

The 20% sample is nested within the 50% sample.

The deployed dashboard uses the **50% sample's precomputed forecasting results**. Therefore, its results represent the included sample rather than every product in the full M5 dataset.

## 3. Exploratory Data Analysis

Our EDA focused on understanding demand patterns and identifying factors that could affect forecasting accuracy.

The following findings come from our midterm presentation.

### 3.1 Demand Sparsity

Approximately **68% of item–store days have zero recorded sales**.

Products were classified into four demand types:

| Demand Type | Share |
| --- | ---: |
| Intermittent | 72.7% |
| Lumpy | 18.3% |
| Smooth | 6.2% |
| Erratic | 2.8% |

Intermittent and lumpy demand account for approximately **90.9%** of the classified series.

This matters because models that perform well on frequently purchased products may struggle with products that sell irregularly.

### 3.2 Historical Sales Growth

Average daily sales increased by approximately **14.7% between 2012 and 2015**.

This suggests that models trained on older sales levels may systematically underpredict later demand.

The forecasting pipeline therefore considers periodic retraining and bias correction.

### 3.3 Category and State Differences

Foods accounted for approximately **68.6% of total units sold**, but only **58.0% of estimated revenue**.

Household products accounted for a smaller share of units but a larger share of revenue relative to their sales volume.

California represented approximately **43.6% of units sold**.

These findings suggest that demand volume and financial importance are different considerations when prioritizing inventory decisions.

### 3.4 Revenue Concentration

Approximately **37% of item–store series generated 80% of estimated revenue**.

The top 10% of series generated approximately 47% of estimated revenue.

This supports the idea of prioritizing products instead of requiring managers to manually examine every item–store combination.

### 3.5 Weekday and Seasonal Patterns

Our analysis found that:

- Saturday sales were approximately **21% above** an average day.
- Wednesday sales were approximately **13% below** average.
- August sales were approximately 7% above average.
- January sales were approximately 9% below average.

The weekly pattern was particularly noticeable.

These observations support including weekday and calendar variables in the forecasting model.

### 3.6 Holiday Effects

Holiday periods showed substantial changes in recorded sales.

Examples from the midterm include:

| Holiday | Approximate change |
| --- | ---: |
| Labor Day | +24% |
| Thanksgiving | -29% |
| New Year's Day | -21% |
| Christmas | -100% |

Christmas closures are especially important because a simple recent-average model could mistake temporary closures for declining demand.

### 3.7 SNAP Benefit Days

SNAP benefit days were associated with different food-sales patterns across states.

| State | Foods sales difference on SNAP days |
| --- | ---: |
| Wisconsin | +30.4% |
| Texas | +15.9% |
| California | +10.6% |

These results support using state-specific SNAP indicators as forecasting features.

However, they represent observed associations and do not establish causality.

### 3.8 Selling Prices

Weekly sales were approximately **1.24 times normal** when an item's price was 10–20% below its typical level.

Sales were lower when prices were substantially above the normal level.

Price information was therefore included in the forecasting model.

However, these results do not prove that changing prices directly caused the observed differences in sales.

### 3.9 Main EDA Findings

Our EDA identified several challenges:

- Most item–store series have intermittent or lumpy sales.
- Demand differs across states and product categories.
- Weekdays, holidays, SNAP schedules, and prices are associated with sales variation.
- Historical sales levels change over time.
- The M5 dataset does not include actual inventory levels.

These findings shaped our forecasting methodology and the dashboard's inventory-planning design.

## 4. Forecasting Methodology

### 4.1 Chronological Data Splitting

We evaluated models using a time-based split:

| Period | Purpose |
| --- | --- |
| 2011–2014 | Training |
| 2015 | Validation and model selection |
| January–May 2016 | Final holdout testing |

The 2015 validation used rolling-origin backtesting with approximately 50 weekly forecast dates.

We evaluated three forecasting horizons:

- 7 days
- 14 days
- 28 days

This approach ensures that models are evaluated on future observations rather than using randomly mixed historical dates.

Demand-type classifications were also calculated using training-period information.

The midterm reported a leakage audit confirming that future sales were not used to generate forecasts.

### 4.2 Demand Classification

Demand behavior was classified using:

- **Average Demand Interval (ADI):** Measures how frequently nonzero sales occur.
- **Squared Coefficient of Variation (CV²):** Measures variation in positive sales quantities.

The midterm used thresholds of:

- ADI = 1.32
- CV² = 0.49

These measures were used to distinguish smooth, intermittent, erratic, and lumpy demand.

A separate no-sales group captured series without positive sales in the training history.

### 4.3 Forecasting Models

We compared several forecasting methods.

**Baseline models:**

- Zero Naive
- Weekly Seasonal Naive
- Recent Mean (7-day)
- Recent Mean (28-day control)

**Intermittent-demand models:**

- Croston
- Syntetos–Boylan Approximation (SBA)
- Teunter–Syntetos–Babai (TSB)

**Machine learning:**

- LightGBM

LightGBM was selected as the final forecasting model.

Its Tweedie objective was appropriate for nonnegative sales with many zero observations.

### 4.4 Feature Engineering

The LightGBM model used **28 features across six groups**.

| Feature Group | Examples |
| --- | --- |
| Recent sales | Rolling averages over 7, 14, 28, 56, and 365 days |
| Price | Current selling price, historical price comparisons, price changes |
| Item and store | Category, department, store, state, demand type |
| Calendar | Weekday, month, holidays, events, SNAP |
| Same-weekday history | Historical sales on corresponding weekdays |
| Forecast horizon | Days ahead |

Recent sales features accounted for approximately **89.3% of model gain**.

The 28-day historical average was particularly important.

Feature gain represents model reliance rather than the causal contribution of each variable.

### 4.5 Bias Correction

The initial LightGBM model tended to underpredict later demand.

Our midterm reported the following changes:

| Model adjustment | Signed bias |
| --- | ---: |
| Initial LightGBM | -4.4% |
| Quarterly retraining | -3.1% |
| Retraining with 13-week correction | -0.3% |

The correction used recent known forecasting errors by demand type, with an adjustment capped between 0.8 and 1.25 times the original prediction.

### 4.6 Evaluation Metrics

Our main metrics were:

**Weighted Absolute Percentage Error (WAPE)**

WAPE measures total absolute forecast error relative to actual demand.

Lower WAPE indicates better forecasting accuracy.

**Signed Forecast Bias**

Measures whether forecasts systematically overpredict or underpredict actual sales.

**Root Mean Squared Error (RMSE)**

Places more weight on large forecasting errors.

**Pinball Loss and Coverage**

Used to evaluate the quality of P10 and P90 prediction ranges.

Because we forecast at the daily item–store level, our WAPE values should not be directly compared with models evaluated only on aggregated weekly or store-level totals.

## 5. Forecasting Results

### 5.1 Model Performance on the 2016 Holdout

Our final LightGBM model performed better than the Recent Mean baselines across all three horizons.

| Forecast Horizon | LightGBM WAPE | Recent Mean (7d) | Recent Mean (28d) |
| --- | ---: | ---: | ---: |
| 7 days | **71.6%** | 74.6% | 73.9% |
| 14 days | **72.6%** | 75.8% | 74.7% |
| 28 days | **73.9%** | 77.3% | 75.9% |

The **seven-day horizon performed best** among the three evaluated horizons.

At seven days, LightGBM also achieved:

- RMSE: **2.040**, compared with 2.219 for Recent Mean.
- Signed bias: **-0.7%**, compared with -1.8% for Recent Mean.

The results indicate improved forecasting performance on historical holdout data, not proven improvements in real-world inventory outcomes.

### 5.2 Validation Performance

During 2015 validation:

| Model | 7-Day WAPE |
| --- | ---: |
| LightGBM final | 75.0% |
| TSB | 76.4% |
| Recent Mean (7d) | 77.5% |
| SBA | 85.9% |
| Croston | 87.2% |

LightGBM produced the lowest WAPE among these reported candidates.

### 5.3 Historical Example: January 1–7, 2016

The midterm presentation included a practical comparison following Christmas.

For the 15,245-series sample:

| Measure | Result |
| --- | ---: |
| Actual sales | 130,915 units |
| LightGBM forecast | Approximately 136,272 units |
| LightGBM error relative to actual | +4.1% |
| Recent Mean error relative to actual | -19.6% |

The Recent Mean baseline underpredicted demand partly because the previous week included Christmas Day, when stores were closed.

This demonstrates why calendar-aware forecasting can be more useful than simply averaging recent sales.

**This historical January 2016 forecast is also used in our current Streamlit demonstration.**

### 5.4 Forecasting Error by Demand Type

Our baseline error analysis found that:

- Intermittent and lumpy series contributed approximately **71.9% of total absolute error**.
- Foods contributed approximately **59.4% of total absolute error**.
- Hobbies had the highest category-level WAPE at approximately **104.7%**.
- High-volume series contributed approximately **62.4% of the total absolute error**.

These findings came from the Recent Mean baseline analysis, rather than a segmented decomposition of the final LightGBM model.

They highlight why both relative error and the total number of incorrectly forecast units matter.

### 5.5 Prediction Uncertainty

The forecasting work generated daily P10 and P90 estimates.

The midterm reported approximately **77.7% coverage** on 2016 daily sales observations, compared with the nominal 80% target.

Demand-type-based calibration was considered because a single pooled interval did not perform equally well across demand types.

**Important limitation:** Adding daily P10 and P90 values across seven days does not produce a statistically calibrated seven-day prediction interval.

The dashboard therefore treats summed daily bounds as illustrative scenarios.

The calibration method stored in the deployed forecast file should be verified against the final forecasting notebook before claiming that the dashboard uses the selected demand-type calibration.

## 6. Retrieval-Augmented Generation (RAG)

### 6.1 Objective

The RAG component was developed to provide historical evidence that helps explain forecast and inventory results.

Instead of relying entirely on a language model's general knowledge, the system retrieves relevant documents and provides them as supporting context.

### 6.2 Historical Evidence Sources

The midterm RAG work used several public sources:

- Federal Reserve Beige Book
- NOAA Storm Events Database
- SNAP-related government information
- Walmart SEC filings
- M5-derived historical evidence cards

The midterm reported approximately **7,333 passages from 4,796 documents**.

These passages were embedded using a BGE-small sentence-embedding model.

Documents were tagged with dates and relevant geographic information.

### 6.3 Preventing Future-Information Leakage

The RAG retrieval system filters evidence by the forecast decision date.

For example, when generating an explanation for January 2016, the system should not retrieve a document published later in 2016.

This helps avoid using historical information that would not have been available to a store manager at the time of the forecast.

### 6.4 Retrieval Evaluation

The midterm reported:

| Evaluation | Result |
| --- | --- |
| Questions returning the expected type of source | 13 of 15 |
| Future-dated documents in the reported leakage tests | 0 |
| Similarity score for a no-answer COVID-19 question | 0.68 |

The COVID-19 question demonstrated an important limitation: a high retrieval similarity score does not necessarily indicate that relevant evidence exists.

Therefore, a fixed similarity threshold alone is not sufficient to determine whether the system should answer a question.

### 6.5 Comparison of AI Approaches

The midterm compared four explanation methods:

- Baseline template
- RAG + LLM
- LangChain agent
- LLM without RAG

An early evaluation using three test orders found:

| Measure | Baseline | RAG + LLM | LangChain | LLM without RAG |
| --- | ---: | ---: | ---: | ---: |
| Answers completed | 3/3 | 3/3 | 2/3 | 3/3 |
| Reasons with citations | 67% | 84% | 57% | 0% |
| Average kinds of evidence used | 1.0 | 1.3 | 3.0 | 0 |

RAG + LLM provided more consistently cited explanations, while the LangChain approach retrieved more varied evidence but required more model calls.

These were small-sample midterm experiments, not production-level benchmarks.

### 6.6 Latest RAG Integration

Since the midterm, the Streamlit dashboard has been updated so Gemini receives:

1. Python-calculated forecasting results.
2. Inventory calculations based on uploaded or simulated stock.
3. Historical evidence retrieved through RAG.
4. The user's question and selected analysis scope.

Gemini then generates a natural-language explanation using this context.

The assistant does **not** independently calculate forecasts or change the numerical results.

Tyler has also shared an updated RAG notebook, `M5_RAG_agent_v3.ipynb`, which had not yet been uploaded and tested in the working application at the time of this documentation update.

## 7. Streamlit Dashboard

**Live dashboard:**  
https://ids-560-capstone-group-4-walmart.streamlit.app/

The current dashboard integrates precomputed LightGBM forecasts, inventory calculations, and Gemini + RAG explanations.

It contains four pages.

### 7.1 Executive Overview

The Executive Overview provides a summary of forecasting and inventory-planning results for a selected store.

Main features include:

- Seven-day demand summary
- Inventory action KPIs
- Estimated units to order
- Estimated order costs
- Items fully covered
- Items with potential shortages
- Store-level comparisons
- Prioritized inventory items

Users can also adjust simulation settings such as:

- Assumed unit cost
- Assumed selling price
- Assumed replenishment lead time

The **Top Priorities** table allows users to display the top **5, 10, 20, or 50 products**.

The table includes:

- Product ID
- Department
- Predicted demand gap
- Illustrative units to order
- Estimated order cost

Estimated costs are calculated using the assumed unit cost rather than verified Walmart purchasing costs.

### 7.2 Demand Forecasting

The Demand Forecasting page allows users to explore forecasts at the department and product levels.

The page is organized as follows:

1. Department-level forecast table
2. Department-level comparison chart
3. Department and product selection
4. Product forecast KPIs
5. Daily forecast table
6. Daily forecast visualization
7. Additional forecast analysis and evaluation

The daily forecasting view displays the LightGBM point forecast and P10/P90 estimates.

The current dashboard uses the historical January 1–7, 2016 forecast period.

### 7.3 Inventory Planning

The Inventory Planning page combines forecasted demand with user-provided or simulated starting inventory.

The page includes:

**Prioritized Inventory Action List**

Products are ranked according to the inventory-planning calculations.

**Today's Order List (Simulated)**

Shows illustrative order quantities, assumed lead times, and estimated costs.

**Department and Product Selection**

Allows users to select an individual product for further analysis.

**Inventory Overrides**

Users can manually change the assumed starting stock for individual products.

**Inventory Assumptions**

Displays the selected product's starting stock and seven-day predicted demand.

**Projected Inventory KPIs**

Summarizes illustrative shortages and stock projections.

**Selected Product Inventory**

Displays projected remaining inventory for an individual product.

**Stock Runway — Next 7 Days**

Users can choose which products to compare.

**Potential Shortage**

Displays shortage-related results under the selected inventory assumptions.

The inventory calculations assume no new replenishment unless otherwise stated.

They do not represent observed Walmart stockouts or validated order recommendations.

### 7.4 Inventory Upload

The dashboard supports two main inventory sources:

- Simulated inventory
- Uploaded inventory

Users can upload a CSV with the following columns:

```csv
store_id,item_id,on_hand
CA_1,FOODS_1_001,25
CA_1,FOODS_1_002,0
TX_1,HOUSEHOLD_1_001,12
```

If a product is missing from the uploaded inventory data, users can choose to:

1. Use simulated inventory for the missing item.
2. Assume the missing item has zero units available.

An explicitly entered inventory value of zero remains zero.

The dashboard also supports product-specific stock overrides.

**Important:** Missing inventory data does not necessarily mean zero inventory. The zero-stock fallback is a user-selected assumption.

### 7.5 Inventory Calculations

For each item–store series, the dashboard compares predicted demand with starting inventory.

The simplified demand-gap calculation is:

```python
positive_gap = max(0, predicted_7_day_demand - starting_stock)
```

Illustrative units to cover are calculated using:

```python
units_to_cover = ceil(positive_gap)
```

Estimated ordering cost is:

```python
estimated_order_cost = units_to_cover * assumed_unit_cost
```

These calculations are useful for demonstrating inventory decision support, but they are not validated replenishment optimization.

For example, the dashboard does not yet account for all real operational constraints such as supplier minimum order quantities, warehouse capacity, or verified replenishment lead times.

When aggregating across multiple items, we sum positive item-level shortages rather than simply subtracting total stock from total demand.

This matters because excess inventory for one product cannot automatically cover a shortage of another product.

### 7.6 Assumed Order Cost

The current dashboard uses a default assumed unit cost of **$5**, which users can change in Simulation Settings.

Although the M5 dataset includes historical selling prices, it does not provide Walmart's actual procurement costs.

Therefore, the estimated ordering expenses are illustrative.

Historical selling prices could potentially be included in a future revenue-related calculation, but they should not be treated as purchasing costs.

We plan to wait for sponsor and faculty feedback before deciding whether this functionality should be expanded.

## 8. AI Assistant Integration

### 8.1 Current Architecture

The current implementation connects forecast and inventory calculations with historical evidence retrieval and Gemini.

```text
M5 Sales + Calendar + Prices
              |
              v
      LightGBM Forecasting
              |
              v
   Precomputed Forecast CSV
              |
              v
    Python Forecast Calculations
              |
              +------------------------+
              |                        |
              v                        v
     Inventory Calculations      Dashboard KPIs
              |
              v
     Structured AI Context
              |
              +------------------------+
              |                        |
              v                        v
      Historical RAG Evidence    Gemini API
              |                        |
              +-----------+------------+
                          |
                          v
                Gemini Explanation
                          |
                          v
                    User Review
```

### 8.2 Automatic Analysis Scope

Users can manually select an analysis scope or allow the dashboard to identify one based on their question.

Available scopes include:

- Automatic (from question)
- Selected product
- Selected department
- Selected store
- All stores

The automatic selection uses predefined rules to interpret the question.

For example:

**Question:**

> Which stores have the highest predicted inventory shortages over the next seven days?

The dashboard selects the all-store analysis scope and calculates store-level results.

Gemini then explains those numerical results.

This feature uses rule-based question classification rather than a fully autonomous planning agent.

### 8.3 Store-Level Analysis

The dashboard can compare inventory shortages across all ten M5 stores.

The numerical results are calculated in Python and displayed in comparison tables before being included in the Gemini prompt.

In our manual testing, Gemini correctly identified the stores with the largest calculated shortages and matched the displayed numerical rankings.

### 8.4 Department-Level Analysis

Users can also ask questions about departments across multiple stores.

For example:

> Which departments have the highest predicted inventory shortages across all stores?

The system aggregates the relevant data and provides Gemini with the calculated department-level results.

### 8.5 Gemini Explanations

The updated assistant generally organizes responses into:

- Main finding
- Explanation or comparison
- Relevant historical context
- Monitoring suggestion

The dashboard also distinguishes between:

- Total item–store series
- Item–store series with calculated shortages

This was added after a previous Gemini response incorrectly interpreted the total number of products as the number experiencing shortages.

### 8.6 Historical Evidence

RAG passages are shown in an expandable section so users can inspect the supporting sources.

The assistant may retrieve historical reports, M5 evidence cards, and other dated documents.

However, retrieval similarity does not guarantee that the selected evidence is relevant.

Users should treat Gemini's explanations as decision-support information and verify important claims against the numerical tables and original sources.

### 8.7 Example Questions

Users can test the AI Assistant using questions such as:

- Which stores have the largest predicted inventory shortages over the next seven days?
- Which departments have the highest predicted inventory gaps across all stores?
- What is the forecast and starting stock for this selected product?
- What historical evidence could help explain Foods demand in Wisconsin?

The store- and department-ranking examples were tested manually and produced responses consistent with the displayed numerical results.

These checks do not constitute a full audit of every possible question or calculation.

## 9. Repository Structure

```text
IDS-560-Capstone-Group-4-walmart/
│
├── app.py
│   # Streamlit dashboard and page logic
│
├── analysis_scope.py
│   # Automatic analysis scope and aggregated summaries
│
├── inventory_bridge.py
│   # Uploaded inventory and fallback calculations
│
├── rag_backend.py
│   # Historical evidence retrieval used by the dashboard
│
├── dashboard_forecasts.csv
│   # Precomputed LightGBM forecasts
│
├── requirements.txt
│   # Python dependencies
│
├── forecasting/
│   # Forecasting notebooks and experiments
│
├── RAG/
│   # RAG development files and evidence
│
├── evaluation/
│   # Evaluation files
│
├── Development Updates/
│   # Development notes and progress updates
│
├── README.md
│
└── .gitignore
```

The dashboard files currently remain in the repository root because the working Streamlit deployment uses `app.py` as its entrypoint.

Moving those files into another folder would require updating the deployment entrypoint and relevant file paths.

## 10. Running the Dashboard

Install the required packages:

```bash
pip install -r requirements.txt
```

Start the dashboard:

```bash
streamlit run app.py
```

The deployed application uses Streamlit Community Cloud.

### Gemini API Configuration

The Gemini API key must be configured through Streamlit Secrets or another secure environment configuration.

Example:

```toml
GEMINI_API_KEY = "YOUR_API_KEY"
```

API keys should never be committed directly to the public GitHub repository.

The application uses precomputed forecasts and does not retrain LightGBM when a user opens the dashboard.

Historical RAG retrieval also requires the relevant evidence index files.

## 11. Current Limitations

### 11.1 Historical Dataset

The M5 dataset covers 2011–2016.

The dashboard demonstrates historical forecasting and inventory planning rather than making current Walmart forecasts.

### 11.2 Sales Versus Actual Demand

Recorded sales may not represent true customer demand if products were unavailable.

The dataset does not allow us to directly identify all stockout events or lost sales.

### 11.3 Missing Inventory Data

M5 does not provide actual inventory-on-hand records.

The dashboard therefore uses simulated inventory or quantities uploaded by users.

### 11.4 Procurement Cost

M5 includes historical selling prices but not verified procurement costs.

The default $5 unit cost is an assumption.

Estimated order costs should not be interpreted as Walmart's actual purchasing expenses.

### 11.5 Replenishment Constraints

The dashboard does not include verified:

- Supplier lead times
- Warehouse capacity
- Minimum order quantities
- Supplier purchasing costs
- Actual inventory replenishment records
- Observed stockout events

The current ordering calculations are illustrative.

### 11.6 Forecast Accuracy

LightGBM improved upon the evaluated baselines, but daily item–store forecasts still have substantial errors, especially for sparse demand.

### 11.7 Prediction Intervals

Daily P10/P90 coverage was below the nominal 80% target in the reported historical evaluation.

Summing daily prediction bounds does not generate a validated seven-day prediction interval.

### 11.8 RAG and Gemini

Gemini may misinterpret numerical information, overgeneralize from historical evidence, or produce explanations that need additional verification.

Automatic analysis scope is rule-based and may not correctly classify every question.

### 11.9 No Production Integration

The dashboard is not connected to Walmart's live inventory, warehouse, or purchasing systems.

It does not execute orders automatically.

### 11.10 Additional Validation Needed

Manual testing has confirmed selected dashboard functions, including store and department comparisons.

However, further testing is needed for:

- Inventory CSV uploads
- Missing inventory records
- Zero-inventory handling
- Manual stock overrides
- Cross-page calculation consistency
- Gemini responses
- RAG evidence relevance

## 12. Next Steps

The dashboard is currently functioning as an academic prototype.

For now, our priority is to wait for feedback from our sponsor and professor before deciding which additional features are necessary.

Potential follow-up tasks include:

1. Test uploaded inventory records with missing values and explicit zeros.
2. Verify inventory overrides and cross-page calculations.
3. Test more Gemini questions across products, departments, and stores.
4. Review RAG retrieval quality and unsupported claims.
5. Confirm the assumed cost calculations under different simulation settings.
6. Review whether historical selling prices should be added for revenue-related analysis.
7. Test Tyler's updated RAG notebook before integrating it into the dashboard.

Any further improvements should be guided by the project's academic requirements and sponsor feedback.

## 13. Project Documentation

This README combines the project's midterm methodology and results with the updated Streamlit dashboard implementation.

### Midterm Presentation

**Group 4 Midterm — Agentic AI for Retail Demand & Inventory Decision Support**

The midterm provides the basis for:

- Business problem and project objectives
- Dataset preparation
- Exploratory data analysis
- Model training and evaluation
- Forecasting performance
- Prediction uncertainty
- RAG methodology and initial evaluation

### Forecasting Notebooks

See the `forecasting/` directory for the forecasting implementation and related results.

### RAG Development

See the `RAG/` directory for the historical evidence retrieval development files.

### Dashboard Code

The dashboard implementation is maintained in:

- `app.py`
- `analysis_scope.py`
- `inventory_bridge.py`
- `rag_backend.py`
- `dashboard_forecasts.csv`

### Development History

See the `Development Updates/` folder for notes about dashboard improvements made after the midterm.

**Note:** Midterm model-evaluation results, manual dashboard testing, and proposed future work are separate categories of evidence and should not be treated as equally validated results.
