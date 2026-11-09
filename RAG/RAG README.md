# M5 Inventory Agent (RAG + LLM), v4

IDS 560, Group 4. For an item, store and date, the agent:
- computes an order from the M5 forecasting model (LightGBM),
- checks it for warning signs,
- finds dated evidence that explains it,
- has a language model write the recommendation for a store manager, with every reason cited.

The order quantity always comes from the forecast and the ordering rule. The language model only explains it.

## Run order

**1 → 2 → 4 → 3 → 5.** Notebook 4 picks the new-item method that notebook 3 uses. Running 3 before 4 also works, because the default method is the last winner.

| Notebook | What it does | Run time | Model calls |
|---|---|---|---|
| **1. Evidence library** | Downloads public documents (NOAA storms, Beige Book, Walmart SEC filings, SNAP policy). Builds evidence cards from the M5 data (seasonality, events, SNAP, weather, prices, demand spikes, revenue concentration, forecast accuracy). Splits everything into dated passages and embeds them. Includes a **passage viewer**: filterable tables, counts, the full text of any passage, a CSV export and a 2-D map of the embeddings. | A few minutes (15–30 the first time, while downloading) | none |
| **2. Retrieval tests** | Searches the library **as of a date**: metadata filters, meaning + keyword search, then a reranker. Runs 56 test questions, compares retrieval methods and checks that no future document is ever returned. Sets the "no relevant evidence" cut-off. | A few minutes | none |
| **3. Agent** | 10 scenario orders: forecast, segment, flags, the hold rule, new-item forecast, confidence. `recommend()` explains an order and `ask()` answers a manager's question; forecast questions use the **forecast-data lookup**. Produces a **validation matrix** for both. | 30+ minutes, depending on the free model | about 30 |
| **4. Evaluation** | **New-item simulation** (chooses the method), **confidence check** (does Low mean bigger errors?) and **order backtest** (did orders cover actual demand?). | About 10 minutes | none |
| **5. Benchmark** | Precision, recall, MRR, nDCG, F1, AUC and coverage for retrieval, no-answer detection, answers, flags, confidence, orders and new items, each against simple baselines. | 10–15 minutes | none |

## Files and folders

- **`RAG FILES/` on Drive** holds everything the notebooks read and write:
  - `rag_common.py`, the shared code
  - `raw/`, the download cache
  - `walmart_docs/`, optional hand-downloaded Walmart documents
  - all outputs, whose names end in `_v4`
- **`rag_common.py`** is the code used by more than one notebook: paths, data loading, `retrieve()`, forecasts, flags, new items, confidence, `recommend()`, `ask()`, the forecast-data lookup and the 56 test questions. Each notebook starts with `import rag_common as rc`. Change a setting with `rc.NAME = value`. The Streamlit app can import the same file.
- **`settings_v4.json`** holds values one notebook chooses and the others use: the relevance cut-off (notebook 2) and the new-item method (notebook 4).
- **Inputs:**
  - `M5 EDA/datasets/`: `calendar.csv`, `sales_train_evaluation.csv`, `sell_prices.csv`, `weather_by_state_day.csv`
  - `M5 Forecasting/M5_Forecasting_Outputs/`: `forecast_cube_lightgbm_50.npz` (forecasting notebook cell 16.19), plus `series_segments_50.csv` and `lightgbm_error_by_segment_50.csv` (cell 16.18)

## Main outputs (in `RAG FILES/`)

| File | From |
|---|---|
| `chunks_v4.pkl`, `index_v4.pkl`: the passages and their embeddings | 1 |
| `library_passages_v4.csv`, `embedding_map_v4.html`: the library, to browse | 1 |
| `retrieval_test_results_v4.csv`, `retrieval_methods_v4.csv` | 2 |
| `scenario_orders_v4.csv`, `agent_validation_v4.csv`, `question_validation_v4.csv`, `validation_matrix_v4.png`, `validation_matrix_questions_v4.png` | 3 |
| `new_item_simulation_v4.csv`, `confidence_check_v4.csv/.png`, `order_backtest_v4.csv/.png` | 4 |
| `benchmark_summary_v4.csv`, plus one CSV and chart per part | 5 |

## Things to know

- **No look-ahead.** Every decision uses only data and documents dated on or before the decision date.
- **No inventory data in M5.** Stock on hand, lead time and service level are assumptions. Questions about shortages and excess use an assumed stock of 7 days of recent sales, and the answer has to say so.
- **Free model limits.** OpenRouter's free tier allows about 50 requests a day, or 1,000 after a one-time $10 credit. Only notebook 3 uses the model, so if it hits the limit, rerun just notebook 3. Add `OPENROUTER_API_KEY` in Colab's Secrets panel.
- **Benchmark labels are automatic.** A passage counts as relevant if it matches a test's expected document type and terms. That's an approximation, not a hand label.
