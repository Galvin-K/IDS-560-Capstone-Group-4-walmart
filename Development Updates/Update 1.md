# Walmart IDS 560 Capstone — Development Update

**Date:** October 27, 2026  
**Status:** Dashboard implementation working in demonstration; business assumptions and academic validation awaiting sponsor/faculty feedback.

## Summary of changes 

The earlier update described an assistant that could retrieve historical evidence but was not connected to numerical forecasting. That description is **out of date**.

- **Forecast-aware AI:** Python-calculated forecast and inventory summaries are passed to Gemini alongside historical RAG passages. Store-level and department-level questions have produced numeric explanations aligned with on-screen comparison tables in manual tests.
- **Automatic question scope:** User can select a scope explicitly or use Automatic to identify product, department, store, and cross-store questions. Automatic classification is rule-based and should be manually checked for ambiguous questions.
- **Inventory sources:** Simulated stock or a user-uploaded `store_id,item_id,on_hand` CSV. Missing uploaded quantities can fall back to simulated stock or be assumed zero. Manual product overrides are supported.
- **Executive Overview:** Restored summary and inventory KPI design, simulation assumptions, action prioritization, 5/10/20/50 selectable priority rows, and illustrative order costs.
- **Demand Forecasting:** Department summary table/chart first, then page-specific product selection, product KPIs, daily table and chart.
- **Inventory Planning:** Action list, illustrative order list, selection/override, stock assumptions, KPIs, product graph, chosen-products runway, and potential-shortage section.
- **Presentation:** Compact KPI cards, structured comparison tables, more focused Gemini explanations, and collapsible historical evidence.

## Confirmed manual demonstration checks

- New Streamlit deployment runs from the active GitHub repository rather than the older dashboard repository.
- Gemini API configured in the new Streamlit app's Secrets; generation has succeeded.
- A store-ranking question was interpreted as cross-store analysis, and its displayed numeric order matched the computed table.
- A department comparison question returned department-level forecasts and gaps.
- A prior error interpreting total item-store series as affected items was corrected by exposing separate `total_item_series` and `item_series_with_shortage` fields.

These are manual UI observations, **not** complete regression tests or an independent audit of all calculations.

## Known limitations and outstanding decisions

### Cost assumption — await sponsor/professor response

M5 includes **historical selling prices**, but **not actual procurement cost**. The current dashboard assumes a configurable default unit cost of **$5** for illustrative order-cost calculations. We should not call it a verified Walmart cost or replace it with selling price without redefining the metric. The sponsor/professor may advise whether to retain the current assumption, add user-provided procurement costs, or join M5 historical selling prices for a separate revenue-oriented measure.

### Other limitations

- Forecasts are historical/precomputed and cover the included 50% modeling sample.
- User-supplied or simulated inventory is not observed live Walmart stock.
- No validated stockout, inbound replenishment, supplier lead-time, or order-policy data.
- Summed daily P10/P90 values are illustrative scenarios, not calibrated multi-day intervals.
- RAG passage relevance is imperfect; Gemini can overgeneralize, duplicate context, or misinterpret retrieved statistics.
- No production data pipeline or autonomous order execution.

## Suggested validation before next presentation

1. Confirm all four pages load after the documentation-only commit.
2. Upload a tiny CSV with a valid `on_hand=0`, a positive quantity, and a missing item-store pair. Verify both fallback choices and that zero stays zero.
3. Confirm page-specific product/department selectors and stock override values are consistent with the action and runoff views.
4. Ask Gemini one store-level and one department-level question; compare its figures directly against the numerical table.
5. Check that changing the assumed unit cost changes illustrative order-cost amounts but not forecast demand or units to order.
6. Remove any temporary verbose Gemini API error display from the public app if it remains.


## RAG agent notebook and possible next steps

Tyler has developed a separate LightGBM-based RAG agent notebook (`M5_RAG_agent_v2_lightgbm.ipynb`). It now includes an illustrative order-up-to rule based on forecast demand, lead time, review period, and safety stock, plus flags for questionable zero-sales patterns. The notebook also compares four explanation options: a template, RAG + LLM, a LangChain agent, and LLM without RAG. Its evaluation checks whether order quantities are preserved, whether claims are supported and cited, and whether future-dated evidence enters the answer.

The working Streamlit dashboard already combines forecast/inventory numbers with historical RAG and Gemini, but it **does not yet use Tyler's more detailed ordering rule**. She is still testing the newer work, and the updated chunks/index files have not been added to the deployed app. We should review those files and test compatibility before changing the dashboard.

### New-product forecasting idea

New products may not have enough sales history for the current forecasting features. One idea is to estimate their initial demand from similar products grouped by **category, price, and region**. As sales data come in for the new product, we could gradually use more of its own history. When there are too few comparable products, we could broaden the matching criteria. This will take some brainstorming and trial-and-error testing; historical cold-start simulations could help us compare different grouping rules. **Not implemented yet.**

### RAG retrieval quality

Selecting the top K passages does not mean that every returned passage is actually relevant. Options worth testing include tighter metadata filters, similarity thresholds, reranking, and evaluation with no-answer questions. A threshold alone is not enough: our earlier test retrieved a plausible match for a question that did not have a relevant historical source. **These are proposed improvements, not confirmed deployment changes.**

## Pending external feedback

- **Sponsor:** Utility of assumed unit cost versus selling-price/revenue metrics; preferred action priorities; decision-support expectations.
- **Professor:** Appropriate limitations, metric terminology, model-evaluation requirements, and what constitutes sufficient validation for the capstone.

