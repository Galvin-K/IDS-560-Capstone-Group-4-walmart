# Walmart IDS 560 Capstone — Development Update

**Date:** October 27, 2026  
**Status:** Dashboard implementation working in demonstration; business assumptions and academic validation awaiting sponsor/faculty feedback.

## Summary of changes 

The earlier update described an assistant that could retrieve historical evidence but was not connected to numerical forecasting. That description is **out of date**.


- **Forecasting AI integration:** Connected Gemini to our LightGBM forecasting and inventory calculations, alongside historical evidence retrieved through RAG. The assistant can now answer questions about individual products, departments, categories, states, and multiple stores. We tested several comparisons and confirmed that Gemini's numerical results matched the dashboard tables.

- **Automatic question scope:** Improved the automatic analysis scope so the AI Assistant no longer depends on the store or product selected in the sidebar. It can identify specific products, stores, states, and categories from the user's question. For example, users can compare TX_1 and CA_1 or analyze Foods demand across California stores. Users can also manually select an analysis scope if needed. The classification is rule-based, so some questions may still require manual checking.

- **Inventory sources:** Users can choose estimated inventory or upload their own CSV containing `store_id`, `item_id`, and `on_hand`. Estimated inventory is generated using predefined assumptions, while demand forecasts come from LightGBM. If inventory values are missing from the uploaded file, users can choose to use estimated stock or assume zero inventory. Individual product inventory levels can also be manually adjusted.

- **Executive Overview:** Kept the original inventory summary and KPI layout, including inventory assumptions, store comparisons, action prioritization, and estimated order costs. Users can now choose to display the top 5, 10, 20, or 50 priority items using a dropdown menu. Estimated order costs use an adjustable assumed unit cost, currently set to $5 by default.

- **Demand Forecasting:** Reorganized the page to display the department-level forecast table and graph first, followed by department/product selection, product KPIs, and daily forecasting results. Added interactive product selection and a heatmap that allows users to inspect demand patterns for individual products.

- **Inventory Planning:** Reorganized the page to show the prioritized inventory action list and estimated order list first, followed by department/product selection, inventory overrides, inventory assumptions, projected inventory KPIs, product-level graphs, a customizable seven-day stock runway, and potential shortage analysis. Users can choose which products to display in the stock runway instead of relying on automatically selected products.

- **Dashboard presentation:** Updated the header with the Walmart logo and simplified the title to "Demand & Inventory Planner." Removed the technical subtitle and footer, replaced several references to simulated inventory with estimated inventory, and added an expandable "About the Forecast & Data" section. The dashboard now uses cleaner KPI cards, structured comparison tables, shorter Gemini explanations, and collapsible historical evidence.


- **Walmart logo:** Added the Walmart logo beside the "Demand & Inventory Planner" title to improve the dashboard's appearance and make it more consistent with Walmart's branding.

## Confirmed manual demonstration checks

- New Streamlit deployment runs from the active GitHub repository rather than the older dashboard repository. (https://ids-560-capstone-group-4-walmart.streamlit.app/)
- Gemini API configured in the new Streamlit app's Secrets; generation has succeeded.
- A store-ranking question was interpreted as cross-store analysis, and its displayed numeric order matched the computed table.
- A department comparison question returned department-level forecasts and gaps.
- A prior error interpreting total item-store series as affected items was corrected by exposing separate `total_item_series` and `item_series_with_shortage` fields.


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

## Planned Validation 

1. Confirm all four pages load after the documentation-only commit.
2. Upload a tiny CSV with a valid `on_hand=0`, a positive quantity, and a missing item-store pair. Verify both fallback choices and that zero stays zero.
3. Confirm page-specific product/department selectors and stock override values are consistent with the action and runoff views.
4. Ask Gemini one store-level and one department-level question; compare its figures directly against the numerical table.
5. Check that changing the assumed unit cost changes illustrative order-cost amounts but not forecast demand or units to order.
6. Remove any temporary verbose Gemini API error display from the public app if it remains.


### New-product forecasting idea

New products may not have enough sales history for the current forecasting features. One idea is to estimate their initial demand from similar products grouped by **category, price, and region**. As sales data come in for the new product, we could gradually use more of its own history. When there are too few comparable products, we could broaden the matching criteria. This will take some brainstorming and trial-and-error testing; historical cold-start simulations could help us compare different grouping rules. **Not implemented yet.**

### RAG retrieval quality

Selecting the top K passages does not mean that every returned passage is actually relevant. Options worth testing include tighter metadata filters, similarity thresholds, reranking, and evaluation with no-answer questions. A threshold alone is not enough: our earlier test retrieved a plausible match for a question that did not have a relevant historical source. **These are proposed improvements, not confirmed deployment changes.**

## Pending external feedback

- **Sponsor:** Utility of assumed unit cost versus selling-price/revenue metrics; preferred action priorities; decision-support expectations.
- **Professor:** Appropriate limitations, metric terminology, model-evaluation requirements, and what constitutes sufficient validation for the capstone.

