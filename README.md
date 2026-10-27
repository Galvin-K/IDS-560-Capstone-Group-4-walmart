# IDS-560-Capstone-Group-4-walmart-project
# Walmart M5 – Retail Demand Forecasting & Inventory Decision Support

**UIC IDS 560 Capstone Project | Group 4 | Fall 2026**

## Project Overview

This project develops an AI-assisted decision-support system to help retail managers make better inventory-planning decisions. Using historical Walmart M5 sales data, we forecast product demand, evaluate potential inventory shortages, and provide supporting business information through a Retrieval-Augmented Generation (RAG) assistant.

The project combines three main components:

1. **Demand Forecasting:** Predict future item-store sales using LightGBM and compare its performance with traditional forecasting methods.
2. **Inventory Planning:** Use demand forecasts and simulated or user-provided inventory levels to support replenishment decisions.
3. **AI Assistant:** Retrieve relevant historical evidence and provide contextual information through RAG.

The system is presented through an interactive Streamlit dashboard.

## Live Dashboard

**[SmartStock AI – Streamlit Dashboard](https://ids560-walmart-group4.streamlit.app/)**

The dashboard provides:

- Interactive store, department, and product selection
- LightGBM demand forecasts and prediction intervals
- Inventory-planning scenarios with adjustable starting stock
- Executive KPIs and forecasting visualizations
- RAG-based AI assistant for retrieving supporting information

## Dataset

The project primarily uses the public **M5 Forecasting Accuracy and M5 Forecasting Uncertainty** datasets, which contain historical Walmart sales information from 2011–2016.

The data includes daily item-store sales, product and store identifiers, pricing, calendar events, and SNAP indicators. An external weather dataset was also joined with the M5 data for additional analysis.

**Important limitation:** The M5 dataset does not provide actual inventory levels or stockout records. Inventory-planning scenarios therefore rely on simulated or user-provided values.

## Forecasting Methodology

The team evaluated several forecasting approaches, including Zero Naive, Weekly Seasonal Naive, Recent Mean, Croston, SBA, TSB, and LightGBM.

Models were evaluated using chronological validation and rolling-origin backtesting across 7-, 14-, and 28-day forecasting horizons.

Evaluation metrics include WAPE, MAE, RMSE, Forecast Bias, and Forecast Value Added (FVA). Residual-based prediction intervals were evaluated using coverage, interval width, and Pinball Loss.

### Key Results

| Model | 7-Day WAPE (2016 Test) |
|---|---:|
| LightGBM | 71.6% |
| Recent Mean | 74.6% |

LightGBM achieved the lowest forecasting error among the evaluated models, with the 7-day horizon performing best.

## RAG-Based AI Assistant

The RAG component retrieves information from historical documents and external sources, including NOAA Storm Events data, economic reports, and other relevant business information.

The team is evaluating multiple approaches, including Pure RAG, Pure LLM, RAG + LLM, and a LangChain-based agent.

Evaluation focuses on retrieval relevance, answer correctness, citation quality, and unsupported-answer handling.

The current assistant retrieves supporting evidence but is not yet fully integrated with numerical forecasting and inventory-planning calculations.

## Repository Structure

```text
IDS-560-Capstone-Group-4-walmart-dashboard/
│
├── app.py                    # Streamlit dashboard
├── rag_backend.py            # RAG backend integration
├── dashboard_forecasts.csv   # Precomputed forecasting results
├── requirements.txt          # Application dependencies
│
├── forecasting/
│   └── updated_M5_Final_Forecast.ipynb
│
├── RAG/                      # RAG development files
├── evaluation/               # Evaluation files
│
├── README.md
└── .gitignore
```

Additional forecasting, RAG, and evaluation files will be incorporated as development continues.

## Current Limitations

- Historical sales do not necessarily represent actual customer demand when products are out of stock.
- Inventory values are simulated or entered manually.
- Forecasting results are precomputed rather than generated through real-time model inference.
- The AI assistant is not yet fully connected to numerical forecasting and inventory calculations.
- The system is not connected to Walmart's operational databases or inventory systems.
- The prototype is intended for academic demonstration and evaluation, not production deployment.

## Future Development

The next development priorities include improving forecasting–RAG integration, expanding RAG evaluation, refining prediction intervals, investigating cold-start forecasting for new products, and improving inventory-planning functionality.

## Project Team

University of Illinois Chicago — IDS 560 Capstone, Group 4

**Industry Sponsor:** Walmart Anjali Garg

**Academic Supervisor:** Professor Fatemeh Sarayloo

---

*This project uses publicly available data and is developed for academic research and demonstration purposes. It is not an official Walmart production system.*
