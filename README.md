# Electricity Market Intelligence Dashboard
### Short-Term Price Forecasting · IEX India & HPX India
*Internship Project — Pari Kulkarni & Hritika Das · June 2026*

---

## Project Overview
An end-to-end ML pipeline that ingests publicly available data from the Indian Energy Exchange (IEX) and Hindustan Power Exchange (HPX), performs EDA, trains price-forecasting models, and serves results through an interactive Streamlit dashboard.

## Repository Structure
```
iex-hpx-dashboard/
├── data/
│   ├── raw/
│   │   ├── iex/          # Downloaded IEX DAM/RTM reports (CSV/XLS)
│   │   └── hpx/          # Downloaded HPX trade reports
│   └── processed/        # Cleaned, merged master datasets
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_eda.ipynb
│   └── 03_ml_pipeline.ipynb
├── src/
│   ├── ingestion/        # Data loading & parsing scripts
│   ├── eda/              # EDA helper functions
│   ├── features/         # Feature engineering
│   ├── models/           # Model training & evaluation
│   └── dashboard/        # Streamlit app
├── reports/              # Final PDF report
├── slides/               # Presentation slides
├── tests/                # Unit tests
├── requirements.txt
├── .gitignore
└── README.md
```

## Quickstart
```bash
# 1. Clone and set up environment
git clone https://github.com/<your-username>/iex-hpx-dashboard.git
cd iex-hpx-dashboard
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Download data (see notebooks/01_data_ingestion.ipynb for instructions)

# 3. Run the full pipeline
python src/ingestion/load_data.py
python src/features/engineer.py
python src/models/train.py

# 4. Launch the dashboard
streamlit run src/dashboard/app.py
```

## Data Sources
| Exchange | URL | Segments | History |
|----------|-----|----------|---------|
| IEX | [iexindia.com](https://www.iexindia.com) | DAM, RTM, GTAM | 2008– |
| HPX | [hpxindia.com](https://www.hpxindia.com) | DAM, RTM | Jul 2022– |

> **Note:** All data is downloaded manually from public market-data sections. No login or paid API is required.

## Models
| Model | Role | Status |
|-------|------|--------|
| Linear Regression | Baseline | ✅ Implemented |
| Random Forest | Primary | ✅ Implemented |
| XGBoost | Stretch goal | 🔲 Optional |

## Evaluation Metrics
- MAE · RMSE · R² — reported for train and test splits.

## Team
- **Pari Kulkarni** — Data engineering, feature engineering, modelling
- **Hritika Das** — EDA, visualisation, dashboard, reporting
