# Enterprise Pricing & Elasticity Engine

## Overview

Adaptive structural pricing engine built in Python to estimate product-level elasticity, simulate demand response, and optimize standalone and bundled pricing strategies.

This project demonstrates applied econometrics, pricing strategy modeling, and data-driven decision support.

---

## Key Features

- Adaptive econometric specification based on sample size
- Structural log-log demand estimation
- Statistical elasticity diagnostics
- TAM-based market simulation
- Bundle strategy optimization
- Profit curve visualization
- Trust region extrapolation control

---

## Technical Stack

- Python
- Pandas
- NumPy
- Statsmodels
- Matplotlib
- Streamlit

---

## Modeling Approach

The elasticity model automatically adapts its specification:

- Small datasets → Basic log-log model
- Medium datasets → Trend-adjusted model
- Large datasets → Fixed effects model

This ensures statistical validity and prevents overfitting.

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py