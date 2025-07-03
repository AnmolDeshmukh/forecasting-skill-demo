# Forecasting Skill Demonstration

This project demonstrates a 24-hour load forecasting pipeline using both baseline and machine learning (Random Forest) approaches.
---

## Project Structure

```
forecasting-skill-demo/
├── data/
│   └── load_data.csv
├── notebook.ipynb
├── plots/
├── utils.py
├── README.md
└── requirements.txt
```

---

## Objectives

- Clean and preprocess time-series load data
- Implement baseline forecasting using historical patterns
- Develop and evaluate an advanced ML model
- Visualize and compare forecasts
- Package and document the pipeline

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/AnmolDeshmukh/forecasting-skill-demo.git
cd forecasting-skill-demo
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Open the notebook
```bash
jupyter notebook notebook.ipynb
```

---

## Data Description

- **File**: `load_data.csv`
- **Frequency**: Hourly
- **Columns**:
  - `timestamp`: Datetime object
  - `load_kW`: Power load in kilowatts

Includes missing values, outage periods, and outliers to simulate real-world noise.

---

## Project Steps

1. **Data Preprocessing**
   - Interpolate missing values
   - Detect and fix outages (zero-load)
   - Remove outliers (IQR)
   - Engineer time-based features

2. **Baseline Forecast**
   - Forecast using previous day's values
   - Compute MAE, RMSE, MAPE

3. **Advanced Forecast (Random Forest)**
   - Lag features used: t-1, t-2, t-3
   - Predict next 24 hours as a batch
   - Evaluate and compare to baseline

4. **Evaluation & Visualization**
   - Error metrics for both models
   - Forecast and residual plots

---

## Results

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| Baseline | ~5.46 | ~6.72 | ~17.69% |
| Random Forest | ~3.88 | ~4.99 | ~12.41% |

---

## Notes

- Future enhancements could include neural networks, hyperparameter tuning, or model ensembling.
- All code is organized into modular sections for readability.
- Additional utility functions can be moved to `utils.py` for reuse.

---

## Contact

Created by [Anmol Deshmukh](https://github.com/anmolDeshmukh/)
