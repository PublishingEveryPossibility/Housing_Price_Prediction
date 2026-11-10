# Housing Price Prediction

This workspace contains a small regression project for predicting median housing values using a tabular housing dataset.

## Included files
- `housing_regression_report_ready.ipynb` – notebook report for the regression workflow
- `src/housing_model.py` – model pipeline and evaluation logic
- `requirements.txt` – Python dependencies

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/housing_model.py
```

The script downloads the California housing dataset, trains a random forest regressor, and prints MAE, RMSE, and $R^2$ metrics.
