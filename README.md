# 💧 Smart Water Quality Monitoring & Potability Prediction (SDG 6)

ML + web app that predicts whether a water sample is safe to drink from 9 readings
(pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity).

**Mapped SDG:** SDG 6 - Clean Water and Sanitation (secondary: SDG 3, SDG 11)

## Features
- Streamlit web app: single-sample prediction with confidence and guideline (WHO/BIS-style) range checks
- Batch prediction from a CSV with downloadable results
- Model dashboard: accuracy, precision, recall, F1, ROC-AUC, 5-fold CV, confusion matrix, feature importance

## Tech stack
Python, pandas, NumPy, scikit-learn (Random Forest, Logistic Regression), Matplotlib, Joblib, Streamlit

## Project structure
```
app.py            # Streamlit web app
train.py          # training + evaluation, saves models/ and images/
water_utils.py    # features, guideline limits, synthetic data generator
data/             # generated sample dataset
models/           # water_model.pkl, metrics.json
images/           # charts, QR code
```

## Run
```bash
pip install -r requirements.txt
python train.py                                    # synthetic data
python train.py --csv data/water_potability.csv    # or the Kaggle Water Potability dataset
streamlit run app.py
```

## Results (synthetic data, 80:20 split, 600 test samples)
| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 76.8% | 0.147 | 0.729 |
| Random Forest | 94.7% | 0.878 | 0.905 |

Results on real-world data will differ. This is a screening aid, not a replacement for lab testing
