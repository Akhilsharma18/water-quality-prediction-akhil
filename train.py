"""Train, evaluate and save the potability model.
  python train.py                       # synthetic data
  python train.py --csv data/water_potability.csv   # Kaggle dataset
"""
import argparse, json, joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)
from water_utils import FEATURES, make_synthetic

def main(csv=None):
    df = pd.read_csv(csv) if csv else make_synthetic()
    if not csv:
        df.to_csv("data/water_samples_synthetic.csv", index=False)
    X, y = df[FEATURES], df["Potability"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    models = {
        "Logistic Regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                             LogisticRegression(max_iter=1000)),
        "Random Forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestClassifier(n_estimators=300, random_state=42)),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    metrics = {}
    for name, m in models.items():
        m.fit(Xtr, ytr); p = m.predict(Xte)
        metrics[name] = {
            "accuracy": accuracy_score(yte, p), "precision": precision_score(yte, p),
            "recall": recall_score(yte, p), "f1": f1_score(yte, p),
            "roc_auc": roc_auc_score(yte, m.predict_proba(Xte)[:, 1]),
            "cv_accuracy_mean": cross_val_score(m, X, y, cv=cv).mean(),
        }
        print(name, {k: round(v, 3) for k, v in metrics[name].items()})
    best = models["Random Forest"]
    joblib.dump(best, "models/water_model.pkl")
    imp = pd.Series(best[-1].feature_importances_, index=FEATURES).sort_values()
    json.dump({"metrics": metrics, "importance": imp.to_dict(), "n_test": len(yte)},
              open("models/metrics.json", "w"), indent=2)
    ConfusionMatrixDisplay(confusion_matrix(yte, best.predict(Xte)),
                           display_labels=["Unsafe", "Potable"]).plot(cmap="Blues")
    plt.title("Random Forest - Confusion Matrix"); plt.tight_layout()
    plt.savefig("images/confusion.png", dpi=150); plt.close()
    imp.plot.barh(color="#0a7ea4"); plt.title("Feature Importance"); plt.tight_layout()
    plt.savefig("images/importance.png", dpi=150); plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--csv"); main(ap.parse_args().csv)
