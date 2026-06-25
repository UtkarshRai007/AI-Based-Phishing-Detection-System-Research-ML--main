"""Generate confusion matrix and correlation matrix for the phishing dataset.

Outputs (saved next to dataset):
 - confusion_matrix.csv
 - confusion_matrix.png
 - correlation_matrix.csv
 - correlation_matrix.png

Also prints the numeric matrices to stdout.
"""
from pathlib import Path
import re
import pickle
import joblib
import warnings

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import xgboost as xgb
except Exception:
    xgb = None


BASE = Path(__file__).resolve().parent
TEST_CSV = BASE / 'phishing_test_20.csv'
MODEL_PKL = BASE / 'phishing_model.pkl'


def extract_features_from_url(s: pd.Series) -> pd.DataFrame:
    s = s.fillna("").astype(str)
    url_length = s.str.len()
    lower = s.str.lower()
    has_https = lower.str.contains('https') | lower.str.contains('https://')
    num_dots = s.str.count(r"\.")
    has_at = s.str.contains('@')
    has_dash = s.str.contains('-')

    ipv4_re = re.compile(r"(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)")
    has_ip = s.apply(lambda x: 1 if ipv4_re.search(x) else 0)

    df = pd.DataFrame({
        'url_length': url_length.astype(int),
        'has_https': has_https.astype(int),
        'num_dots': num_dots.astype(int),
        'has_at': has_at.astype(int),
        'has_dash': has_dash.astype(int),
        'has_ip': has_ip.astype(int),
    })
    return df


def load_model(path: Path):
    path_str = str(path)
    # joblib
    try:
        return joblib.load(path_str)
    except Exception:
        pass
    try:
        with open(path_str, 'rb') as f:
            return pickle.load(f)
    except Exception:
        pass
    # try native xgb
    if xgb is not None and path.suffix in ('.model', '.bin', '.json'):
        try:
            booster = xgb.Booster()
            booster.load_model(path_str)

            class Wrapper:
                def __init__(self, booster):
                    self.booster = booster

                def predict(self, X):
                    d = xgb.DMatrix(X)
                    p = self.booster.predict(d)
                    if p.ndim == 1:
                        return (p > 0.5).astype(int)
                    return np.argmax(p, axis=1)

            return Wrapper(booster)
        except Exception:
            pass
    raise ValueError(f"Could not load model from {path}")


def main():
    print('Loading test CSV:', TEST_CSV)
    df = pd.read_csv(TEST_CSV)

    # detect label column
    label_col = None
    for c in ('Label','label','target','is_phishing','phishing'):
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        label_col = df.columns[-1]

    # Standardize label to 0/1
    y_raw = df[label_col]
    if y_raw.dtype == object:
        y = y_raw.str.lower().map({'good':0,'legitimate':0,'0':0,'bad':1,'1':1}).fillna(0).astype(int)
    else:
        y = y_raw.astype(int)

    # Extract features
    if 'URL' in df.columns:
        url_col = 'URL'
    elif 'url' in df.columns:
        url_col = 'url'
    else:
        # assume first column
        url_col = df.columns[0]

    X_features = extract_features_from_url(df[url_col])

    # Correlation matrix between features and label
    corr_df = X_features.copy()
    corr_df['label'] = y
    corr = corr_df.corr(method='pearson')

    # Save correlation matrix
    corr_out = BASE / 'correlation_matrix.csv'
    corr.to_csv(corr_out)
    print('\nCorrelation matrix (features vs label):')
    print(corr[['label']].to_string())

    # Load model and predict
    print('\nLoading model:', MODEL_PKL)
    model = load_model(MODEL_PKL)

    # Align columns if model has feature_names_in_
    X_for_pred = X_features.copy()
    try:
        if hasattr(model, 'feature_names_in_'):
            expected = list(model.feature_names_in_)
            X_for_pred = X_for_pred.reindex(columns=expected, fill_value=0)
        elif hasattr(model, 'get_booster'):
            try:
                booster = model.get_booster()
                fn = booster.feature_names
                if fn:
                    X_for_pred = X_for_pred.reindex(columns=fn, fill_value=0)
            except Exception:
                pass
    except Exception:
        pass

    # Predict
    try:
        y_pred = model.predict(X_for_pred)
    except Exception:
        y_pred = model.predict(X_for_pred.values)

    # If predicted probabilities, convert
    y_pred = np.array(y_pred)
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)
    elif y_pred.dtype.kind == 'f':
        y_pred = (y_pred > 0.5).astype(int)

    # Confusion matrix
    labels = [0,1]
    cm = confusion_matrix(y, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=['true_0','true_1'], columns=['pred_0','pred_1'])
    cm_out = BASE / 'confusion_matrix.csv'
    cm_df.to_csv(cm_out)

    print('\nConfusion matrix (rows=true, cols=pred):')
    print(cm_df.to_string())

    # Save heatmaps
    sns.set(font_scale=1.1)
    plt.figure(figsize=(5,4))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_xticklabels(['0','1'])
    ax.set_yticklabels(['0','1'])
    plt.title('Confusion Matrix for Phishing URL Detection')
    plt.tight_layout()
    plt.savefig(BASE / 'confusion_matrix.png', dpi=200)
    plt.close()

    plt.figure(figsize=(6,5))
    ax = sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdBu', center=0)
    plt.title('Correlation Matrix (features + label)')
    plt.tight_layout()
    plt.savefig(BASE / 'correlation_matrix.png', dpi=200)
    plt.close()

    # Print classification report
    print('\nClassification report:\n')
    print(classification_report(y, y_pred, target_names=['good(0)','bad(1)'], zero_division=0))

    print('\nSaved files:')
    print(' -', cm_out)
    print(' -', BASE / 'confusion_matrix.png')
    print(' -', corr_out)
    print(' -', BASE / 'correlation_matrix.png')


if __name__ == '__main__':
    main()
