"""Evaluate a trained phishing detection model.

Loads `phishing_dataset.csv`, splits into train/test (80/20), loads a trained
model from `phishing_model.pkl` (or other path), runs predictions on the test
set, and prints Accuracy, Classification Report, and a Confusion Matrix
visualized as a seaborn heatmap.

Usage:
    python evaluate_model.py

You can pass custom paths by editing the constants below or by importing the
functions in another script.
"""
from pathlib import Path
import argparse
import pickle
import joblib
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import xgboost as xgb
except Exception:
    xgb = None

# Defaults (assume script lives next to dataset and model)
DEFAULT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = DEFAULT_DIR / "phishing_dataset.csv"
DEFAULT_MODEL = DEFAULT_DIR / "phishing_model.pkl"


def load_dataset(path: Path):
    """Load dataset CSV into features X and labels y.

    Assumes the CSV contains a column named 'label' (or 'Label') for the
    target. If not, the last column will be used as the label.
    """
    df = pd.read_csv(path)

    # Normalize label column name
    label_col = None
    for candidate in ("label", "Label", "target", "is_phishing", "phishing"):
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        # fallback: last column
        label_col = df.columns[-1]
        warnings.warn(f"No standard label column found. Using last column '{label_col}' as label.")

    X = df.drop(columns=[label_col])
    y = df[label_col]

    return X, y


def load_model(path: Path):
    """Load a model from disk, handling common formats (pickle, joblib, XGBoost).

    Returns an object with a `predict` method that accepts a 2D array or DataFrame.
    """
    path_str = str(path)
    # Try joblib first
    try:
        model = joblib.load(path_str)
        return model
    except Exception:
        pass

    # Try pickle
    try:
        with open(path_str, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception:
        pass

    # Try XGBoost native model
    if xgb is not None and path.suffix in (".model", ".bin", ".json"):
        try:
            booster = xgb.Booster()
            booster.load_model(path_str)

            class XGBWrapper:
                def __init__(self, booster):
                    self.booster = booster

                def predict(self, X):
                    # XGBoost Booster expects DMatrix
                    dmat = xgb.DMatrix(X)
                    preds = self.booster.predict(dmat)
                    # If binary prob, convert to label 0/1 using 0.5
                    if preds.ndim == 1:
                        return (preds > 0.5).astype(int)
                    return np.argmax(preds, axis=1)

            return XGBWrapper(booster)
        except Exception:
            pass

    raise ValueError(f"Could not load model from {path}")


def get_model_feature_names(model):
    """Try to extract feature names the model expects.

    Returns a list of feature names or None if not available.
    """
    # scikit-learn-style
    if hasattr(model, "feature_names_in_"):
        try:
            return list(model.feature_names_in_)
        except Exception:
            pass

    # XGBoost sklearn wrapper
    if hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            if hasattr(booster, "feature_names"):
                return list(booster.feature_names)
        except Exception:
            pass

    # native booster wrapped earlier (our XGBWrapper)
    if hasattr(model, "booster") and hasattr(model.booster, "feature_names"):
        try:
            return list(model.booster.feature_names)
        except Exception:
            pass

    return None


def extract_url_features(url_series: pd.Series) -> pd.DataFrame:
        """Create numeric features from a URL series.

        Features created (matching common phishing feature engineering):
            - url_length: length of URL string
            - has_https: 1 if 'https' appears in URL, else 0
            - num_dots: number of '.' in URL
            - has_at: 1 if '@' in URL else 0
            - has_dash: 1 if '-' in URL else 0
            - has_ip: 1 if an IPv4 address appears in the URL else 0
        """
        s = url_series.fillna("").astype(str)
        url_length = s.str.len()
        lower = s.str.lower()
        has_https = lower.str.contains("https") | lower.str.contains("https://")
        num_dots = s.str.count(r"\.")
        has_at = s.str.contains("@")
        has_dash = s.str.contains("-")

        # simple IPv4 regex
        import re
        ipv4_re = re.compile(r"(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)")
        has_ip = s.apply(lambda x: 1 if ipv4_re.search(x) else 0)

        df = pd.DataFrame({
                "url_length": url_length.astype(int),
                "has_https": has_https.astype(int),
                "num_dots": num_dots.astype(int),
                "has_at": has_at.astype(int),
                "has_dash": has_dash.astype(int),
                "has_ip": has_ip.astype(int),
        })
        return df


def evaluate(y_true, y_pred, labels=None):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return acc, report, cm


def plot_confusion_matrix(cm, labels, title="Confusion Matrix for Phishing URL Detection"):
    plt.figure(figsize=(6, 5))
    sns.set(font_scale=1.1)
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                     xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    plt.tight_layout()
    return plt


def main(data_path: Path, model_path: Path, test_size=0.2, random_state=42, show_plot=True):
    print("Loading dataset:", data_path)
    X, y = load_dataset(data_path)

    # If labels are non-numeric, keep the mapping for later
    labels = None
    if y.dtype == object or y.dtype.name == 'category':
        labels = list(pd.Categorical(y).categories)
        # Convert y to codes for model compatibility if necessary
        y = pd.Categorical(y).codes

    # Train/test split (we only need the test set for evaluation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )

    print(f"Dataset split: train={len(X_train)} rows, test={len(X_test)} rows")

    print("Loading model:", model_path)
    model = load_model(model_path)

    # Some models (like XGBoost sk-learn wrappers) expect the same feature set
    # used during training. Try to extract feature names from the model and
    # select those columns from X_test. Also coerce non-numeric columns to
    # numeric or drop them (e.g., a raw 'URL' column) to avoid XGBoost errors.
    feature_names = get_model_feature_names(model)
    X_test_for_pred = X_test.copy()

    if feature_names is not None:
        # Keep only the features that exist in X_test
        present = [f for f in feature_names if f in X_test_for_pred.columns]
        if not present:
            # nothing matched, fall back to X_test as-is
            pass
        else:
            X_test_for_pred = X_test_for_pred.loc[:, present]

    # Coerce object columns to numeric where possible; drop obviously non-numeric
    # columns like 'URL' if present.
    for col in list(X_test_for_pred.columns):
        if X_test_for_pred[col].dtype == object:
            # If column is URL-like, extract features from it
            if col.lower() in ("url", "uri", "link"):
                url_feats = extract_url_features(X_test_for_pred[col])
                # merge url_feats into X_test_for_pred (drop original URL)
                X_test_for_pred = X_test_for_pred.drop(columns=[col])
                X_test_for_pred = pd.concat([X_test_for_pred.reset_index(drop=True), url_feats.reset_index(drop=True)], axis=1)
                continue
            # try to coerce to numeric
            X_test_for_pred[col] = pd.to_numeric(X_test_for_pred[col], errors="coerce")

    # Fill NaNs with 0 (safe default for many models); if you prefer a different
    # strategy, change this.
    X_test_for_pred = X_test_for_pred.fillna(0)

    try:
        y_pred = model.predict(X_test_for_pred)
    except Exception:
        # Try converting to numpy as last resort
        y_pred = model.predict(X_test_for_pred.values)

    # If predictions are probabilities, convert to class labels
    if y_pred.ndim == 2:
        # multiclass probabilities -> argmax
        y_pred = np.argmax(y_pred, axis=1)
    elif y_pred.dtype.kind == 'f' or y_pred.dtype == object:
        # binary probabilities to 0/1
        try:
            # Values like [0.1, 0.9]
            if np.all((y_pred >= 0) & (y_pred <= 1)):
                y_pred = (np.array(y_pred) > 0.5).astype(int)
        except Exception:
            pass

    # If we converted y earlier from categorical codes and have labels, translate y_test back
    if labels is not None:
        y_test_display = [labels[i] for i in y_test]
        y_pred_display = [labels[i] if (isinstance(i, (int, np.integer)) and 0 <= i < len(labels)) else i for i in y_pred]
        display_labels = labels
    else:
        y_test_display = y_test
        y_pred_display = y_pred
        display_labels = np.unique(np.concatenate([y_test_display, y_pred_display]))

    acc, report, cm = evaluate(y_test_display, y_pred_display, labels=display_labels)

    # Print results cleanly
    print("\n===== Evaluation Results =====")
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:\n")
    print(report)
    print("Confusion Matrix (numeric):")
    print(pd.DataFrame(cm, index=display_labels, columns=display_labels))

    # Plot confusion matrix
    plt_obj = plot_confusion_matrix(cm, labels=display_labels)
    if show_plot:
        plt_obj.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate phishing detection model")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to phishing_dataset.csv")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to trained model (pkl/joblib/xgb)")
    parser.add_argument("--no-plot", dest="show_plot", action="store_false", help="Do not show the confusion matrix plot")
    args = parser.parse_args()

    main(args.data, args.model, show_plot=args.show_plot)
