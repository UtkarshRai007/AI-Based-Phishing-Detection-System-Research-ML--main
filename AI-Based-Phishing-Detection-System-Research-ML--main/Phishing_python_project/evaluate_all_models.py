"""Evaluate all models found in the project directory.

This script searches the current project directory for common model files
(.pkl, .joblib, .model, .bin, .json) and evaluates each against
`phishing_test_20.csv`, computing accuracy, precision, recall, f1 and
confusion matrix. Outputs are printed and saved to `model_metrics.csv`.
"""
from pathlib import Path
import pickle
import joblib
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import xgboost as xgb
except Exception:
    xgb = None


BASE = Path(__file__).resolve().parent
TEST_CSV = BASE / 'phishing_test_20.csv'


OUT_DIR = BASE / 'model_reports'
OUT_DIR.mkdir(exist_ok=True)


def extract_features_from_url(s: pd.Series) -> pd.DataFrame:
    s = s.fillna("").astype(str)
    url_length = s.str.len()
    lower = s.str.lower()
    has_https = lower.str.contains('https') | lower.str.contains('https://')
    num_dots = s.str.count(r"\\.")
    has_at = s.str.contains('@')
    has_dash = s.str.contains('-')

    ipv4_re = pd.Series(s).str.contains(r"(?:(?:25[0-5]|2[0-4]\\d|[01]?\\d?\\d)\\.){3}(?:25[0-5]|2[0-4]\\d|[01]?\\d?\\d)")
    has_ip = ipv4_re.astype(int)

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
    try:
        return joblib.load(path_str)
    except Exception:
        pass
    try:
        with open(path_str, 'rb') as f:
            return pickle.load(f)
    except Exception:
        pass
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


def evaluate_model_on_test(model_path: Path, df_test: pd.DataFrame):
    model = load_model(model_path)
    # find label column
    label_col = None
    for c in ('Label','label','target','is_phishing','phishing'):
        if c in df_test.columns:
            label_col = c
            break
    if label_col is None:
        label_col = df_test.columns[-1]
    y_raw = df_test[label_col]
    if y_raw.dtype == object:
        y = y_raw.str.lower().map({'good':0,'legitimate':0,'0':0,'bad':1,'1':1}).fillna(0).astype(int)
    else:
        y = y_raw.astype(int)

    if 'URL' in df_test.columns:
        url_col = 'URL'
    elif 'url' in df_test.columns:
        url_col = 'url'
    else:
        url_col = df_test.columns[0]

    X_features = extract_features_from_url(df_test[url_col])

    # Align features if model expects named features
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

    try:
        y_pred = model.predict(X_for_pred)
    except Exception:
        y_pred = model.predict(X_for_pred.values)

    y_pred = np.array(y_pred)
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)
    elif y_pred.dtype.kind == 'f':
        y_pred = (y_pred > 0.5).astype(int)

    acc = accuracy_score(y, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, labels=[0,1], zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=[0,1])

    # save classification report
    cls_report = classification_report(y, y_pred, target_names=['good(0)','bad(1)'], zero_division=0)
    report_path = OUT_DIR / f"{model_path.stem}_classification_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_path.name}\n\n")
        f.write(cls_report)

    # save confusion matrix image
    try:
        sns.set(font_scale=1.1)
        plt.figure(figsize=(5,4))
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_xticklabels(['0','1'])
        ax.set_yticklabels(['0','1'])
        plt.title(f'Confusion Matrix: {model_path.name}')
        plt.tight_layout()
        img_path = OUT_DIR / f"{model_path.stem}_confusion_matrix.png"
        plt.savefig(img_path, dpi=160)
        plt.close()
    except Exception as e:
        img_path = None

    out = {
        'model': str(model_path.name),
        'accuracy': acc,
        'precision_0': precision[0],
        'recall_0': recall[0],
        'f1_0': f1[0],
        'support_0': int(support[0]),
        'precision_1': precision[1],
        'recall_1': recall[1],
        'f1_1': f1[1],
        'support_1': int(support[1]),
        'cm_00': int(cm[0,0]),
        'cm_01': int(cm[0,1]),
        'cm_10': int(cm[1,0]),
        'cm_11': int(cm[1,1]),
        'report_path': str(report_path),
        'confusion_image': str(img_path) if img_path is not None else '',
    }
    return out


def main():
    parser = argparse.ArgumentParser(description='Evaluate all models and save reports')
    parser.add_argument('--dir', type=Path, default=BASE, help='Directory to search for models')
    parser.add_argument('--out', type=Path, default=BASE / 'model_metrics.csv', help='CSV output file')
    parser.add_argument('--test', type=Path, help='Path to a test CSV file (if omitted, use default test CSV)')
    parser.add_argument('--data', type=Path, help='Path to a full dataset CSV to split into train/test')
    parser.add_argument('--test-size', type=float, default=0.2, help='If --data provided, fraction to use as test')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for split')
    args = parser.parse_args()

    # decide test dataframe
    if args.test:
        if not args.test.exists():
            raise SystemExit(f'Test file not found: {args.test}')
        df_test = pd.read_csv(args.test)
    elif args.data:
        if not args.data.exists():
            raise SystemExit(f'Data file not found: {args.data}')
        df_all = pd.read_csv(args.data)
        from sklearn.model_selection import train_test_split
        _, df_test = train_test_split(df_all, test_size=args.test_size, random_state=args.random_state, stratify=None)
    else:
        df_test = pd.read_csv(TEST_CSV)

    model_files = []
    for ext in ('*.pkl','*.joblib','*.model','*.bin','*.json'):
        model_files.extend(list(args.dir.glob(ext)))

    if not model_files:
        print('No model files found in', args.dir)
        return

    results = []
    for m in sorted(model_files):
        print('Evaluating', m.name)
        try:
            res = evaluate_model_on_test(m, df_test)
            results.append(res)
            print(f"  accuracy: {res['accuracy']:.4f}")
        except Exception as e:
            print('  Failed to evaluate', m.name, '->', e)

    if results:
        df_out = pd.DataFrame(results)
        df_out.to_csv(args.out, index=False)
        # print summary table
        display_df = df_out[['model','accuracy','precision_0','recall_0','f1_0','precision_1','recall_1','f1_1']]
        print('\nSummary:')
        print(display_df.to_string(index=False, float_format='%.4f'))
        print('\nPer-model reports and confusion matrix images saved to:', OUT_DIR)
        print('\nSaved metrics to', args.out)


if __name__ == '__main__':
    main()
