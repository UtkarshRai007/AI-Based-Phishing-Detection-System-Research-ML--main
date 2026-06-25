import pandas as pd
from pathlib import Path
p = Path(__file__).resolve().parent / 'phishing_dataset.csv'
print('CSV path:', p)
# Read in chunks to compute counts without loading entire file into memory if needed
# But pandas can handle ~550k rows; we'll load but guard memory
try:
    df = pd.read_csv(p)
except Exception as e:
    print('Error reading CSV:', e)
    raise

print('\nRows:', len(df))
print('Columns:', len(df.columns))
print('Column names:', df.columns.tolist())
print('\nDtypes:')
print(df.dtypes)

# Label info
label_cols = [c for c in ['Label','label','target','is_phishing','phishing'] if c in df.columns]
label_col = label_cols[0] if label_cols else df.columns[-1]
print('\nUsing label column:', label_col)
print(df[label_col].value_counts(dropna=False).to_string())
print('\nLabel value counts (normalized):')
print(df[label_col].value_counts(normalize=True).to_string())

# Sample rows
print('\nSample rows (first 5):')
print(df.head().to_string(index=False))

# Basic URL stats if URL column exists
if 'URL' in df.columns:
    s = df['URL'].astype(str)
    print('\nURL length stats:')
    print('min', s.str.len().min(), 'max', s.str.len().max(), 'mean', s.str.len().mean())
    print('\nTop 10 domains (by simple split):')
    domains = s.str.split('/').str[0]
    print(domains.value_counts().head(10).to_string())

print('\nDone')
