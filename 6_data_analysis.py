import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr

# === Step 1: Load and Clean Sentiment & Market Data ===

# Read sentiment data
sentiments = pd.read_csv("/Users/shreeakojwar/Downloads/IIMK_Project/rbi_speech_sentiments_vader.csv")

# Read market index data
nifty = pd.read_csv("/Users/shreeakojwar/Downloads/IIMK_Project/nifty_3days_before_after_speech.csv")
sensex = pd.read_csv("/Users/shreeakojwar/Downloads/IIMK_Project/sensex_3days_before_after_speech.csv")

# Clean sentiment data - extract just the date part
sentiments['Speech_Date'] = pd.to_datetime(sentiments['Speech_Date'].str.split().str[0])

# Clean market data
for df in [nifty, sensex]:
    # Convert dates to datetime
    df['date'] = pd.to_datetime(df['Date'])
    df['Speech_Date'] = pd.to_datetime(df['Speech_Date'])
    # Convert index values to float
    df['index_value'] = pd.to_numeric(df['Close'], errors='coerce')
    # Drop rows where Speech_Date is empty
    df.dropna(subset=['Speech_Date'], inplace=True)

# === Step 2: Compute Daily Returns Around Speeches ===

def calculate_returns(df):
    df = df.sort_values(by=['Speech_Date', 'date'])
    df['return'] = df.groupby('Speech_Date')['index_value'].pct_change()
    return df

nifty = calculate_returns(nifty)
sensex = calculate_returns(sensex)

# === Step 3: Label Each Day as Before/After ===

def label_days(df):
    df['day_diff'] = (df['date'] - df['Speech_Date']).dt.days
    df['period'] = df['day_diff'].apply(lambda x: 'before' if x < 0 else ('speech_day' if x == 0 else 'after'))
    return df

nifty = label_days(nifty)
sensex = label_days(sensex)

# === Step 4: Average Market Reaction Summary ===

def summarize_market_reaction(df):
    summary = (
        df[df['period'].isin(['before', 'after'])]
        .groupby(['Speech_Date', 'period'])['return']
        .mean()
        .unstack(fill_value=0)
        .reset_index()
    )
    summary['return_diff'] = summary['after'] - summary['before']
    return summary

nifty_summary = summarize_market_reaction(nifty)
sensex_summary = summarize_market_reaction(sensex)

# === Step 5: Merge with Sentiment Data ===

# Print unique dates before merge for debugging
print("\n=== Date Comparison Before Merge ===")
print("Sentiment Dates:", sorted(sentiments['Speech_Date'].unique())[:5], "...")
print("Nifty Dates:", sorted(nifty_summary['Speech_Date'].unique())[:5], "...")
print("Sensex Dates:", sorted(sensex_summary['Speech_Date'].unique())[:5], "...")

nifty_combined = pd.merge(sentiments, nifty_summary, on='Speech_Date', how='inner')
sensex_combined = pd.merge(sentiments, sensex_summary, on='Speech_Date', how='inner')

# Print merge diagnostics
print("\n=== Merge Diagnostics ===")
print(f"Original sentiment data rows: {len(sentiments)}")
print(f"Nifty matches: {len(nifty_combined)}")
print(f"Sensex matches: {len(sensex_combined)}")

# === Step 6: Clean for Analysis ===

def clean_for_analysis(df):
    clean_df = df[
        np.isfinite(df['Sentiment_Score']) & 
        np.isfinite(df['return_diff'])
    ].copy()
    clean_df = clean_df.dropna(subset=['Sentiment_Score', 'return_diff'])
    return clean_df

nifty_clean = clean_for_analysis(nifty_combined)
sensex_clean = clean_for_analysis(sensex_combined)

# Print cleaning diagnostics
print("\n=== Cleaning Diagnostics ===")
print(f"Nifty rows after cleaning: {len(nifty_clean)}")
print(f"Sensex rows after cleaning: {len(sensex_clean)}")

# === Step 7: Diagnostics ===

print("\n=== NIFTY Cleaned Data ===")
print(nifty_clean[['Sentiment_Score', 'return_diff']].describe())
print("\nUnique sentiment scores:", nifty_clean['Sentiment_Score'].nunique())
print("Unique return_diff values:", nifty_clean['return_diff'].nunique())

print("\n=== SENSEX Cleaned Data ===")
print(sensex_clean[['Sentiment_Score', 'return_diff']].describe())
print("\nUnique sentiment scores:", sensex_clean['Sentiment_Score'].nunique())
print("Unique return_diff values:", sensex_clean['return_diff'].nunique())

# === Step 8: Run Correlation & Regression ===

def run_analysis(df, label=""):
    print(f"\n=== Analysis for {label} ===")
    if len(df) < 3:
        print("Not enough data to analyze. Need at least 3 points.")
        return

    if df['Sentiment_Score'].nunique() == 1 or df['return_diff'].nunique() == 1:
        print("No variation in data. Cannot run correlation.")
        return

    # Pearson correlation
    corr, p_value = pearsonr(df['Sentiment_Score'], df['return_diff'])
    print(f"Pearson Correlation: {corr:.3f} | P-value: {p_value:.4f}")

    # Linear regression
    X = sm.add_constant(df['Sentiment_Score'])
    y = df['return_diff']
    model = sm.OLS(y, X).fit()
    print("\nRegression Summary:")
    print(model.summary().tables[1])

run_analysis(nifty_clean, "NIFTY")
run_analysis(sensex_clean, "SENSEX")

# === Step 9: Visualizations ===

# Set the style parameters directly
plt.rcParams['figure.figsize'] = (15, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.axisbelow'] = True

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2)

# NIFTY Plot
sns.regplot(data=nifty_clean, x='Sentiment_Score', y='return_diff',
            scatter_kws={'alpha':0.5}, line_kws={'color': 'red'},
            ax=ax1)
ax1.set_xlabel('Sentiment Score')
ax1.set_ylabel('Return Difference (%)')
ax1.set_title(f'NIFTY: Sentiment vs Returns\n(n={len(nifty_clean)})')
ax1.grid(True, alpha=0.3)

# SENSEX Plot
sns.regplot(data=sensex_clean, x='Sentiment_Score', y='return_diff',
            scatter_kws={'alpha':0.5}, line_kws={'color': 'red'},
            ax=ax2)
ax2.set_xlabel('Sentiment Score')
ax2.set_ylabel('Return Difference (%)')
ax2.set_title(f'SENSEX: Sentiment vs Returns\n(n={len(sensex_clean)})')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Save results to CSV for further analysis if needed
nifty_clean.to_csv('nifty_analysis_results.csv', index=False)
sensex_clean.to_csv('sensex_analysis_results.csv', index=False)