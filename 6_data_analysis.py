import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better display
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr
import matplotlib.dates as mdates
import os
from wordcloud import WordCloud

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

# === DIAGNOSTICS: Why are only some speeches analyzed? ===
print("\n=== DIAGNOSTICS: Speech Date Coverage ===")
# 1. Unique dates in each file
sentiment_dates = set(pd.to_datetime(sentiments['Speech_Date']))
nifty_dates = set(pd.to_datetime(nifty['Speech_Date']))
sensex_dates = set(pd.to_datetime(sensex['Speech_Date']))
print(f"Unique speech dates in sentiment file: {len(sentiment_dates)}")
print(f"Unique speech dates in NIFTY market data: {len(nifty_dates)}")
print(f"Unique speech dates in SENSEX market data: {len(sensex_dates)}")

# 2. Dates in sentiment file but missing from market data
missing_nifty = sorted([d.strftime('%Y-%m-%d') for d in sentiment_dates - nifty_dates])
missing_sensex = sorted([d.strftime('%Y-%m-%d') for d in sentiment_dates - sensex_dates])
print(f"Speech dates in sentiment file but missing from NIFTY data: {missing_nifty}")
print(f"Speech dates in sentiment file but missing from SENSEX data: {missing_sensex}")

# 3. After merge, how many speeches remain?
print(f"Rows after merging with NIFTY: {len(nifty_combined)}")
print(f"Rows after merging with SENSEX: {len(sensex_combined)}")

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

# After cleaning
print(f"Rows after cleaning NIFTY: {len(nifty_clean)}")
print(f"Rows after cleaning SENSEX: {len(sensex_clean)}")

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

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)

# === Step 10: Highlight Big Market Moves and More Visualizations ===

for label, df in zip(["NIFTY", "SENSEX"], [nifty_clean, sensex_clean]):
    df_sorted = df.copy()
    df_sorted['abs_return_diff'] = df_sorted['return_diff'].abs()
    df_sorted = df_sorted.sort_values('Speech_Date')
    
    # 1. Bar chart of signed return_diff
    plt.figure(figsize=(14, 6))
    bars = plt.bar(df_sorted['Speech_Date'].dt.strftime('%Y-%m-%d'), df_sorted['return_diff'], color='lightcoral')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Return Difference')
    plt.xlabel('Speech Date')
    plt.title(f'{label}: Market Move After Speech (Signed)')
    plt.axhline(0, color='black', linewidth=0.8)
    top_pos = df_sorted.nlargest(3, 'return_diff')
    top_neg = df_sorted.nsmallest(3, 'return_diff')
    for idx, row in pd.concat([top_pos, top_neg]).iterrows():
        plt.annotate(f"{row['return_diff']:.2%}",
                     (df_sorted.loc[idx, 'Speech_Date'].strftime('%Y-%m-%d'), row['return_diff']),
                     textcoords="offset points", xytext=(0,10), ha='center', color='blue', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"plots/{label.lower()}_signed_return_diff.png", dpi=200)
    plt.show()

    # 2. Bar chart of absolute return_diff
    plt.figure(figsize=(14, 6))
    bars = plt.bar(df_sorted['Speech_Date'].dt.strftime('%Y-%m-%d'), df_sorted['abs_return_diff'], color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Absolute Return Difference')
    plt.xlabel('Speech Date')
    plt.title(f'{label}: Magnitude of Market Move After Speech')
    top_n = 5
    top_moves = df_sorted.nlargest(top_n, 'abs_return_diff')
    for idx, row in top_moves.iterrows():
        plt.annotate(f"{row['abs_return_diff']:.2%}",
                     (df_sorted.loc[idx, 'Speech_Date'].strftime('%Y-%m-%d'), row['abs_return_diff']),
                     textcoords="offset points", xytext=(0,10), ha='center', color='red', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"plots/{label.lower()}_abs_return_diff.png", dpi=200)
    plt.show()

    # 3. Boxplot of return_diff
    plt.figure(figsize=(6, 6))
    plt.boxplot(df_sorted['return_diff'], vert=True, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    plt.ylabel('Return Difference')
    plt.title(f'{label}: Distribution of Market Moves After Speech')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/{label.lower()}_boxplot_return_diff.png", dpi=200)
    plt.show()

    # 4. Enhanced scatterplot: color by magnitude
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df_sorted['Sentiment_Score'], df_sorted['return_diff'],
                         c=df_sorted['abs_return_diff'], cmap='coolwarm', s=80, edgecolor='k')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Return Difference')
    plt.title(f'{label}: Sentiment vs Return Difference (Color = Magnitude)')
    plt.colorbar(scatter, label='|Return Difference|')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/{label.lower()}_scatter_sentiment_vs_return.png", dpi=200)
    plt.show()

# === Step 11: More Statistical Variations ===

for label, df, market_df in zip(["NIFTY", "SENSEX"], [nifty_clean, sensex_clean], [nifty, sensex]):
    # 1. Volatility before/after each speech
    vol_before = []
    vol_after = []
    speech_dates = []
    for speech_date in df['Speech_Date']:
        window = market_df[market_df['Speech_Date'] == speech_date]
        before = window[window['period'] == 'before']['return']
        after = window[window['period'] == 'after']['return']
        if len(before) > 0 and len(after) > 0:
            vol_before.append(before.std())
            vol_after.append(after.std())
            speech_dates.append(speech_date)
    plt.figure(figsize=(12, 5))
    plt.plot(speech_dates, vol_before, marker='o', label='Volatility Before')
    plt.plot(speech_dates, vol_after, marker='o', label='Volatility After')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Volatility (Std Dev of Returns)')
    plt.xlabel('Speech Date')
    plt.title(f'{label}: Volatility Before vs After Speech')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{label.lower()}_volatility_before_after.png", dpi=200)
    plt.show()

    # 2. Histogram of return_diff
    plt.figure(figsize=(8, 5))
    plt.hist(df['return_diff'], bins=10, color='orchid', edgecolor='black', alpha=0.7)
    plt.xlabel('Return Difference')
    plt.ylabel('Frequency')
    plt.title(f'{label}: Histogram of Market Moves After Speech')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/{label.lower()}_histogram_return_diff.png", dpi=200)
    plt.show()

    # 3. Cumulative sum of return_diff
    df_sorted = df.sort_values('Speech_Date')
    plt.figure(figsize=(12, 5))
    plt.plot(df_sorted['Speech_Date'], df_sorted['return_diff'].cumsum(), marker='o', color='teal')
    plt.xlabel('Speech Date')
    plt.ylabel('Cumulative Return Difference')
    plt.title(f'{label}: Cumulative Market Impact of RBI Speeches')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/{label.lower()}_cumulative_return_diff.png", dpi=200)
    plt.show()

# Create sentiment trend plot using actual data structure
print("\nCreating sentiment trend plot...")

# Load sentiment data
sentiment_df = pd.read_csv('rbi_speech_sentiments_vader.csv')

# Extract year from Speech_Date (first 4 characters)
sentiment_df['year'] = sentiment_df['Speech_Date'].str[:4].astype(int)

# Group by year and calculate average sentiment
yearly_sentiment = sentiment_df.groupby('year')['Sentiment_Score'].mean().reset_index()

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(yearly_sentiment['year'], yearly_sentiment['Sentiment_Score'], marker='o', linewidth=2, markersize=8, label='Average VADER Sentiment Score')
plt.title('Sentiment Trend in RBI Speeches (2019-2025)', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Average Sentiment Score', fontsize=12)
plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(yearly_sentiment['year'])
plt.ylim(0.97, 1.01)  # Adjust y-axis to show the variation better

# Add value labels on points
for idx, row in yearly_sentiment.iterrows():
    plt.annotate(f'{row["Sentiment_Score"]:.4f}', 
                (row['year'], row['Sentiment_Score']), 
                textcoords="offset points", 
                xytext=(0,10), 
                ha='center', 
                fontsize=9)

plt.tight_layout()
plt.savefig('plots/sentiment_trend_over_years.png', dpi=300, bbox_inches='tight')
plt.show()

print("Sentiment trend plot saved as 'plots/sentiment_trend_over_years.png'")

# Create wordcloud from topic words
print("\nCreating wordcloud...")

# Sample topic words (you can replace with actual analysis results)
topic_words = {
    "monetary": 0.15,
    "policy": 0.13,
    "liquidity": 0.12,
    "inflation": 0.10,
    "growth": 0.09,
    "banking": 0.08,
    "credit": 0.07,
    "rates": 0.06,
    "market": 0.05,
    "economy": 0.05,
    "financial": 0.04,
    "stability": 0.04,
    "regulation": 0.03,
    "development": 0.03
}

# Generate wordcloud
wordcloud = WordCloud(width=800, height=400, background_color='white', 
                     colormap='viridis', max_words=20).generate_from_frequencies(topic_words)

# Display wordcloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("RBI Speech Topics: Monetary Policy & Financial Stability", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("plots/rbi_speech_wordcloud.png", dpi=300, bbox_inches='tight')
plt.show()

print("Wordcloud saved as 'plots/rbi_speech_wordcloud.png'")
print("\n=== Analysis Complete ===")
print("All plots have been saved to the 'plots/' directory")
print("Analysis results saved as 'nifty_analysis_results.csv' and 'sensex_analysis_results.csv'")