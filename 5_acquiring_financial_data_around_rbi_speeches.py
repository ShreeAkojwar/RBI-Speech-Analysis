import pandas as pd
import yfinance as yf

# 1. Read the RBI speech dates from your CSV
speech_df = pd.read_csv('/Users/shreeakojwar/Downloads/IIMK_Project/rbi_speeches_list.csv')
speech_df['Date'] = pd.to_datetime(speech_df['Date'])

# 2. Set up index tickers and output file names
indices = {
    'NIFTY50': {'ticker': '^NSEI', 'output': 'nifty_3days_before_after_speech.csv'},
    'SENSEX':  {'ticker': '^BSESN', 'output': 'sensex_3days_before_after_speech.csv'}
}

# 3. Loop over both indices
for idx_name, idx_info in indices.items():
    print(f"Processing {idx_name}...")
    ticker = idx_info['ticker']
    output_file = idx_info['output']
    
    # Download index data (buffer extra days to cover window edges)
    start_date = speech_df['Date'].min() - pd.Timedelta(days=10)
    end_date = speech_df['Date'].max() + pd.Timedelta(days=10)
    stock_df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    stock_df.reset_index(inplace=True)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    stock_df.set_index('Date', inplace=True)
    
    # Extract 3 days before and after each speech date
    def get_window_data(stock_df, event_dates, window=3):
        results = []
        for event_date in event_dates:
            start = event_date - pd.Timedelta(days=window)
            end = event_date + pd.Timedelta(days=window)
            window_data = stock_df.loc[start:end].copy()
            window_data['Speech_Date'] = event_date
            results.append(window_data)
        return pd.concat(results).reset_index()
    
    windowed_data = get_window_data(stock_df, speech_df['Date'], window=3)
    windowed_data.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")

print("Done! You now have both NIFTY and SENSEX windows around each RBI speech.")
