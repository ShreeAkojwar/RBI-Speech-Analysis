# RBI Speech Sentiment Analysis and Market Impact

This project analyzes the impact of RBI (Reserve Bank of India) Governor's speeches on Indian financial markets (NIFTY and SENSEX).

## Project Structure

1. `1_convert_pdf_to_txt.py`: Converts RBI speech PDFs to text format
2. `2_text_cleaning.py`: Cleans and preprocesses the speech text
3. `3_combining_txt speeches_to csv.py`: Combines processed speeches into a CSV file
4. `4_sentiment_analysis_of_rbi_text_speeches.py`: Performs sentiment analysis on speeches
5. `5_acquiring_financial_data_around_rbi_speeches.py`: Collects market data around speech dates
6. `6_data_analysis.py`: Analyzes the relationship between speech sentiments and market returns

## Data Files

- `rbi_speech_sentiments_vader.csv`: Contains sentiment scores for speeches
- `nifty_3days_before_after_speech.csv`: NIFTY index data
- `sensex_3days_before_after_speech.csv`: SENSEX index data
- `cleaned_rbi_speeches.csv`: Processed speech text data

## Results

The analysis includes:
- Sentiment analysis of RBI speeches
- Market return calculations before and after speeches
- Statistical analysis of the relationship between speech sentiments and market returns
- Visualization of results

## Requirements

```python
pandas
numpy
matplotlib
seaborn
statsmodels
scipy
```

## Usage

1. Run the scripts in numerical order (1 through 6)
2. View the generated analysis results in:
   - `nifty_analysis_results.csv`
   - `sensex_analysis_results.csv`
   - Generated scatter plots

## Findings

- RBI speeches show consistently positive sentiment (scores > 0.98)
- Market reactions vary by speech
- Individual speech impacts range from -2.26% to +1.24% for NIFTY 