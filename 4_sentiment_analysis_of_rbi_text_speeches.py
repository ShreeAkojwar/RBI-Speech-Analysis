import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# 1. Load your cleaned speeches
# Your CSV should have at least two columns: 'Speech_Date' and 'Speech_Text'
df = pd.read_csv('/Users/shreeakojwar/Downloads/IIMK_Project/cleaned_rbi_speeches.csv')

# 2. Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# 3. Define a function to get the compound sentiment score
def get_vader_sentiment(text):
    scores = analyzer.polarity_scores(text)
    return scores['compound']  # Compound score is the overall sentiment (-1 to 1)

# 4. Apply the function to each speech
df['Sentiment_Score'] = df['Speech_Text'].apply(get_vader_sentiment)

# 5. (Optional) Classify sentiment as positive, negative, or neutral
def classify_sentiment(compound):
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['Sentiment_Label'] = df['Sentiment_Score'].apply(classify_sentiment)

# 6. Save the results
df[['Speech_Date', 'Sentiment_Score', 'Sentiment_Label']].to_csv('rbi_speech_sentiments_vader.csv', index=False)

print("Sentiment analysis complete! Results saved to 'rbi_speech_sentiments_vader.csv'.")
