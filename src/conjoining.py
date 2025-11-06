import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# reading data

raw_snp_df = pd.read_csv("../data/raw/all_stocks_snp.csv", header = 0)
raw_news_df = pd.read_csv("../data/raw/news_djia.csv", header = 0)

# separate dates 2013-05-01 to 2016-05-02

raw_snp_df["date"] = pd.to_datetime(raw_snp_df["date"])
raw_news_df["Date"] = pd.to_datetime(raw_news_df["Date"])

dates_snp_df = raw_snp_df[
    (raw_snp_df["date"] >= "2013-05-01") &
    (raw_snp_df["date"] <= "2016-05-02")
]
dates_news_df = raw_news_df[
    (raw_news_df["Date"] >= "2013-05-01") &
    (raw_news_df["Date"] <= "2016-05-02")
]

# Make snp df columns: date, ticker, oc_range, hl_range

oc_range = round(dates_snp_df["open"] - dates_snp_df["close"], 2)
hl_range = round(dates_snp_df["high"] - dates_snp_df["low"], 2)

snp_df = pd.DataFrame({
    'date': dates_snp_df['date'],
    'name': dates_snp_df['Name'],
    'oc_range': oc_range,
    'hl_range': hl_range,
})

# Make news df columns: date, djia, article

article_news_df = dates_news_df.melt(
    id_vars=['Date', 'Label'],
    value_vars=[f'Top{y}' for y in range(1, 26)],
    value_name='article'
).rename(columns={'Date': 'date', 'Label': 'djia'})[['date', 'djia', 'article']]

# print(article_news_df)

# Make vectorized version of news df

vectorizer = TfidfVectorizer()

vectorized_articles = vectorizer.fit_transform(article_news_df['article'])

vector_news_df = pd.DataFrame({
    'date': article_news_df['date'].values,
    'djia': article_news_df['djia'].values,
    'article_vector': list(vectorized_articles)
})


# make csv files

vector_news_df.to_csv('data/processed/vector_news.csv', index=False)
snp_df.to_csv('data/processed/snp_500.csv', index=False)