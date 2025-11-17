import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import datetime
import numpy as np

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # project root
DATA = ROOT / "data" / "processed" / "snp_500.csv"
DATA2 = ROOT / "data" / "raw" / "news_djia.csv"

snp_df = pd.read_csv(DATA) #snp500 data
news_df = pd.read_csv(DATA2) #dija_news data


# convert date column to datetime if it's not already
snp_df["date"] = pd.to_datetime(snp_df["date"])
news_df["Date"] = pd.to_datetime(news_df["Date"])

# a few of the most popular stocks in the S&P 500
AAPL_df = snp_df[snp_df["name"] == "AAPL"]
AMZN_df = snp_df[snp_df["name"] == "AMZN"]
GOOG_df = snp_df[snp_df["name"] == "GOOG"]
IBM_df = snp_df[snp_df["name"] == "IBM"]
JPM_df = snp_df[snp_df["name"] == "JPM"]
MSFT_df = snp_df[snp_df["name"] == "MSFT"]
XOM_df = snp_df[snp_df["name"] == "XOM"]
V_df = snp_df[snp_df["name"] == "V"]

df_used = [AAPL_df, AMZN_df, GOOG_df, IBM_df, JPM_df, MSFT_df, XOM_df, V_df]
df_names = ["AAPL", "AMZN", "GOOG", "IBM", "JPM", "MSFT", "XOM", "V"]

#---------------------------------------------------------------------------------------
#max and min of amzn, google, aapl, ibm open and close range data
tickers = {
    "AMZN": AMZN_df,
    "GOOG": GOOG_df,
    "AAPL": AAPL_df,
    "IBM": IBM_df
}

results = []
for name, df in tickers.items():
    max_oc = df["oc_range"].max()
    min_oc = df["oc_range"].min()
    max_date = df.loc[df["oc_range"].idxmax(), "date"]
    min_date = df.loc[df["oc_range"].idxmin(), "date"]
    
    results.append({
        "ticker": name,
        "max_oc": max_oc,
        "max_date": max_date,
        "min_oc": min_oc,
        "min_date": min_date
    })

# summary dataframe
summary_df = pd.DataFrame(results)
print(summary_df)

# testing
def get_news_on_date(date, df=news_df):
    """Return the 25 headlines for a given date (datetime or string)."""
    date = pd.to_datetime(date)
    row = df[df["Date"] == date]
    if row.empty:
        return None
    top_cols = [f"Top{i}" for i in range(1, 26)]
    headlines = row[top_cols].iloc[0].dropna().tolist()
    return headlines

summary_df["max_headlines"] = summary_df["max_date"].apply(get_news_on_date)
summary_df["min_headlines"] = summary_df["min_date"].apply(get_news_on_date)


for name in tickers:
    print()
    row = summary_df.loc[summary_df["ticker"] == name].iloc[0]
    print(f"{name} MAX ({row['max_date'].date()}):\n")
    for h in row["max_headlines"]:
        print("-", h)
    print(f"\n{name} MIN ({row['min_date'].date()}):\n")
    for h in row["min_headlines"]:
        print("-", h)
#-------------------------------------------------------------------------------------------
print('\n')
print('---------------------------------------------------------------------')
print('\n')
#------------------------------------------------------------------------------------------
#max and min of amzn, google, aapl, ibm high and low range data
resultshl = []
for name, df in tickers.items():
    max_hl = df["hl_range"].max()
    min_hl = df["hl_range"].min()
    max_hl_date = df.loc[df["hl_range"].idxmax(), "date"]
    min_hl_date = df.loc[df["hl_range"].idxmin(), "date"]
    
    resultshl.append({
        "ticker": name,
        "max_hl": max_hl,
        "max_date": max_hl_date,
        "min_hl": min_hl,
        "min_date": min_hl_date
    })

# summary dataframe
summaryhl_df = pd.DataFrame(resultshl)
print(summaryhl_df)

summaryhl_df["max_headlines"] = summaryhl_df["max_date"].apply(get_news_on_date)
summaryhl_df["min_headlines"] = summaryhl_df["min_date"].apply(get_news_on_date)

for name in tickers:
    row = summaryhl_df.loc[summaryhl_df["ticker"] == name].iloc[0]
    print(f"\n{name} MAX ({row['max_date'].date()}):\n")
    for h in row["max_headlines"]:
        print("-", h)
    print(f"\n{name} MIN ({row['min_date'].date()}):\n")
    for h in row["min_headlines"]:
        print("-", h)

#-------------------------------------------------------------------------------------------------------

#colors for lineplots
colors = ['#1618E2', '#48B347', '#C3C3F9', '#DDDC7E', '#27B06A', '#8182F3', '#030320', '#7CA6D0']

matplotlib.use("MacOSX") # allows the plot to show on MacOSX computers, comment out if not using/error occurs

# Open-Close Range Plot
plt.figure(figsize=(12, 5))
for x in range(len(df_used)):
    plt.plot(df_used[x]["date"], df_used[x]["oc_range"], color=colors[x], label=df_names[x])
plt.title("S&P 500 Open-Close Range Over Time")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8) # legend
plt.xlabel("Date")
plt.ylabel("Open-Close Range")

### format x-axis to show years and months
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3)) 
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  
plt.xticks(rotation=45) 

plt.tight_layout()
plt.show()

# High-Low Range Plot
plt.figure(figsize=(12, 5))
for x in range(len(df_used)):
    plt.plot(df_used[x]["date"], df_used[x]["hl_range"], color=colors[x], label=df_names[x])
plt.title("S&P 500 High-Low Range Over Time")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8) # legend
plt.xlabel("Date")
plt.ylabel("High-Low Range")

### format x-axis to show years and months
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3)) 
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  
plt.xticks(rotation=45) 

plt.tight_layout()
plt.show()

# for the summary statistics

range_type = ["high low", "open close"]
range_abv = ["hl", "oc"]


print("\n\n")
for x in range(len(df_used)):
    print(f"ticker {df_names[x]}")
    for y in range(len(range_type)):
        print(f"\n{range_type[y]} range statistics\n")
        print(f"mean: {round(df_used[x][f"{range_abv[y]}_range"].mean(), 2)}")
        print(f"median: {round(df_used[x][f"{range_abv[y]}_range"].median(), 2)}")
        print(f"variance: {round(df_used[x][f"{range_abv[y]}_range"].var(), 2)}")
        print(f"standard deviation: {round(df_used[x][f"{range_abv[y]}_range"].std(), 2)}")
    print("\n\n")
