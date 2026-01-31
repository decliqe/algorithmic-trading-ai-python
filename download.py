import json
import datetime
import yfinance as yf
import os

def main():
    download_ticker()
    download_news()
    prepare_data()

## Download BTC-USD historical data from Yahoo Finance
## Minute resolution data for the last 60 days
def download_ticker():
    if not os.path.exists('BTC-USD_historical_data.json'):
        print(f"ticker data file not found, creating new one.")
        with open('BTC-USD_historical_data.json', 'w') as f:
            data = yf.download(tickers='BTC-USD', period='1mo', interval='5m')
            encoded = data.to_json()
            decoded = json.loads(encoded)
            close = decoded["('Open', 'BTC-USD')"]
            json.dump(close, f, indent=4)
    else:
        print(f"ticker data file found, updating existing one.")
        with open('BTC-USD_historical_data.json', 'r') as f:
            existing_data = json.load(f)

        data = yf.download(tickers='BTC-USD', period='1mo', interval='5m')
        encoded = data.to_json()
        decoded = json.loads(encoded)
        close   = decoded["('Open', 'BTC-USD')"]

        ## Merge with existing data with close
        for kj in existing_data:
            if kj not in close:
                close[kj] = existing_data[kj]

        ## Save to file
        with open('BTC-USD_historical_data.json', 'w') as f:
            json.dump(close, f, indent=4)

## Download BTC-USD news from Yahoo Finance
def download_news():
    ## Load existing news to avoid duplicates
    news = []
    if not os.path.exists('BTC-USD_news.json'):
        print(f"news data file not found, creating new one.")
        news = yf.Ticker('BTC-USD').get_news(count=1000)
        with open('BTC-USD_news.json', 'w') as f:
            json.dump(news, f, indent=4)
    else:
        print(f"news data file found, updating existing one.")
        with open('BTC-USD_news.json', 'r') as f:
            news = json.load(f)

        ## Download News for BTC-USD from Yahoo Finance
        new_news = yf.Ticker('BTC-USD').get_news(count=1000)
        if new_news is None: new_news = []
        news += new_news

        with open('BTC-USD_news.json', 'w') as f:
            json.dump(news, f, indent=4)

## Prepare data for training
def prepare_data():
    output = []

    ## Load data from files
    with open('BTC-USD_historical_data.json', 'r') as f:
        ticker = json.load(f)
    with open('BTC-USD_news.json', 'r') as f:
        news = json.load(f)

    ## Augment with Pricing data
    for item in news:
        title   = item['content']['title']
        summary = item['content']['summary']
        pubDate = item['content']['pubDate']

        ## Convert pubDate to unix timestamp
        pub_datetime = datetime.datetime.strptime(pubDate, '%Y-%m-%dT%H:%M:%SZ')
        pubDate_ts = int(pub_datetime.timestamp())

        ## Extract time features
        day_of_week = pub_datetime.strftime('%A')  # e.g., "Monday", "Tuesday"
        hour_of_day = pub_datetime.hour          # 0-23


        # Round down to nearest 5 minutes
        index = pubDate_ts - (pubDate_ts % 300)  
        price = ticker.get(f"{index}000")
        future_price = ticker.get(f"{index + 300}000")

        if price is None or future_price is None:
            print(f"Skipping entry with missing price data: title={title}, pubDate={pubDate}, index={index}")
            continue    

        difference = price - future_price
        output.append({
            'title': title,
            'index': index,
            'price' : price,
            'future_price': future_price,
            'difference': difference,
            'percentage': (difference / price) * 100,
            'summary': summary,
            'pubDate': pubDate,
            'pubDate_ts': pubDate_ts,
            'day_of_week': day_of_week,
            'hour_of_day': hour_of_day,
        })

    with open('BTC-USD_news_with_price.json', 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__": main()
