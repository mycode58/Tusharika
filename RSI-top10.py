import importlib.util
import yfinance as yf
import pandas as pd


    
##ticker = input("Enter the stock ticker symbol (e.g., NVDA, AAPL, MSFT, GOOGL, AMZN, AVGO, META, TSLA, BRK.B, JPM): ").upper()

ticker = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "AVGO", "META", "TSLA", "JPM"]


def get_stock_data(ticker):
    data = yf.download(ticker, period="3mo", interval="1d", progress=False)
    if data.empty or 'Close' not in data:
        return None
    close_prices = data['Close']
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    dropna_rsi = rsi.dropna()
    if not dropna_rsi.empty:
        latest_rsi = dropna_rsi.iloc[-1].item()
        return latest_rsi
    else:
        return None

# Collect RSI values for all tickers
rsi_results = []
for t in ticker:
    rsi = get_stock_data(t)
    if rsi is not None:
        rsi_results.append((t, rsi))
    else:
        rsi_results.append((t, float('-inf')))

# Sort by RSI descending (highest first)
rsi_results_sorted = sorted(rsi_results, key=lambda x: x[1], reverse=True)

print("\nStocks sorted by RSI (highest to lowest):")
for t, rsi in rsi_results_sorted:
    if rsi == float('-inf'):
        print(f"{t} - Not enough data to calculate RSI.")
    else:
        print(f"{t} - RSI: {rsi:.2f}")
