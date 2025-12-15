import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_data(tickers, start_date, end_date):
    """
    Downloads adjusted close prices for a list of tickers.
    """
    print(f"Downloading data for {tickers}...")
    raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)

    if isinstance(raw, pd.Series):
        data = raw.to_frame()
    else:
        if isinstance(raw.columns, pd.MultiIndex):
            top_levels = raw.columns.get_level_values(0)
            if 'Adj Close' in top_levels:
                data = raw['Adj Close']
            elif 'Close' in top_levels:
                data = raw['Close']
            else:
                raise KeyError("Downloaded data does not contain 'Adj Close' or 'Close' levels")
        else:
            if 'Adj Close' in raw.columns:
                data = raw['Adj Close']
            elif 'Close' in raw.columns:
                data = raw['Close']
            else:
                data = raw

    data = data.dropna()
    return data

if __name__ == "__main__":
    # PEP = Pepsi, KO = Coca-Cola
    tickers = ['PEP', 'KO'] 
    start_date = '2022-01-01'
    end_date = '2024-01-01'

    df = get_data(tickers, start_date, end_date)

    print("\n--- Data Head (First 5 Rows) ---")
    print(df.head())
    
    df.plot(figsize=(10, 5))
    plt.title("Price History: PEP vs KO")
    plt.ylabel("Price ($)")
    plt.show()