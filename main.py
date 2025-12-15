import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def calculate_spread(df, stock_A, stock_B):
    """
    Calculates the spread using Linear Regression (OLS).
    Spread = Stock_A - (Hedge_Ratio * Stock_B)
    """
    X = df[stock_B]
    Y = df[stock_A]
    X = sm.add_constant(X)
    
    model = sm.OLS(Y, X).fit()
    hedge_ratio = model.params[stock_B]
    
    print(f"\nCalculated Hedge Ratio: {hedge_ratio:.4f}")
    
    spread = df[stock_A] - (hedge_ratio * df[stock_B])
    return spread, hedge_ratio

def check_cointegration(spread):
    """
    Runs the Augmented Dickey-Fuller test on the spread.
    If p-value < 0.05, the spread is stationary (cointegrated).
    """
    adf_result = adfuller(spread)
    p_value = adf_result[1]
    
    print(f"\n--- ADF Test Results ---")
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"P-Value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("RESULT: The pair is COINTEGRATED. (Good for trading)")
        return True
    else:
        print("RESULT: The pair is NOT cointegrated. (Do not trade)")
        return False

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
    tickers = ['PEP', 'KO']
    df = get_data(tickers, '2022-01-01', '2024-01-01')

    spread, ratio = calculate_spread(df, 'PEP', 'KO')

    is_tradable = check_cointegration(spread)

    plt.figure(figsize=(10, 5))
    spread.plot()
    plt.title(f"Spread: PEP - ({ratio:.2f} * KO)")
    plt.axhline(spread.mean(), color='black', linestyle='--')
    plt.ylabel("Spread Value")
    plt.show()