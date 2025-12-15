import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def get_data(tickers, start_date, end_date):
    print(f"Downloading data for {tickers}...")
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data['Close']
        except KeyError:
            print("Warning: 'Close' level not found, trying 'Adj Close'...")
            data = data['Adj Close']
            
    if data.empty:
        raise ValueError("Data download failed or returned empty.")
        
    return data.dropna()

def calculate_spread(df, stock_A, stock_B):
    if stock_A not in df.columns or stock_B not in df.columns:
        raise KeyError(f"Tickers {stock_A} or {stock_B} not found in downloaded data. Available: {df.columns.tolist()}")

    X = df[stock_B]
    Y = df[stock_A]
    X_const = sm.add_constant(X)
    
    model = sm.OLS(Y, X_const).fit()
    hedge_ratio = model.params[stock_B]
    
    print(f"\nCalculated Hedge Ratio: {hedge_ratio:.4f}")
    
    spread = df[stock_A] - (hedge_ratio * df[stock_B])
    return spread, hedge_ratio

def check_cointegration(spread):
    adf_result = adfuller(spread)
    p_value = adf_result[1]
    print(f"\n--- ADF Test Results ---")
    print(f"P-Value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("RESULT: Cointegrated. Proceed to strategy.")
        return True
    else:
        print("RESULT: Not cointegrated. (Strategy might fail)")
        return False

def backtest_strategy(df, stock_A, stock_B, spread):
    window = 30
    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()
    z_score = (spread - spread_mean) / spread_std

    signals = pd.DataFrame(index=df.index)
    signals['signal'] = np.nan
    
    signals['signal'] = np.where(z_score > 1.5, -1, signals['signal']) 
    signals['signal'] = np.where(z_score < -1.5, 1, signals['signal']) 
    signals['signal'] = np.where(abs(z_score) < 0.5, 0, signals['signal']) 
    
    signals['signal'] = signals['signal'].ffill().fillna(0)
    
    ret_A = df[stock_A].pct_change()
    ret_B = df[stock_B].pct_change()
    
    strategy_returns = signals['signal'].shift(1) * (ret_A - ret_B)
    
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    if strategy_returns.std() == 0:
        sharpe_ratio = 0
    else:
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    
    final_return = (cumulative_returns.iloc[-1] - 1) * 100
    
    print("\n--- Backtest Performance ---")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Total Return: {final_return:.2f}%")
    
    return cumulative_returns, z_score

if __name__ == "__main__":
    try:
        tickers = ['GLD', 'GDX'] 
        df = get_data(tickers, '2020-01-01', '2024-01-01')

        spread, ratio = calculate_spread(df, 'GLD', 'GDX')
        is_tradable = check_cointegration(spread)

        cumulative_returns, z_score = backtest_strategy(df, 'GLD', 'GDX', spread)

        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        z_score.plot(label='Z-Score')
        plt.axhline(1.5, color='red', linestyle='--', label='Short Threshold')
        plt.axhline(-1.5, color='green', linestyle='--', label='Long Threshold')
        plt.legend()
        plt.title("Spread Z-Score Signal")
        
        plt.subplot(2, 1, 2)
        cumulative_returns.plot(label='Strategy Returns', color='blue')
        plt.title("Cumulative PnL")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('strategy_results.png')
        print("\nChart saved as 'strategy_results.png'. Open it to see your PnL!")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")