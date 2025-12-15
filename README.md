# Statistical Arbitrage Strategy (Pairs Trading)

A quantitative trading strategy that exploits mean-reversion properties between cointegrated assets (e.g., Gold vs. Gold Miners).

## Overview

This project implements a classic **Pairs Trading** strategy using the **Engle-Granger two-step method**. It identifies pairs with stationary spreads and trades based on Z-Score deviations from the mean.

## Key Features

- **Cointegration Testing:** Uses Augmented Dickey-Fuller (ADF) test to validate pairs.
- **Signal Generation:** Dynamic Z-Score calculation on rolling spread windows.
- **Vectorized Backtest:** Efficient pandas-based engine to simulate performance.
- **Risk Metrics:** Calculates Sharpe Ratio, Cumulative Returns, and Drawdowns.

## Tech Stack

- **Python:** Core logic
- **Statsmodels:** OLS Regression & ADF Tests
- **Pandas/NumPy:** Data manipulation & Vectorization
- **YFinance:** Market Data ingestion

## Results (GLD/GDX)

- **Cumulative Return:** ~29.6% (2020-2024)
- **Strategy:** Mean Reversion on 2.0 Sigma bounds.
