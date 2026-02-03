import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# 1. Select 5 assets
tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']

# 2. Download data
# Added auto_adjust=True to ensure adjusted prices and handle potential MultiIndex
raw_data = yf.download(tickers, start="2020-01-01", progress=False)

# 3. [Fix point] Safely extract Adj Close
# Modern yfinance versions typically use 'Price' as first level, 'Ticker' as second level
# 3. Safely extract Adjusted Close prices
# yfinance may return:
# - MultiIndex columns: levels=['Price', 'Ticker'] with 'Adj Close' / 'Close' under 'Price'
# - Single ticker: flat columns like 'Adj Close', 'Close', 'Open', etc.
# - Rare cases: different naming or structure

if isinstance(raw_data.columns, pd.MultiIndex):
    # Multi-ticker case (most common when downloading multiple symbols)
    if 'Adj Close' in raw_data.columns.levels[0]:
        adj_close = raw_data['Adj Close']
    elif 'Close' in raw_data.columns.levels[0]:
        adj_close = raw_data['Close']
        print("Warning: Only 'Close' found, not 'Adj Close'. Using unadjusted prices.")
    else:
        raise KeyError("Cannot find 'Adj Close' or 'Close' in MultiIndex columns")
else:
    # Single ticker or flat columns case
    if 'Adj Close' in raw_data.columns:
        adj_close = raw_data['Adj Close']
    elif 'Close' in raw_data.columns:
        adj_close = raw_data['Close']
        print("Warning: Only 'Close' found, not 'Adj Close'. Using unadjusted prices.")
    else:
        # Very rare fallback using xs (cross-section)
        try:
            adj_close = raw_data.xs('Adj Close', axis=1, level=0)
        except KeyError:
            raise KeyError("Cannot locate 'Adj Close' or 'Close' in the downloaded data")


# Optional: forward/backward fill any missing values (common in illiquid stocks)
# 4. Fill missing values and calculate daily returns
adj_close = adj_close.ffill().bfill()
returns = adj_close.pct_change().dropna()

# 5. Annualized mean and covariance matrix
mu = returns.mean() * 252
cov_matrix = returns.cov() * 252

# 6. Monte Carlo simulation for 10,000 random portfolios
num_portfolios = 10000
results = np.zeros((3, num_portfolios))
weights_record = []

np.random.seed(42)

for i in range(num_portfolios):
    weights = np.random.random(len(tickers))
    weights /= weights.sum()
    weights_record.append(weights)

    port_return = np.dot(weights, mu)

    # Matrix calculation for risk: sqrt(W^T * Cov * W)
    port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = port_return / port_risk if port_risk > 0 else 0

    results[0, i] = port_return
    results[1, i] = port_risk
    results[2, i] = sharpe

# 7. Plot the chart
plt.figure(figsize=(12, 8))
scatter = plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Sharpe Ratio')

max_sharpe_idx = np.argmax(results[2])
min_vol_idx = np.argmin(results[1])

plt.scatter(results[1, max_sharpe_idx], results[0, max_sharpe_idx],
            c='red', s=200, marker='*', label='Max Sharpe Ratio')
plt.scatter(results[1, min_vol_idx], results[0, min_vol_idx],
            c='blue', s=200, marker='o', label='Min Volatility')

plt.title('Efficient Frontier - Monte Carlo Simulation', fontsize=18)
plt.xlabel('Annualized Risk (Volatility)', fontsize=14)
plt.ylabel('Annualized Expected Return', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 8. Output the best portfolio
best_weights = weights_record[max_sharpe_idx]

print("\n=== Best Sharpe Ratio Portfolio ===")
for ticker, weight in zip(tickers, best_weights):
    print(f"{ticker}: {weight:.1%}")
print(f"Expected Annualized Return: {results[0, max_sharpe_idx]:.1%}")
print(f"Annualized Risk: {results[1, max_sharpe_idx]:.1%}")
print(f"Sharpe Ratio: {results[2, max_sharpe_idx]:.2f}")
