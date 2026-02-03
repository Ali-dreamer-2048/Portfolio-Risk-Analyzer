# Portfolio Risk Analyzer

A Python-based tool for portfolio optimization and risk assessment using real market data.  
Developed as part of CityU Year 1 Fintech/Quantitative Finance learning project.

## Features

- **Efficient Frontier Simulation**
  - Monte Carlo simulation for 10,000+ random portfolios
  - Identify optimal portfolios: Max Sharpe Ratio & Minimum Volatility
  - Interactive scatter plot with color-coded Sharpe Ratio

- **Value at Risk (VaR) Calculator**
  - Historical, Parametric (Gaussian), and Monte Carlo VaR methods
  - Distribution histogram with VaR thresholds marked
  - Supports customizable confidence levels

## Technologies Used

- yfinance (real-time stock data)
- pandas & NumPy (data processing)
- matplotlib & scipy (optimization & visualization)

## Installation & Run

1. Clone the repository:
(bash)
   git clone https://github.com/Ali-dreamer-2048/Portfolio-Risk-Analyzer.git
   cd Portfolio-Risk-Analyzer
   
2. Install dependencies:
(Bash)
pip install -r requirements.txt
3. Run the scripts:
(Bash)
python portfolio_efficient_frontier.py
python var_risk_calculator.py