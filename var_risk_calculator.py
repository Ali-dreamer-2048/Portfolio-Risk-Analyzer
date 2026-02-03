import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi


class QuantDeepDive:
    """
    Institutional-grade pair-wise fundamental comparison engine.
    Optimized for radar visualization and compact financial reporting.
    """

    def __init__(self, tickers):
        self.tickers = tickers
        self.raw_stats = {}

    def fetch_data(self, t1, t2):
        pair = [t1, t2]
        radar_metrics = []

        for tkr in pair:
            stock = yf.Ticker(tkr)
            info = stock.info

            # 1. Radar Analysis Dimensions (Scaled to avoid zero points)
            radar_metrics.append({
                'Ticker': tkr,
                'Net Margin (%)': info.get('profitMargins', 0) * 100,
                'Earnings Yield (%)': (1 / info.get('forwardPE', 1)) * 100 if info.get('forwardPE') else 0,
                'Debt Safety (1/D2E)': 1 / (info.get('debtToEquity', 100) / 100) if info.get('debtToEquity') else 0.5,
                'Rev Growth (%)': info.get('revenueGrowth', 0) * 100,
                'Return on Equity (%)': info.get('returnOnEquity', 0) * 100,
                'Price Stability (1/Beta)': 1 / info.get('beta', 1) if info.get('beta') else 1
            })

            # 2. Key Performance Indicators (KPIs) for the table
            self.raw_stats[tkr] = {
                'Market Cap ($B)': f"{info.get('marketCap', 0) / 1e9:.1f}B",
                'P/E Ratio (Fwd)': f"{info.get('forwardPE', 0):.2f}",
                'ROE (%)': f"{info.get('returnOnEquity', 0) * 100:.1f}%",
                'Profit Margin (%)': f"{info.get('profitMargins', 0) * 100:.1f}%",
                'Debt/Equity': f"{info.get('debtToEquity', 0):.1f}%",
                'FCF Yield (%)': f"{(info.get('freeCashflow', 0) / info.get('marketCap', 1)) * 100:.2f}%" if info.get(
                    'marketCap') else "N/A"
            }

        # 3. Enhanced Normalization: Offset by 0.2 to prevent 'zero-point' distortion
        df = pd.DataFrame(radar_metrics).set_index('Ticker')
        self.norm = df.copy()
        for col in df.columns:
            max_val = df[col].max() if df[col].max() != 0 else 1
            self.norm[col] = 0.2 + (df[col] / max_val) * 0.8

        return pd.DataFrame(self.raw_stats)

    def plot_integrated_report(self, t1, t2, table_df):
        categories = list(self.norm.columns)
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        # Colors: Deep Navy and Crimson Red
        colors = ['#003366', '#B22222']

        fig = plt.figure(figsize=(9, 11), facecolor='#FAFAFA')

        # --- Top Section: Radar Chart ---
        ax = plt.subplot(211, polar=True)
        ax.set_facecolor('#FFFFFF')
        plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.1)

        for i, (index, row) in enumerate(self.norm.iterrows()):
            values = row.values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, color=colors[i], linewidth=2.5, label=f"TARGET: {index}", zorder=3)
            ax.fill(angles, values, color=colors[i], alpha=0.12)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=8, fontweight='bold', color='#444444')
        ax.set_ylim(0, 1.1)
        plt.title(f"Strategic Quantitative Alpha: {t1} vs {t2}", size=14, pad=25, fontweight='bold')
        plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=9)

        # --- Bottom Section: Compact Financial Table ---
        ax_table = plt.subplot(212)
        ax_table.axis('off')

        # Styling the table for a compact, clean look
        the_table = ax_table.table(
            cellText=table_df.values,
            rowLabels=table_df.index,
            colLabels=table_df.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.25, 0.25]  # Tight column width
        )

        # Professional Table Formatting
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(9)
        the_table.scale(1.1, 1.6)  # Taller rows for readability, but narrower width

        # Header Styling
        for (row, col), cell in the_table.get_celld().items():
            if row == 0:
                cell.set_text_props(fontweight='bold', color='white')
                cell.set_facecolor('#333333')
            if col == -1:  # Row labels
                cell.set_text_props(fontweight='bold')
                cell.set_facecolor('#F2F2F2')

        # Add brief methodology text
        plt.figtext(0.5, 0.15, "Normalization: Min-Max scaled with 0.2 baseline to preserve geometric integrity.",
                    ha="center", fontsize=8, color='gray', style='italic')

        plt.show()


# --- Interactive Module ---
if __name__ == "__main__":
    t1 = input("Enter Primary Ticker (e.g., AAPL): ").upper() or "AAPL"
    t2 = input("Enter Benchmark Ticker (e.g., MSFT): ").upper() or "MSFT"

    engine = QuantDeepDive([t1, t2])
    print(f"Executing Deep Dive Analysis for {t1} vs {t2}...")

    try:
        data_table = engine.fetch_data(t1, t2)
        engine.plot_integrated_report(t1, t2, data_table)
    except Exception as e:
        print(f"Error: {e}. Please check ticker symbols and connection.")