# We load the necessary libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import quantstats as qs
import statsmodels.api as sm
import seaborn as sns
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import grangercausalitytests
from arch.unitroot import ADF, PhillipsPerron, KPSS
from sklearn.metrics import confusion_matrix
import warnings

# we ignore deprecation warnings and futurewarnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning) # ignore future warnings
warnings.simplefilter(action="ignore", category=UserWarning)  # ignore warnings
warnings.simplefilter(action="ignore", category=RuntimeWarning)  # ignore runtime warnings

from functions.plot_heatmap import plot_heatmap
from functions.plot_positions import plot_positions_ma
from functions.plot_positions import plot_positions_2mas
from functions.plot_positions import plot_positions_vb
from functions.position_VB import positionVB

# lets add the functions path to sys.path
import sys
sys.path.append('functions')

# Loading the data

quarters = ['2023_Q1', '2023_Q3', '2023_Q4',
            '2024_Q2', '2024_Q4',
            '2025_Q1', '2025_Q2']

# create an empty DataFrame to store summary for all quarters
summary_data1_all_quarters = pd.DataFrame()

# Additional local functions

def mySR(x, scale):
    return np.sqrt(scale) * np.nanmean(x) / np.nanstd(x)

#Application of customized function
def run_volatility_breakout(
    price,
    fast_span,
    slow_span,
    vol_window,
    m,
    pos_flat,
    point_value
):
    # EMAs
    fast = price.ewm(span=fast_span).mean()
    slow = price.ewm(span=slow_span).mean()

    # Volatility
    vol = price.rolling(vol_window).std()

    # NaN handling
    fast[price.isna()] = np.nan
    slow[price.isna()] = np.nan
    vol[price.isna()] = np.nan

    # Positions
    pos = positionVB(
        signal=fast.to_numpy(),
        lower=(slow - m * vol).to_numpy(),
        upper=(slow + m * vol).to_numpy(),
        pos_flat=pos_flat,
        strategy="mom"
    )

    # PnL
    pnl_gross = np.where(
        np.isnan(pos * price.diff()),
        0,
        pos * price.diff() * point_value
    )

    ntrans = np.abs(np.diff(pos, prepend=0))
    pnl_net = pnl_gross - ntrans * 12

    return pnl_gross, pnl_net, ntrans

#Plotting net PnL for entire data
def plot_total_net_pnl(pnl_net_dict, symbol):
    pnl_all = pd.concat(pnl_net_dict.values()).sort_index()

    plt.figure(figsize=(12, 6))
    plt.plot(pnl_all.cumsum(), linewidth=2)

    plt.title(f'Cumulative Net PnL – {symbol} (All Quarters)')
    plt.xlabel('Date')
    plt.ylabel('PnL [$]')
    plt.grid(True)

    plt.show()

#Plotting net PnL for quarter-to-quarter
def plot_quarter_pnl(pnl_gross_d, pnl_net_d, quarter, symbol, save=True):
    plt.figure(figsize=(12, 6))

    plt.plot(
        pnl_gross_d.cumsum(),
        label="Gross PnL",
        linewidth=2
    )
    plt.plot(
        pnl_net_d.cumsum(),
        label="Net PnL",
        linewidth=2
    )

    plt.title(f"Cumulative Gross & Net PnL – {symbol} ({quarter})")
    plt.xlabel("Date")
    plt.ylabel("PnL [$]")
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig(
            f"data1_{quarter}_{symbol}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()
    else:
        plt.show()

#Defining position of strategy in breakout model
def volatility_breakout_position(signal, slow, vol, m, pos_flat):
    upper = slow + m * vol
    lower = slow - m * vol

    sig_lag = signal.shift(1)
    upper_lag = upper.shift(1)
    lower_lag = lower.shift(1)

    pos = np.where(
        sig_lag.notna() & upper_lag.notna() & lower_lag.notna(),
        np.where(sig_lag > upper_lag, 1,
        np.where(sig_lag < lower_lag, -1, 0)),
        np.nan
    )

    pos[pos_flat == 1] = 0
    return pos

#Computing proper metrics for evaluation
def compute_metrics(pnl_gross, pnl_net, price_series, ntrans, scale=252):
    idx = price_series.index

    pnl_gross_d = (
        pd.Series(pnl_gross, index=idx)
        .groupby(idx.date)
        .sum()
    )

    pnl_net_d = (
        pd.Series(pnl_net, index=idx)
        .groupby(idx.date)
        .sum()
    )

    ntrans_d = (
        pd.Series(ntrans, index=idx)
        .groupby(idx.date)
        .sum()
    )

    return {
        "pnl_gross_d": pnl_gross_d,
        "pnl_net_d": pnl_net_d,

        "gross_PnL": pnl_gross_d.sum(),
        "net_PnL": pnl_net_d.sum(),

        "gross_SR": mySR(pnl_gross_d, scale=scale),
        "net_SR": mySR(pnl_net_d, scale=scale),

        "gross_CR": pnl_gross_d.mean() / pnl_gross_d.std(),
        "net_CR": pnl_net_d.mean() / pnl_net_d.std(),

        "av_daily_ntrans": ntrans_d.mean(),
    }

#FINAL SUMMARY OF DEVELOPPED MODEL
pnl_net_store_NQ = {}
pnl_net_store_SP = {}
summary_data1_all_quarters = pd.DataFrame()

for quarter in quarters:

    data1 = pd.read_parquet(f"data/data1_{quarter}.parquet")
    data1.set_index("datetime", inplace=True)

    # ---------------- Assumption 1 ----------------
    data1.loc[data1.between_time("9:31", "9:40").index] = np.nan
    data1.loc[data1.between_time("15:51", "16:00").index] = np.nan

    # ---------------- Assumption 2 ----------------
    pos_flat = np.zeros(len(data1))
    pos_flat[data1.index.time <= pd.to_datetime("9:55").time()] = 1
    pos_flat[data1.index.time >= pd.to_datetime("15:40").time()] = 1

    # ==================================================
    # =================== NASDAQ =======================
    # ==================================================
    signalEMA_NQ = data1["NQ"].ewm(span=10).mean()
    slowEMA_NQ   = data1["NQ"].ewm(span=120).mean()
    vol_NQ       = data1["NQ"].rolling(120).std()

    pos_NQ = volatility_breakout_position(
        signalEMA_NQ, slowEMA_NQ, vol_NQ, m=2, pos_flat=pos_flat
    )

    price_diff = data1["NQ"].diff()
    pnl_gross  = np.nan_to_num(pos_NQ * price_diff * 20)
    ntrans     = np.abs(np.diff(np.nan_to_num(pos_NQ), prepend=0))
    pnl_net    = pnl_gross - ntrans * 12

    m_NQ = compute_metrics(pnl_gross, pnl_net, data1["NQ"], ntrans)

    stat_NQ = (m_NQ["net_SR"] - 0.5) * np.maximum(
        0, np.log(np.abs(m_NQ["net_PnL"] / 1000))
    )

    pnl_net_store_NQ[quarter] = m_NQ["pnl_net_d"]

    plot_quarter_pnl(
        m_NQ["pnl_gross_d"],
        m_NQ["pnl_net_d"],
        quarter,
        "NQ"
    )

    # ==================================================
    # =================== S&P500 =======================
    # ==================================================
    signalEMA_SP = data1["SP"].ewm(span=20).mean()
    slowEMA_SP   = data1["SP"].ewm(span=150).mean()
    vol_SP       = data1["SP"].rolling(60).std()

    pos_SP = volatility_breakout_position(
        signalEMA_SP, slowEMA_SP, vol_SP, m=1, pos_flat=pos_flat
    )

    price_diff = data1["SP"].diff()
    pnl_gross  = np.nan_to_num(pos_SP * price_diff * 50)
    ntrans     = np.abs(np.diff(np.nan_to_num(pos_SP), prepend=0))
    pnl_net    = pnl_gross - ntrans * 12

    m_SP = compute_metrics(pnl_gross, pnl_net, data1["SP"], ntrans)

    stat_SP = (m_SP["net_SR"] - 0.5) * np.maximum(
        0, np.log(np.abs(m_SP["net_PnL"] / 1000))
    )

    pnl_net_store_SP[quarter] = m_SP["pnl_net_d"]

    plot_quarter_pnl(
        m_SP["pnl_gross_d"],
        m_SP["pnl_net_d"],
        quarter,
        "SP"
    )

    # ---------------- Summary ----------------
    summary_data1_all_quarters = pd.concat(
        [
            summary_data1_all_quarters,
            pd.DataFrame({
                "quarter": quarter,

                "gross_SR_NQ": m_NQ["gross_SR"],
                "net_SR_NQ":   m_NQ["net_SR"],
                "gross_PnL_NQ": m_NQ["gross_PnL"],
                "net_PnL_NQ":   m_NQ["net_PnL"],
                "gross_CR_NQ":  m_NQ["gross_CR"],
                "net_CR_NQ":    m_NQ["net_CR"],
                "av_daily_ntrans_NQ": m_NQ["av_daily_ntrans"],
                "stat_NQ": stat_NQ,

                "gross_SR_SP": m_SP["gross_SR"],
                "net_SR_SP":   m_SP["net_SR"],
                "gross_PnL_SP": m_SP["gross_PnL"],
                "net_PnL_SP":   m_SP["net_PnL"],
                "gross_CR_SP":  m_SP["gross_CR"],
                "net_CR_SP":    m_SP["net_CR"],
                "av_daily_ntrans_SP": m_SP["av_daily_ntrans"],
                "stat_SP": stat_SP,
            }, index=[0])
        ],
        ignore_index=True
    )

# =================== FULL SAMPLE PLOTS ===================
plot_total_net_pnl(pnl_net_store_NQ, "NQ")
plot_total_net_pnl(pnl_net_store_SP, "SP")

summary_data1_all_quarters

#Cumulative PnL value
total_net_pnl_nq = summary_data1_all_quarters['net_PnL_NQ'].sum()
total_net_pnl_nq

#Evaluation by defined 'stat' metric
summary_data1_all_quarters.sort_values(by = 'stat_NQ', 
                                 ascending = False)
#Saving final output
summary_data1_all_quarters.to_csv("summary_data1_all_quarters.csv", index=False)
