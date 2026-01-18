# ---------------------------
# final_strategy_group2.py
# ---------------------------

# Load necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quantstats as qs

quarters = ['2023_Q1', '2023_Q3', '2023_Q4',
            '2024_Q2', '2024_Q4',
            '2025_Q1', '2025_Q2']

# ---------------------------
# Strategy: final Group2 XAG mean-reversion
# ---------------------------

POINT_VALUE = 5000
TCOST = 10

BEST_PARAMS = {
    'win': 64,
    'z': 0.675,
    'hold_min': 3,
    'trend_span': 100,
    'vol_cap': 6,
    'stop_mult': 2.0
}

# ---------------------------
# Utility functions
# ---------------------------

def mySR(x, scale=252):
    return np.sqrt(scale) * np.nanmean(x) / np.nanstd(x) if np.nanstd(x) != 0 else 0.0

def calmar_ratio(cum_pnl):
    dd = cum_pnl.cummax() - cum_pnl
    max_dd = dd.max()
    return cum_pnl.iloc[-1] / max_dd if max_dd != 0 else np.nan

def mean_reversion(price, win, z):
    ma = price.rolling(win).mean()
    std = price.rolling(win).std()
    zscore = (price - ma) / std
    return np.where(zscore > z, -1, np.where(zscore < -z, 1, 0))

def trend_filter(price, slow):
    ema = price.ewm(span=slow, adjust=False).mean()
    return price > ema

def enforce_min_hold(pos, min_hold):
    pos = pos.copy()
    last, cnt = 0, 0
    for i in range(len(pos)):
        if pos[i] != last:
            last = pos[i]
            cnt = 0
        if last != 0:
            cnt += 1
            if cnt < min_hold:
                pos[i] = 0
    return pos

def apply_group2_constraints(index):
    pos_flat = np.zeros(len(index))
    t = index.time
    # exit 10 minutes before break
    pos_flat[(t >= pd.to_datetime("16:50").time()) & (t < pd.to_datetime("18:00").time())] = 1
    # do not trade first 10 minutes after break
    pos_flat[(t >= pd.to_datetime("18:00").time()) & (t <= pd.to_datetime("18:10").time())] = 1
    return pos_flat

def apply_soft_vol_filter(price, pos, win, vol_cap):
    vol = price.rolling(win).std()
    scale = (1 / vol).clip(upper=vol_cap).fillna(0)
    return pos * scale

def compute_gross_pnl(price, pos):
    ret = price.diff().fillna(0)
    return pos.shift(1) * ret * POINT_VALUE

def compute_net_pnl(price, pos):
    ret = price.diff().fillna(0)
    trades = pos.diff().abs().fillna(0)
    return pos.shift(1) * ret * POINT_VALUE - trades * TCOST

def apply_hard_stop(price, pos, win, mult, cooldown=2):
    pos = pos.copy()
    pnl = compute_net_pnl(price, pos)
    vol = price.rolling(win).std() * POINT_VALUE
    cum = 0.0
    cooldown_counter = 0
    for i in range(len(pnl)):
        if cooldown_counter > 0:
            pos.iloc[i] = 0
            cooldown_counter -= 1
            continue
        cum += pnl.iloc[i]
        if abs(cum) > mult * vol.iloc[i]:
            pos.iloc[i] = 0
            cum = 0.0
            cooldown_counter = cooldown
    return pos

# ---------------------------
# Storage
# ---------------------------

summary_data_all_quarters = pd.DataFrame()
equity_curves = {}
summary_stat_total = 0.0

# ---------------------------
# Main loop per quarter
# ---------------------------

for quarter in quarters:
    print(f"Processing {quarter}")

    # Load data
    data = pd.read_parquet(f'data/data2_{quarter}.parquet')
    data.set_index('datetime', inplace=True)
    price = data['XAG']

    # Group2 flat periods
    pos_flat = apply_group2_constraints(price.index)

    # Generate mean-reversion signal
    pos = pd.Series(mean_reversion(price, BEST_PARAMS['win'], BEST_PARAMS['z']), index=price.index)

    # Trend filter
    tf = trend_filter(price, BEST_PARAMS['trend_span'])
    pos = np.where((pos == 1) & tf, 0, pos)
    pos = np.where((pos == -1) & (~tf), 0, pos)
    pos = pd.Series(pos, index=price.index)

    # Volatility scaling (soft)
    pos = apply_soft_vol_filter(price, pos, BEST_PARAMS['win'], BEST_PARAMS['vol_cap'])

    # Minimum hold
    pos = enforce_min_hold(pos, BEST_PARAMS['hold_min'])

    # Group2 constraints
    pos[pos_flat == 1] = 0

    # Hard stop
    pos = apply_hard_stop(price, pos, BEST_PARAMS['win'], BEST_PARAMS['stop_mult'])

    # Volatility-scaled sizing
    vol = price.rolling(BEST_PARAMS['win']).std()
    size = (1 / vol).clip(upper=BEST_PARAMS['vol_cap']).fillna(0)
    size /= size.median()
    pos = pos * size

    # ---------------------------
    # PnL calculations
    # ---------------------------
    pnl_gross = compute_gross_pnl(price, pos)
    pnl_net = compute_net_pnl(price, pos)

    # Daily aggregation
    pnl_gross_d = pnl_gross.fillna(0).groupby(pnl_gross.index.date).sum()
    pnl_net_d = pnl_net.fillna(0).groupby(pnl_net.index.date).sum()
    ntrades_d = pos.diff().abs().fillna(0).groupby(pos.index.date).sum()

    # Metrics
    gross_SR = mySR(pnl_gross_d)
    net_SR = mySR(pnl_net_d)
    gross_CR = calmar_ratio(pnl_gross_d.cumsum())
    net_CR = calmar_ratio(pnl_net_d.cumsum())
    gross_PnL = pnl_gross_d.sum()
    net_PnL = pnl_net_d.sum()
    av_ntrades = ntrades_d.mean()

    # Summary statistic
    stat = (net_SR - 0.5) * np.maximum(0, np.log(np.abs(net_PnL/1000)))
    summary_stat_total += stat

    # Collect summary
    summary = pd.DataFrame({'quarter': quarter,
                            'gross_SR': gross_SR,
                            'net_SR': net_SR,
                            'gross_PnL': gross_PnL,
                            'net_PnL': net_PnL,
                            'gross_CR': gross_CR,
                            'net_CR': net_CR,
                            'av_daily_ntrans': av_ntrades,
                            'stat': stat}, index=[0])
    summary_data_all_quarters = pd.concat([summary_data_all_quarters, summary], ignore_index=True)

    # Equity curve
    equity_curves[quarter] = pnl_net_d.cumsum()
    plt.figure(figsize=(12,6))
    plt.plot(pnl_gross_d.cumsum(), label='Gross PnL', color='blue')
    plt.plot(pnl_net_d.cumsum(), label='Net PnL', color='red')
    plt.title(f'Cumulative Gross and Net PnL ({quarter})')
    plt.legend()
    plt.grid(axis='x')
    plt.savefig(f"data2_{quarter}.png", dpi=300, bbox_inches='tight')
    plt.close()

# ---------------------------
# Save final summary
# ---------------------------
summary_data_all_quarters['stat_total'] = summary_stat_total
summary_data_all_quarters.to_csv('summary_data2_all_quarters.csv', index=False)

print(summary_data_all_quarters)
print(f"Total summary statistic across all quarters: {summary_stat_total:.4f}")
