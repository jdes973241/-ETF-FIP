import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import zscore
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ==========================================
# 頁面設定
# ==========================================
st.set_page_config(page_title="多重資產動能策略", layout="wide")
st.title("🛡️ 多重資產因子動能輪動策略 (Live & Backtest)")

# ==========================================
# 核心邏輯函數
# ==========================================
def calculate_daily_beta(asset, bench, daily_df, lookback=252):
    subset = daily_df[[asset, bench]].dropna().tail(lookback)
    if len(subset) < lookback * 0.8: return 1.0
    cov = np.cov(subset[asset], subset[bench])
    return cov[0, 1] / cov[1, 1]

def calculate_fip(daily_series, lookback=252):
    """計算 FIP: 過去 lookback 天數中，正報酬天數的佔比"""
    subset = daily_series.tail(lookback).dropna()
    if len(subset) < lookback * 0.5: return np.nan
    return (subset > 0).sum() / len(subset)

@st.cache_data(ttl=3600)
def load_all_data():
    # 1. 定義所有需要的標的 (包含即時監控用 & 回測用)
    
    # A. 即時監控池 (Live)
    live_assets = {
        'IMOM', 'IVAL', 'IDHQ', 'GWX', 
        'QMOM', 'QVAL', 'SPHQ', 'SCHA', 
        'PIE',  'DFEV', 'DEHP', 'EEMS'
    }
    
    # B. 回測池 (Backtest): DFEV->DFEVX, 去除 DEHP
    backtest_assets = {
        'IMOM', 'IVAL', 'IDHQ', 'GWX', 
        'QMOM', 'QVAL', 'SPHQ', 'SCHA', 
        'PIE',  'DFEVX', 'EEMS' # 這裡移除了 DEHP
    }
    
    # C. 基準與避險
    others = {'TLT', 'GLD', 'VT', 'EFA', 'VTI', 'EEM'} # 包含 Benchmarks
    
    all_symbols = list(live_assets | backtest_assets | others)

    # 2. 下載數據 (抓取最長歷史)
    start_date = '2000-01-01'
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    with st.spinner('正在下載並處理所有歷史數據 (2000年至今)...'):
        raw_data = yf.download(all_symbols, start=start_date, end=end_date, progress=False, auto_adjust=False)
    
    if 'Adj Close' in raw_data.columns:
        prices = raw_data['Adj Close']
    elif 'Close' in raw_data.columns:
        prices = raw_data['Close']
    else:
        return None, "❌ 嚴重錯誤: 無法下載價格資料"

    prices = prices.astype(float).ffill() # 填補空值
    
    if prices.empty:
        return None, "❌ 錯誤: 下載的數據為空。"

    # 3. 處理月份數據
    monthly_prices = prices.resample('ME').last()
    
    # 檢查本月是否已結束，若未結束則移除最後一筆 (避免 MTD 偏誤)
    current_date = datetime.now().date()
    last_idx = monthly_prices.index[-1]
    next_day = current_date + timedelta(days=1)
    
    msg = ""
    if last_idx.month == current_date.month and last_idx.year == current_date.year:
         if next_day.month == current_date.month: 
             msg = f"⚠️ 本月 ({last_idx.strftime('%Y-%m')}) 尚未結束，分析使用上個月底數據。"
             monthly_prices = monthly_prices.iloc[:-1]
             prices = prices.loc[:monthly_prices.index[-1]]
         else:
             msg = f"✅ 使用最新完整月份 ({last_idx.strftime('%Y-%m')}) 數據。"
    else:
         msg = f"✅ 使用最新完整月份 ({last_idx.strftime('%Y-%m')}) 數據。"

    return prices, monthly_prices, msg

# ==========================================
# 載入數據
# ==========================================
data_pack = load_all_data()
if data_pack is None or data_pack[0] is None:
    st.error("數據下載失敗")
    st.stop()

prices, monthly_prices, status_msg = data_pack
daily_ret = prices.pct_change()
monthly_ret = monthly_prices.pct_change()

# ==========================================
# PART 1: 即時監控儀表板 (Live Dashboard)
# ==========================================
st.markdown("---")
st.header("📡 本月策略訊號 (Live Dashboard)")
st.caption(status_msg)

# 定義即時監控用的資產池
live_assets_map = {
    'IMOM': 'EFA', 'IVAL': 'EFA', 'IDHQ': 'EFA', 'GWX': 'EFA',
    'QMOM': 'VTI', 'QVAL': 'VTI', 'SPHQ': 'VTI', 'SCHA': 'VTI',
    'PIE': 'EEM',  'DFEV': 'EEM', 'DEHP': 'EEM', 'EEMS': 'EEM'
}
live_tickers = list(live_assets_map.keys())
cutoff_date = monthly_prices.index[-1]

# 1. 市場狀態判斷 (Count >= 6)
periods = [3, 6, 9, 12]
neg_count = 0
valid_count = 0
regime_details = []

for ticker in live_tickers:
    try:
        p_now = monthly_prices.loc[cutoff_date, ticker]
        avg_mom = 0
        for p in periods:
            p_prev = monthly_prices.iloc[-1-p][ticker]
            avg_mom += (p_now / p_prev) - 1
        avg_mom /= 4
        
        if avg_mom < 0: neg_count += 1
        valid_count += 1
        regime_details.append({'Ticker': ticker, 'Avg_Mom': avg_mom})
    except: continue

THRESHOLD = 6
is_bull = neg_count < THRESHOLD

col1, col2, col3 = st.columns(3)
col1.metric("轉弱標的數量", f"{neg_count} / {valid_count}")
col2.metric("市場狀態", "牛市 (進攻)" if is_bull else "熊市 (避險)", delta="Risk-On" if is_bull else "Risk-Off", delta_color="normal" if is_bull else "inverse")

# 2. 策略執行
if not is_bull:
    # 避險模式
    st.warning("⚠️ 觸發避險機制 (Count >= 6)。比較 TLT 與 GLD 12個月報酬。")
    best_hedge = 'TLT'
    best_ret = -999
    hedge_data = []
    for asset in ['TLT', 'GLD']:
        try:
            p_now = monthly_prices.loc[cutoff_date, asset]
            p_12m = monthly_prices.iloc[-13][asset]
            r = (p_now / p_12m) - 1
            hedge_data.append({'Asset': asset, '12M Return': r})
            if r > best_ret:
                best_ret = r
                best_hedge = asset
        except: pass
    
    st.table(pd.DataFrame(hedge_data).style.format({'12M Return': '{:.2%}'}))
    st.success(f"🛡️ 建議持倉: **{best_hedge}** (100%)")

else:
    # 進攻模式
    st.success("✅ 市場狀態良好。執行選股 (Alpha Filter -> Ranking -> Top 3)。")
    
    survivors = []
    for ticker in live_tickers:
        bench = live_assets_map[ticker]
        try:
            # Beta & Alpha
            # 注意: 這裡只取最近 252 天計算 beta
            subset_daily = daily_ret.loc[:cutoff_date].tail(252)
            subset_daily_clean = subset_daily[[ticker, bench]].dropna()
            if len(subset_daily_clean) > 200:
                cov = np.cov(subset_daily_clean[ticker], subset_daily_clean[bench])
                beta = cov[0, 1] / cov[1, 1]
            else: beta = 1.0
            
            r_1m = monthly_ret.loc[cutoff_date, ticker]
            b_1m = monthly_ret.loc[cutoff_date, bench]
            alpha_1m = r_1m - (beta * b_1m)
            
            p_now = monthly_prices.loc[cutoff_date, ticker]
            p_12m = monthly_prices.iloc[-13][ticker]
            r_12m = (p_now / p_12m) - 1
            
            p_b_now = monthly_prices.loc[cutoff_date, bench]
            p_b_12m = monthly_prices.iloc[-13][bench]
            b_12m = (p_b_now / p_b_12m) - 1
            alpha_12m = r_12m - (beta * b_12m)
            
            if alpha_1m > 0 or alpha_12m > 0:
                survivors.append(ticker)
        except: continue
        
    # Ranking
    metrics_df = pd.DataFrame(index=survivors)
    for ticker in survivors:
        try:
            p_now = monthly_prices.loc[cutoff_date, ticker]
            for p in periods:
                p_prev = monthly_prices.iloc[-1-p][ticker]
                metrics_df.loc[ticker, f'R_{p}M'] = (p_now / p_prev) - 1
            metrics_df.loc[ticker, 'FIP'] = calculate_fip(daily_ret.loc[:cutoff_date, ticker])
        except: continue
        
    if not metrics_df.empty:
        z_df = pd.DataFrame(index=metrics_df.index)
        mom_z_cols = []
        for p in periods:
            z_df[f'Z_{p}M'] = zscore(metrics_df[f'R_{p}M'], ddof=1, nan_policy='omit')
            mom_z_cols.append(f'Z_{p}M')
        
        z_df['Avg_Mom_Z'] = z_df[mom_z_cols].mean(axis=1)
        z_df['Z_FIP'] = zscore(metrics_df['FIP'], ddof=1, nan_policy='omit')
        z_df['Score'] = 0.75 * z_df['Avg_Mom_Z'] + 0.25 * z_df['Z_FIP']
        
        top_3 = z_df.sort_values('Score', ascending=False).head(3).index.tolist()
        
        st.write("🏆 **Top 3 標的 (各 33.3%):**")
        c1, c2, c3 = st.columns(3)
        for i, t in enumerate(top_3):
            with [c1, c2, c3][i]:
                st.info(f"**{t}**")
                
        with st.expander("查看排名詳情"):
            # 合併原始數據顯示
            display_df = z_df[['Score', 'Avg_Mom_Z', 'Z_FIP']].copy()
            display_df = display_df.join(metrics_df)
            
            # 手動轉百分比 (解決格式化報錯問題)
            pct_cols = ['FIP', 'R_3M', 'R_6M', 'R_9M', 'R_12M']
            display_df[pct_cols] = display_df[pct_cols] * 100
            
            st.dataframe(
                display_df.sort_values('Score', ascending=False),
                column_config={
                    "Score": st.column_config.NumberColumn("總分", format="%.2f"),
                    "FIP": st.column_config.NumberColumn("FIP", format="%.2f%%"),
                    "R_3M": st.column_config.NumberColumn("3M", format="%.2f%%"),
                    "R_6M": st.column_config.NumberColumn("6M", format="%.2f%%"),
                    "R_9M": st.column_config.NumberColumn("9M", format="%.2f%%"),
                    "R_12M": st.column_config.NumberColumn("12M", format="%.2f%%"),
                }
            )
    else:
        st.error("沒有標的通過篩選。")

# ==========================================
# PART 2: 歷史回測分析 (Historical Backtest)
# ==========================================
st.markdown("---")
st.header("⏳ 歷史回測分析 (Backtest)")
st.markdown("""
**回測設定：**
* **回測標的**：使用 `DFEVX` 替代 `DFEV`，並剔除 `DEHP` 以最大化回測區間。
* **基準 (Benchmark)**：`VT` (Vanguard Total World Stock ETF)。
* **邏輯**：同上 (Count>=6 避險, TLT/GLD 輪動, Top 3 進攻)。
""")

if st.button("開始執行回測"):
    # 1. 定義回測用資產池
    backtest_assets_map = {
        'IMOM': 'EFA', 'IVAL': 'EFA', 'IDHQ': 'EFA', 'GWX': 'EFA',
        'QMOM': 'VTI', 'QVAL': 'VTI', 'SPHQ': 'VTI', 'SCHA': 'VTI',
        'PIE': 'EEM',  'DFEVX': 'EEM', 'EEMS': 'EEM' # 無 DEHP
    }
    backtest_tickers = list(backtest_assets_map.keys())
    
    # 2. 確定回測起始點 (所有標的都有數據的那一天)
    # 我們需要預留 12個月 + 1個月緩衝
    valid_starts = prices[backtest_tickers + ['TLT', 'GLD', 'VT']].apply(lambda x: x.first_valid_index())
    latest_start = valid_starts.max()
    warmup_days = 365 + 30
    required_start = latest_start + timedelta(days=warmup_days)
    
    start_idx = monthly_prices.index.searchsorted(required_start)
    
    if start_idx >= len(monthly_prices):
        st.error(f"數據不足以進行回測。最晚生效日期: {latest_start}")
        st.stop()
        
    st.info(f"回測區間: {monthly_prices.index[start_idx].date()} 至 {monthly_prices.index[-1].date()}")
    
    # 3. 執行回測迴圈
    portfolio_log = []
    dates = monthly_prices.index
    
    progress_bar = st.progress(0)
    total_steps = len(dates) - 1 - start_idx
    
    for i in range(start_idx, len(dates) - 1):
        curr_date = dates[i]
        next_date = dates[i+1]
        
        # 進度條
        step = i - start_idx
        progress_bar.progress(min(step / total_steps, 1.0))
        
        hist_daily = daily_ret.loc[:curr_date]
        hist_monthly = monthly_prices.loc[:curr_date]
        hist_monthly_ret = monthly_ret.loc[:curr_date]
        
        # --- A. 判斷市場狀態 (Count >= 6) ---
        neg_count = 0
        for t in backtest_tickers:
            try:
                p_now = hist_monthly.iloc[-1][t]
                avg_mom = 0
                for p in [3, 6, 9, 12]:
                    avg_mom += (p_now / hist_monthly.iloc[-1-p][t]) - 1
                if avg_mom < 0: neg_count += 1
            except: continue
            
        is_bear = neg_count >= 6
        
        selected_tickers = []
        
        if is_bear:
            # 避險: TLT vs GLD
            best_hedge = 'TLT'
            best_ret = -999
            for asset in ['TLT', 'GLD']:
                try:
                    p_now = hist_monthly.iloc[-1][asset]
                    p_prev = hist_monthly.iloc[-1-12][asset]
                    r = (p_now / p_prev) - 1
                    if r > best_ret:
                        best_ret = r
                        best_hedge = asset
                except: pass
            selected_tickers = [best_hedge]
            
        else:
            # 進攻: Top 3
            survivors = []
            for t in backtest_tickers:
                bench = backtest_assets_map[t]
                try:
                    # Beta Calc
                    subset = hist_daily[[t, bench]].tail(252).dropna()
                    if len(subset) > 200:
                        cov = np.cov(subset[t], subset[bench])
                        beta = cov[0, 1] / cov[1, 1]
                    else: beta = 1.0
                    
                    # Alpha Filter
                    r_1m = hist_monthly_ret.iloc[-1][t]
                    b_1m = hist_monthly_ret.iloc[-1][bench]
                    a_1m = r_1m - beta * b_1m
                    
                    p_now = hist_monthly.iloc[-1][t]
                    p_12m = hist_monthly.iloc[-13][t]
                    r_12m = (p_now / p_12m) - 1
                    
                    p_b_now = hist_monthly.iloc[-1][bench]
                    p_b_12m = hist_monthly.iloc[-13][bench]
                    b_12m = (p_b_now / p_b_12m) - 1
                    a_12m = r_12m - beta * b_12m
                    
                    if a_1m > 0 or a_12m > 0:
                        survivors.append(t)
                except: continue
            
            if survivors:
                metrics = []
                for t in survivors:
                    try:
                        p_now = hist_monthly.iloc[-1][t]
                        t_data = {'ticker': t}
                        for p in [3, 6, 9, 12]:
                            t_data[f'M_{p}'] = (p_now / hist_monthly.iloc[-1-p][t]) - 1
                        t_data['FIP'] = calculate_fip(hist_daily[t])
                        metrics.append(t_data)
                    except: continue
                
                if metrics:
                    m_df = pd.DataFrame(metrics).set_index('ticker')
                    z_df = pd.DataFrame(index=m_df.index)
                    mom_z_cols = []
                    for p in [3, 6, 9, 12]:
                        col = f'Z_{p}'
                        z_df[col] = zscore(m_df[f'M_{p}'], ddof=1, nan_policy='omit')
                        mom_z_cols.append(col)
                    
                    z_df['Avg_Mom_Z'] = z_df[mom_z_cols].mean(axis=1)
                    z_df['Z_FIP'] = zscore(m_df['FIP'], ddof=1, nan_policy='omit')
                    z_df['Score'] = 0.75 * z_df['Avg_Mom_Z'] + 0.25 * z_df['Z_FIP']
                    
                    selected_tickers = z_df.sort_values('Score', ascending=False).head(3).index.tolist()
            
            if not selected_tickers:
                selected_tickers = ['VT'] # Fallback
                
        # 計算績效
        final_ret = monthly_ret.loc[next_date, selected_tickers].mean()
        portfolio_log.append({'Date': next_date, 'Strategy': final_ret})
        
    progress_bar.empty()
    
    # 4. 分析結果
    res_df = pd.DataFrame(portfolio_log).set_index('Date')
    res_df['Equity'] = (1 + res_df['Strategy']).cumprod()
    res_df['DD'] = res_df['Equity'] / res_df['Equity'].cummax() - 1
    
    # Benchmark
    bench_ret = monthly_ret['VT'].loc[res_df.index]
    bench_equity = (1 + bench_ret).cumprod()
    res_df['Benchmark'] = bench_equity
    
    # 統計數據
    total_ret = res_df['Equity'].iloc[-1] - 1
    years = len(res_df) / 12
    cagr = (res_df['Equity'].iloc[-1]) ** (1/years) - 1
    mdd = res_df['DD'].min()
    
    neg_rets = res_df.loc[res_df['Strategy'] < 0, 'Strategy']
    down_std = neg_rets.std() * np.sqrt(12) if len(neg_rets) > 0 else 1e-6
    sortino = (res_df['Strategy'].mean() * 12) / down_std
    
    # 顯示指標
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("CAGR (年化報酬)", f"{cagr:.2%}")
    col_m2.metric("MDD (最大回撤)", f"{mdd:.2%}")
    col_m3.metric("Sortino Ratio", f"{sortino:.2f}")
    col_m4.metric("總報酬率", f"{total_ret:.2%}")
    
    # 繪圖
    st.subheader("📈 權益曲線 (Strategy vs VT)")
    chart_data = pd.concat([res_df['Equity'], bench_equity], axis=1)
    chart_data.columns = ['Strategy', 'Benchmark (VT)']
    st.line_chart(chart_data)
    
    st.subheader("📉 回撤圖 (Drawdown)")
    st.area_chart(res_df['DD'], color='#ff4b4b')
