import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
from scipy.stats import zscore
from datetime import datetime, timedelta
import pytz

# ==========================================
# 頁面設定
# ==========================================
st.set_page_config(page_title="多重資產動能策略", layout="wide")
st.title("🛡️ 多重資產因子動能輪動策略 (Live & Backtest)")
st.markdown("""
**策略邏輯摘要 (Sortino Optimized)：**
1.  **市場狀態 (Regime)**：計算 12 檔股票因子的平均動能 (3,6,9,12M)。若 **>= 6 檔** 動能轉負，則全面避險。
2.  **避險模式 (Risk-Off)**：比較 **TLT** 與 **GLD** 的 12 個月報酬，全倉持有強者。
3.  **進攻模式 (Risk-On)**：
    * **濾網**：Alpha (1M 或 12M > 0)。
    * **評分**：**Raw Sortino (3M+12M) $\\times$ FIP**。
    * **配置**：持有 **前 2 名**，等權重。
""")

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
    if len(subset) < lookback * 0.5: return 0.0
    return (subset > 0).sum() / len(subset)

def calculate_sortino(daily_series, lookback_months):
    """
    計算原始 Sortino Ratio
    Lookback 轉換: 1個月 約為 21 個交易日
    """
    days = int(lookback_months * 21)
    subset = daily_series.tail(days).dropna()
    
    if len(subset) < days * 0.5: return -999.0 # 資料不足
    
    # 1. 平均日報酬 (年化分子)
    avg_ret = subset.mean() * 252 
    
    # 2. 下行標準差 (年化分母)
    downside_returns = subset[subset < 0]
    
    if len(downside_returns) == 0:
        return 10.0 # 無下行風險，給予高分
        
    downside_std = downside_returns.std() * np.sqrt(252)
    
    if downside_std == 0:
        return 10.0
        
    return avg_ret / downside_std

@st.cache_data(ttl=3600)
def fetch_market_data(all_symbols, start_date, end_date):
    """純 I/O 函數，負責數據下載 (Threads=False)"""
    data = yf.download(all_symbols, start=start_date, end=end_date, progress=False, auto_adjust=False, threads=False)
    
    # 數據結構標準化
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.get_level_values(0):
            prices = data['Adj Close']
        elif 'Close' in data.columns.get_level_values(0):
            prices = data['Close']
        else:
            return None, "❌ 嚴重錯誤: 資料中無 Close 或 Adj Close"
    else:
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        elif 'Close' in data.columns:
            prices = data['Close']
        else:
            return None, "❌ 嚴重錯誤: 無法識別價格欄位"
            
    prices.columns.name = None
    return prices, None

def process_data_logic(prices, live_assets_map, backtest_assets, safe_pool, current_datetime):
    prices = prices.astype(float).ffill() 
    
    if prices.empty:
        return None, None, None, None, None, "❌ 錯誤: 下載的數據為空。"

    last_dt = prices.index[-1]
    if (current_datetime.replace(tzinfo=None) - last_dt.replace(tzinfo=None)).days > 7:
        st.warning(f"⚠️ 注意：最新數據日期為 {last_dt.strftime('%Y-%m-%d')}，可能非即時數據。")

    monthly_prices = prices.resample('ME').last()
    
    current_date_only = current_datetime.date()
    last_idx = monthly_prices.index[-1]
    next_day = current_date_only + timedelta(days=1)
    
    msg = ""
    if last_idx.month == current_date_only.month and last_idx.year == current_date_only.year:
         if next_day.month == current_date_only.month: 
             msg = f"⚠️ 本月 ({last_idx.strftime('%Y-%m')}) 尚未結束，使用上個月底數據進行分析。"
             monthly_prices = monthly_prices.iloc[:-1]
             prices = prices.loc[:monthly_prices.index[-1]]
         else:
             msg = f"✅ 使用最新完整月份 ({last_idx.strftime('%Y-%m')}) 數據。"
    else:
         msg = f"✅ 使用最新完整月份 ({last_idx.strftime('%Y-%m')}) 數據。"

    cutoff_date = monthly_prices.index[-1]
    monthly_ret = monthly_prices.pct_change()
    daily_ret = prices.pct_change()
    
    return prices, monthly_ret, daily_ret, monthly_prices, cutoff_date, msg

# ==========================================
# 數據準備與參數配置
# ==========================================

# 1. Live 資產池 (新增 EQLT)
live_assets_map = {
    'IMOM': 'EFA', 'IVAL': 'EFA', 'IDHQ': 'EFA', 'ISCF': 'EFA', 
    'QMOM': 'VTI', 'QVAL': 'VTI', 'SPHQ': 'VTI', 'FDM': 'VTI',  
    'PIE': 'EEM',  'DFEVX': 'EEM', 'EWX': 'EEM', 'EQLT': 'EEM'  # 新增 EQLT
}

# 2. Backtest 資產池 (維持 11 檔，不含 EQLT)
backtest_assets = [
    'IMOM', 'IVAL', 'IDHQ', 'ISCF', 
    'QMOM', 'QVAL', 'SPHQ', 'FDM',  
    'PIE',  'DFEVX', 'EWX'          
]

safe_pool = ['TLT', 'GLD']
others = ['VT'] 

all_symbols = list(set(list(live_assets_map.keys()) + list(live_assets_map.values()) + backtest_assets + safe_pool + others))

tz = pytz.timezone('Asia/Taipei')
now_tw = datetime.now(tz)
start_date_str = '2000-01-01'
end_date_str = (now_tw + timedelta(days=1)).strftime('%Y-%m-%d')

with st.spinner('正在下載所有歷史數據 (Live & Backtest)...'):
    raw_prices, error_msg = fetch_market_data(all_symbols, start_date_str, end_date_str)

if raw_prices is None:
    st.error(error_msg)
    st.stop()

prices, monthly_ret, daily_ret, monthly_prices, cutoff_date, status_msg = process_data_logic(
    raw_prices, live_assets_map, backtest_assets, safe_pool, now_tw
)

if prices is None:
    st.error(status_msg)
    st.stop()

equity_tickers = list(live_assets_map.keys())

with st.sidebar:
    st.header("📈 市場快照")
    st.info(f"分析基準日: {cutoff_date.strftime('%Y-%m-%d')}")
    st.caption(status_msg)
    try:
        vti_p = monthly_prices.loc[cutoff_date, 'VTI']
        tlt_p = monthly_prices.loc[cutoff_date, 'TLT']
        st.metric("VTI (美股)", f"{vti_p:.2f}")
        st.metric("TLT (美債)", f"{tlt_p:.2f}")
    except: pass
    st.divider()

# ==========================================
# 第一階段：市場狀態判斷 (維持不變)
# ==========================================
st.subheader("1️⃣ 第一階段：市場狀態判斷 (Regime Filter)")

hedge_periods = [3, 6, 9, 12]
regime_stats = []
neg_count = 0 
valid_count = 0

for ticker in equity_tickers:
    try:
        p_now = monthly_prices.loc[cutoff_date, ticker]
        ticker_avg_mom = 0
        p_vals = []
        
        for p in hedge_periods:
            p_prev = monthly_prices.iloc[-1-p][ticker] 
            r = (p_now / p_prev) - 1
            ticker_avg_mom += r
            p_vals.append(r)
            
        ticker_avg_mom /= 4
        
        status_icon = "🟢" if ticker_avg_mom > 0 else "🔴"
        if ticker_avg_mom < 0:
            neg_count += 1
            
        regime_stats.append({
            'Ticker': ticker,
            'Status': status_icon,
            'Avg_Mom': ticker_avg_mom,
            '3M': p_vals[0], '6M': p_vals[1], '9M': p_vals[2], '12M': p_vals[3]
        })
        
        if not np.isnan(ticker_avg_mom):
            valid_count += 1
    except Exception as e:
        continue

THRESHOLD_N = 6
is_bull_market = neg_count < THRESHOLD_N

col1, col2 = st.columns([1, 2])
col1.metric("轉弱標的數量 (Count < 0)", f"{neg_count} / {valid_count}", delta_color="inverse")
status_text = "🐂 牛市 (進攻模式)" if is_bull_market else "🐻 熊市 (避險模式)"
status_color = "green" if is_bull_market else "red"
col2.markdown(f"### 市場狀態: :{status_color}[{status_text}]")
col2.caption(f"避險觸發條件：轉弱標的數量 >= {THRESHOLD_N} (總數 12)")

with st.expander("查看全市場 12 檔 ETF 動能細節"):
    df_regime = pd.DataFrame(regime_stats)
    cols = ['Ticker', 'Status', 'Avg_Mom', '3M', '6M', '9M', '12M']
    st.dataframe(df_regime[cols].style.format("{:.2%}", subset=['Avg_Mom', '3M', '6M', '9M', '12M']))

st.divider()

# ==========================================
# 第二階段：策略分支
# ==========================================

if not is_bull_market:
    # 🐻 避險模式
    st.header("2️⃣ 第二階段 (A)：避險模式 (Risk-Off)")
    st.info("全市場動能 < 0，啟動避險。比較 TLT 與 GLD 的 12 個月報酬率。")
    
    hedge_stats = []
    best_hedge = None
    best_hedge_ret = -999
    
    for asset in safe_pool:
        try:
            p_now = monthly_prices.loc[cutoff_date, asset]
            p_12m = monthly_prices.iloc[-13][asset]
            r_12m = (p_now / p_12m) - 1
            
            hedge_stats.append({'Asset': asset, '12M Return': r_12m})
            
            if r_12m > best_hedge_ret:
                best_hedge_ret = r_12m
                best_hedge = asset
        except:
            st.warning(f"缺少 {asset} 數據")

    df_hedge = pd.DataFrame(hedge_stats)
    df_hedge['Selected'] = df_hedge['Asset'].apply(lambda x: '✅' if x == best_hedge else '')
    st.dataframe(df_hedge.style.format({'12M Return': '{:.2%}'}), use_container_width=False)
    st.success(f"🛡️ 本月建議持倉: **{best_hedge}** (100% 權重)")

else:
    # 🐂 進攻模式
    st.header("2️⃣ 第二階段 (B)：進攻模式 (Risk-On)")
    
    # --- Alpha Filter ---
    st.subheader("篩選：Alpha 濾網")
    st.caption("條件：(1M Alpha > 0) OR (12M Alpha > 0)")
    
    survivors = []
    filter_data = []
    
    for ticker in equity_tickers:
        bench = live_assets_map[ticker]
        try:
            beta = calculate_daily_beta(ticker, bench, daily_ret, lookback=252)
            r_asset_1m = monthly_ret.loc[cutoff_date, ticker]
            r_bench_1m = monthly_ret.loc[cutoff_date, bench]
            alpha_1m = r_asset_1m - (beta * r_bench_1m)
            
            p_now = monthly_prices.loc[cutoff_date, ticker]
            p_12m = monthly_prices.iloc[-13][ticker]
            r_asset_12m = (p_now / p_12m) - 1
            p_b_now = monthly_prices.loc[cutoff_date, bench]
            p_b_12m = monthly_prices.iloc[-13][bench]
            r_bench_12m = (p_b_now / p_b_12m) - 1
            alpha_12m = r_asset_12m - (beta * r_bench_12m)
            
            is_pass = (alpha_1m > 0) or (alpha_12m > 0)
            if is_pass: survivors.append(ticker)
            filter_data.append({
                'Ticker': ticker, 'Pass': '✅' if is_pass else '',
                '1M Alpha': alpha_1m, '12M Alpha': alpha_12m, 'Beta': beta
            })
        except Exception as e: continue
            
    df_filter = pd.DataFrame(filter_data)
    st.dataframe(df_filter.style.format({
        '1M Alpha': '{:.2%}', '12M Alpha': '{:.2%}', 'Beta': '{:.2f}'
    }).map(lambda x: 'color: green' if x > 0 else 'color: red', subset=['1M Alpha', '12M Alpha']))
    
    if not survivors:
        st.error("⚠️ 沒有標的通過 Alpha 濾網。建議轉為持有備用資產 (VT) 或現金。")
        st.stop()
        
    # --- Scoring & Ranking (Raw Sortino * FIP) ---
    st.subheader("排名：Raw Sortino (3M+12M) X FIP")
    
    metrics_list = []
    selection_lookbacks = [3, 12]

    for ticker in survivors:
        try:
            # 1. 分別計算 3M 與 12M 的 Sortino
            s_3m = calculate_sortino(daily_ret[ticker], 3)
            s_12m = calculate_sortino(daily_ret[ticker], 12)
            
            # 平均 Sortino
            avg_sortino = (s_3m + s_12m) / 2
            
            # 2. 計算 FIP
            fip = calculate_fip(daily_ret[ticker])
            
            # 3. 乘法評分
            score = avg_sortino * fip
            
            metrics_list.append({
                'Ticker': ticker,
                'Total_Score': score,
                'Avg_Sortino': avg_sortino,
                'Sortino_3M': s_3m,
                'Sortino_12M': s_12m,
                'FIP': fip
            })
        except: continue
    
    # 建立 DataFrame 並排序
    rank_df = pd.DataFrame(metrics_list).set_index('Ticker')
    rank_df = rank_df.sort_values(by='Total_Score', ascending=False)
    
    # 選出 Top 2
    top_N = 2
    top_tickers = rank_df.head(top_N).index.tolist()
    
    # 顯示詳細數據表格 (方便驗算)
    st.dataframe(
        rank_df.style.format("{:.4f}")
        .background_gradient(subset=['Total_Score'], cmap='Greens'),
        use_container_width=True
    )
    
    # --- 2.3 資金配置 (Allocation) ---
    st.subheader(f"🏆 最終資金配置 (Top {top_N} 等權重)")
    
    if len(top_tickers) > 0:
        cols = st.columns(len(top_tickers))
        weight = 100 / len(top_tickers)
        for i, ticker in enumerate(top_tickers):
            with cols[i]:
                st.success(f"**{ticker}**")
                st.markdown(f"#### {weight:.1f}%")
                try:
                    name = yf.Ticker(ticker).info.get('longName', '')
                    st.caption(name)
                except: pass
    
    st.divider()
    st.write("🔗 快速連結:")
    if top_tickers:
        c_links = st.columns(len(top_tickers))
        for i, ticker in enumerate(top_tickers):
            with c_links[i]:
                st.link_button(f"{ticker} Analysis", f"https://finance.yahoo.com/quote/{ticker}")

# ==========================================
# PART 2: 歷史回測分析 (Historical Backtest)
# ==========================================
st.markdown("---")
st.header("⏳ 歷史回測分析 (Backtest)")
st.caption("回測設定：不包含 EQLT (維持11檔)。基準為 VT。選股邏輯：Raw Sortino(3+12) * FIP, Top 2。")

if st.button("🚀 開始執行回測 (Run Backtest)"):
    check_tickers = backtest_assets + safe_pool + ['VT']
    valid_starts = prices[check_tickers].apply(lambda x: x.first_valid_index())
    latest_start = valid_starts.max()
    warmup_days = 365 + 30
    required_start = latest_start + timedelta(days=warmup_days)
    
    start_idx = monthly_prices.index.searchsorted(required_start)
    
    if start_idx >= len(monthly_prices):
        st.error(f"數據不足，無法進行回測。")
        st.stop()
        
    st.info(f"回測區間: {monthly_prices.index[start_idx].date()} 至 {monthly_prices.index[-1].date()}")
    
    portfolio_log = []
    dates = monthly_prices.index
    progress_bar = st.progress(0)
    total_steps = len(dates) - 1 - start_idx
    
    bt_assets_map = {t: live_assets_map.get(t, 'VTI') for t in backtest_assets}
    bt_assets_map['DFEVX'] = 'EEM' 

    for i in range(start_idx, len(dates) - 1):
        curr_date = dates[i]
        next_date = dates[i+1]
        step = i - start_idx
        progress_bar.progress(min(step / total_steps, 1.0))
        
        hist_daily = daily_ret.loc[:curr_date]
        hist_monthly = monthly_prices.loc[:curr_date]
        hist_monthly_ret = monthly_ret.loc[:curr_date]
        
        # 1. 避險判斷 (維持不變 [3,6,9,12])
        neg_count = 0
        for t in backtest_assets:
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
            # 2. 進攻選股
            survivors = []
            # Alpha Filter
            for t in backtest_assets:
                bench = bt_assets_map.get(t, 'VTI')
                try:
                    subset = hist_daily[[t, bench]].tail(252).dropna()
                    if len(subset) > 200:
                        cov = np.cov(subset[t], subset[bench])
                        beta = cov[0, 1] / cov[1, 1]
                    else: beta = 1.0
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
                    if a_1m > 0 or a_12m > 0: survivors.append(t)
                except: continue
            
            # Ranking Logic
            if not survivors:
                selected_tickers = ['VT']
            else:
                metrics = []
                sel_lookbacks = [3, 12]
                for t in survivors:
                    try:
                        avg_s = 0
                        for p in sel_lookbacks:
                            avg_s += calculate_sortino(hist_daily[t], p)
                        avg_s /= len(sel_lookbacks)
                        
                        fip_val = calculate_fip(hist_daily[t])
                        score = avg_s * fip_val
                        
                        metrics.append({'ticker': t, 'Score': score})
                    except: continue
                
                if metrics:
                    m_df = pd.DataFrame(metrics).set_index('ticker')
                    selected_tickers = m_df.sort_values('Score', ascending=False).head(2).index.tolist()
                else:
                    selected_tickers = ['VT']
            
        final_ret = monthly_ret.loc[next_date, selected_tickers].mean()
        portfolio_log.append({'Date': next_date, 'Strategy': final_ret})
        
    progress_bar.empty()
    
    # 4. 分析結果
    res_df = pd.DataFrame(portfolio_log).set_index('Date')
    res_df['Equity'] = (1 + res_df['Strategy']).cumprod()
    res_df['DD'] = res_df['Equity'] / res_df['Equity'].cummax() - 1
    
    bench_ret = monthly_ret['VT'].loc[res_df.index]
    bench_equity = (1 + bench_ret).cumprod()
    bench_dd = bench_equity / bench_equity.cummax() - 1
    
    years = len(res_df) / 12
    cagr = (res_df['Equity'].iloc[-1]) ** (1/years) - 1
    mdd = res_df['DD'].min()
    neg_rets = res_df.loc[res_df['Strategy'] < 0, 'Strategy']
    down_std = neg_rets.std() * np.sqrt(12) if len(neg_rets) > 0 else 1e-6
    sortino = (res_df['Strategy'].mean() * 12) / down_std
    sharpe = (res_df['Strategy'].mean() * 12) / (res_df['Strategy'].std() * np.sqrt(12))
    roll5y = res_df['Equity'].rolling(60).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(1/5) - 1).mean()
    
    b_cagr = (bench_equity.iloc[-1]) ** (1/years) - 1
    b_mdd = bench_dd.min()
    b_neg = bench_ret[bench_ret < 0]
    b_down_std = b_neg.std() * np.sqrt(12) if len(b_neg) > 0 else 1e-6
    b_sortino = (bench_ret.mean() * 12) / b_down_std
    b_sharpe = (bench_ret.mean() * 12) / (bench_ret.std() * np.sqrt(12))
    b_roll5y = bench_equity.rolling(60).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(1/5) - 1).mean()
    
    def display_metric_pair(label, val_strat, val_bench, fmt="{:.2%}"):
        st.markdown(f"""
        <div style="margin-bottom: 10px;">
            <p style="font-size: 14px; margin-bottom: 0px; color: #888;">{label}</p>
            <span style="font-size: 24px; font-weight: bold;">{fmt.format(val_strat)}</span>
            <span style="font-size: 14px; color: gray; margin-left: 8px;">(VT: {fmt.format(val_bench)})</span>
        </div>
        """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: display_metric_pair("CAGR", cagr, b_cagr)
    with c2: display_metric_pair("MDD", mdd, b_mdd)
    with c3: display_metric_pair("Sharpe", sharpe, b_sharpe, "{:.2f}")
    with c4: display_metric_pair("Sortino", sortino, b_sortino, "{:.2f}")
    with c5: display_metric_pair("Avg Rolling 5Y", roll5y, b_roll5y)
    
    st.divider()

    # --- Altair Charts ---
    df_chart = pd.DataFrame({
        'Date': res_df.index,
        'Strategy': (res_df['Equity'] - 1), 
        'Benchmark (VT)': (bench_equity - 1)
    }).melt('Date', var_name='Asset', value_name='Return')
    
    chart_equity = alt.Chart(df_chart).mark_line().encode(
        x='Date',
        y=alt.Y('Return', axis=alt.Axis(format='%')),
        color=alt.Color('Asset', scale=alt.Scale(domain=['Strategy', 'Benchmark (VT)'], range=['#FFD700', '#00B4D8'])), 
        tooltip=['Date', 'Asset', alt.Tooltip('Return', format='.2%')]
    ).properties(title='累積報酬率 (Cumulative Return)')
    
    st.altair_chart(chart_equity, use_container_width=True)
    
    df_dd = pd.DataFrame({
        'Date': res_df.index,
        'Strategy': res_df['DD'],
        'Benchmark (VT)': bench_dd
    }).melt('Date', var_name='Asset', value_name='Drawdown')
    
    chart_dd = alt.Chart(df_dd).mark_line().encode(
        x='Date',
        y=alt.Y('Drawdown', axis=alt.Axis(format='%')),
        color=alt.Color('Asset', scale=alt.Scale(domain=['Strategy', 'Benchmark (VT)'], range=['#FFD700', '#00B4D8'])),
        tooltip=['Date', 'Asset', alt.Tooltip('Drawdown', format='.2%')]
    ).properties(title='回撤 (Drawdown)')
    
    st.altair_chart(chart_dd, use_container_width=True)
    
    df_roll = pd.DataFrame({
        'Date': res_df.index,
        'Strategy': res_df['Equity'].rolling(60).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(1/5) - 1),
        'Benchmark (VT)': bench_equity.rolling(60).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(1/5) - 1)
    }).dropna().melt('Date', var_name='Asset', value_name='Rolling CAGR')
    
    chart_roll = alt.Chart(df_roll).mark_line().encode(
        x='Date',
        y=alt.Y('Rolling CAGR', axis=alt.Axis(format='%')),
        color=alt.Color('Asset', scale=alt.Scale(domain=['Strategy', 'Benchmark (VT)'], range=['#FFD700', '#00B4D8'])),
        tooltip=['Date', 'Asset', alt.Tooltip('Rolling CAGR', format='.2%')]
    ).properties(title='滾動 5 年年化報酬 (Rolling 5-Year CAGR)')
    
    st.altair_chart(chart_roll, use_container_width=True)
