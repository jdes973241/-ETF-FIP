import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
import pytz

# ==========================================
# 頁面設定
# ==========================================
st.set_page_config(page_title="多重資產動能策略", layout="wide")
st.title("🛡️ 因子動能輪動策略 (Arithmetic Sortino $\\times$ FIP)")
st.markdown("""
**策略邏輯摘要 (經數據驗證之最佳化版本)：**
1.  **市場狀態 (Regime)**：計算 11 檔股票因子的平均動能 (3,6,9,12M)。若 **>= 6 檔** 動能轉負，則全面避險。
2.  **避險模式 (Risk-Off)**：比較 **TLT** 與 **GLD** 的 12 個月報酬，全倉持有強者。
3.  **進攻模式 (Risk-On)**：
    * **濾網**：Alpha 防禦 (1M 或 12M Alpha > 0)。
    * **評分**：**學術版算術平均 Sortino (3M+12M) $\\times$ 原版 FIP**。
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
    """計算原版 FIP: 過去 lookback 天數中，正報酬天數的佔比"""
    subset = daily_series.tail(lookback).dropna()
    if len(subset) < lookback * 0.5: return 0.0
    return (subset > 0).sum() / len(subset)

def calculate_sortino_arithmetic(daily_series, lookback_months, target_return_annual=0.0):
    """
    [學術嚴謹版] 計算 Sortino Ratio (LPM Method + Arithmetic Mean)
    """
    days = int(lookback_months * 21)
    subset = daily_series.tail(days).dropna()
    
    if len(subset) < days * 0.5: 
        return -999.0, 0.0, 0.0
    
    # 尺度轉換：年化 Target 轉換為日頻率
    daily_target = (1 + target_return_annual) ** (1/252) - 1
    
    # 分子：算術平均年化 (Academic Standard)
    avg_ret_daily = subset.mean()
    ann_ret_arithmetic = avg_ret_daily * 252 
    
    # 分母：下行偏差 (LPM Method)
    excess_return = subset - daily_target
    downside_return = np.where(excess_return < 0, excess_return, 0)
    
    downside_variance = np.mean(downside_return**2)
    downside_std = np.sqrt(downside_variance) * np.sqrt(252)
    
    if downside_std == 0:
        return np.inf, ann_ret_arithmetic, 0.0
        
    sortino = (ann_ret_arithmetic - target_return_annual) / downside_std
    return sortino, ann_ret_arithmetic, downside_std

@st.cache_data(ttl=3600)
def fetch_market_data(all_symbols, start_date, end_date):
    """純 I/O 函數，負責數據下載 (Threads=False 防禦雲端阻擋)"""
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
    if prices.empty: return None, None, None, None, None, "❌ 錯誤: 下載的數據為空。"

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
live_assets_map = {
    'IMOM': 'EFA', 'IVAL': 'EFA', 'IDHQ': 'EFA', 'ISCF': 'EFA', 
    'QMOM': 'VTI', 'QVAL': 'VTI', 'SPHQ': 'VTI', 'FDM': 'VTI',  
    'PIE': 'EEM',  'DFEV': 'EEM', 'EWX': 'EEM', 'EQLT': 'EEM' 
}

# 回測標的排除 EQLT
backtest_assets = ['IMOM', 'IVAL', 'IDHQ', 'ISCF', 'QMOM', 'QVAL', 'SPHQ', 'FDM', 'PIE', 'DFEVX', 'EWX']
safe_pool = ['TLT', 'GLD']
others = ['VT'] 

all_symbols = list(set(list(live_assets_map.keys()) + list(live_assets_map.values()) + backtest_assets + safe_pool + others))

tz = pytz.timezone('Asia/Taipei')
now_tw = datetime.now(tz)
start_date_str = '2000-01-01'
end_date_str = (now_tw + timedelta(days=1)).strftime('%Y-%m-%d')

with st.spinner('正在下載所有歷史數據...'):
    raw_prices, error_msg = fetch_market_data(all_symbols, start_date_str, end_date_str)

if raw_prices is None:
    st.error(error_msg)
    st.stop()

prices, monthly_ret, daily_ret, monthly_prices, cutoff_date, status_msg = process_data_logic(
    raw_prices, live_assets_map, backtest_assets, safe_pool, now_tw
)

equity_tickers = list(live_assets_map.keys())

with st.sidebar:
    st.header("📈 市場快照")
    st.info(f"分析基準日: {cutoff_date.strftime('%Y-%m-%d')}")
    st.caption(status_msg)
    st.divider()

# ==========================================
# 第一階段：市場狀態判斷
# ==========================================
st.subheader("1️⃣ 第一階段：市場狀態判斷 (Regime Filter)")

neg_count = 0 
valid_count = 0
regime_stats = []

for ticker in equity_tickers:
    try:
        p_now = monthly_prices.loc[cutoff_date, ticker]
        avg_mom = sum([(p_now / monthly_prices.iloc[-1-p][ticker]) - 1 for p in [3, 6, 9, 12]]) / 4
        if avg_mom < 0: neg_count += 1
        regime_stats.append({'Ticker': ticker, 'Status': "🟢" if avg_mom > 0 else "🔴", 'Avg_Mom': avg_mom})
        valid_count += 1
    except: continue

THRESHOLD_N = 6
is_bull_market = neg_count < THRESHOLD_N

col1, col2 = st.columns([1, 2])
col1.metric("轉弱標的數量 (Count < 0)", f"{neg_count} / {valid_count}", delta_color="inverse")
col2.markdown(f"### 市場狀態: :{'green' if is_bull_market else 'red'}[{'🐂 牛市 (進攻)' if is_bull_market else '🐻 熊市 (避險)'}]")
st.divider()

# ==========================================
# 第二階段：策略分支
# ==========================================
if not is_bull_market:
    st.header("2️⃣ 第二階段 (A)：避險模式 (Risk-Off)")
    st.info("全市場動能 < 0，啟動避險。比較 TLT 與 GLD 的 12 個月報酬率。")
    best_hedge, best_ret = None, -999
    
    for asset in safe_pool:
        try:
            r = (monthly_prices.loc[cutoff_date, asset] / monthly_prices.iloc[-13][asset]) - 1
            if r > best_ret: best_ret, best_hedge = r, asset
        except: pass
    st.success(f"🛡️ 本月建議持倉: **{best_hedge}** (100% 權重)")

else:
    st.header("2️⃣ 第二階段 (B)：進攻模式 (Risk-On)")
    
    # --- Alpha Filter ---
    survivors = []
    for ticker in equity_tickers:
        bench = live_assets_map[ticker]
        try:
            beta = calculate_daily_beta(ticker, bench, daily_ret, 252)
            a_1m = monthly_ret.loc[cutoff_date, ticker] - (beta * monthly_ret.loc[cutoff_date, bench])
            a_12m = ((monthly_prices.loc[cutoff_date, ticker] / monthly_prices.iloc[-13][ticker]) - 1) - \
                    (beta * ((monthly_prices.loc[cutoff_date, bench] / monthly_prices.iloc[-13][bench]) - 1))
            if a_1m > 0 or a_12m > 0: survivors.append(ticker)
        except: continue
            
    if not survivors:
        st.error("⚠️ 無標的通過 Alpha 濾網。轉為持有 VT。")
        st.stop()
        
    # --- Scoring & Ranking ---
    st.subheader("排名：Arithmetic Sortino X 原版 FIP")
    metrics_list = []

    for ticker in survivors:
        try:
            s_3m, arith_3m, down_3m = calculate_sortino_arithmetic(daily_ret[ticker], 3)
            s_12m, arith_12m, down_12m = calculate_sortino_arithmetic(daily_ret[ticker], 12)
            avg_sortino = (s_3m + s_12m) / 2
            fip = calculate_fip(daily_ret[ticker])
            score = avg_sortino * fip
            
            metrics_list.append({
                'Ticker': ticker, 'Total_Score': score,
                'Avg_Sortino': avg_sortino, 'FIP': fip,
                'Arith_Ret_3M': arith_3m, 'Down_Std_3M': down_3m,
                'Arith_Ret_12M': arith_12m, 'Down_Std_12M': down_12m
            })
        except: continue
    
    rank_df = pd.DataFrame(metrics_list).set_index('Ticker').sort_values('Total_Score', ascending=False)
    top_tickers = rank_df.head(2).index.tolist()
    
    st.dataframe(rank_df.style.format({
        'Total_Score': '{:.4f}', 'Avg_Sortino': '{:.2f}', 'FIP': '{:.2f}',
        'Arith_Ret_3M': '{:.2%}', 'Down_Std_3M': '{:.2%}',
        'Arith_Ret_12M': '{:.2%}', 'Down_Std_12M': '{:.2%}'
    }).background_gradient(subset=['Total_Score'], cmap='Greens'), use_container_width=True)
    
    st.subheader(f"🏆 最終資金配置 (Top 2 等權重)")
    cols = st.columns(len(top_tickers))
    for i, t in enumerate(top_tickers):
        with cols[i]: st.success(f"**{t}** (50%)")
    st.divider()

# ==========================================
# PART 2: 歷史回測分析
# ==========================================
st.header("⏳ 歷史回測分析 (Backtest)")
st.caption("回測設定：使用 DFEVX (長歷史)，基準為 VT。已計入單次換倉 0.15% 摩擦成本。")

if st.button("🚀 開始執行回測"):
    check_tickers = backtest_assets + safe_pool + ['VT']
    valid_starts = prices[check_tickers].apply(lambda x: x.first_valid_index())
    start_idx = monthly_prices.index.searchsorted(valid_starts.max() + timedelta(days=395))
    
    if start_idx >= len(monthly_prices):
        st.error("數據不足，無法進行回測。")
        st.stop()
        
    portfolio_log = []
    dates = monthly_prices.index
    progress_bar = st.progress(0)
    total_steps = len(dates) - 1 - start_idx
    
    bt_assets_map = {t: live_assets_map.get(t, 'VTI') for t in backtest_assets}
    bt_assets_map['DFEVX'] = 'EEM' 
    prev_tickers = []
    tc_rate = 0.0015 

    for i in range(start_idx, len(dates) - 1):
        curr_date, next_date = dates[i], dates[i+1]
        progress_bar.progress(min((i - start_idx) / total_steps, 1.0))
        
        h_daily, h_monthly, h_m_ret = daily_ret.loc[:curr_date], monthly_prices.loc[:curr_date], monthly_ret.loc[:curr_date]
        
        # 避險
        neg_count = sum([1 for t in backtest_assets if sum([(h_monthly.iloc[-1][t]/h_monthly.iloc[-1-p][t])-1 for p in [3,6,9,12]])/4 < 0])
        if neg_count >= 6:
            best_hedge = max(safe_pool, key=lambda a: (h_monthly.iloc[-1][a]/h_monthly.iloc[-13][a])-1)
            sel_tickers = [best_hedge]
        else:
            # 進攻
            survs = []
            for t in backtest_assets:
                bench = bt_assets_map.get(t, 'VTI')
                sub = h_daily[[t, bench]].tail(252).dropna()
                beta = np.cov(sub[t], sub[bench])[0, 1] / np.cov(sub[t], sub[bench])[1, 1] if len(sub) > 200 else 1.0
                if (h_m_ret.iloc[-1][t] - beta * h_m_ret.iloc[-1][bench] > 0) or \
                   (((h_monthly.iloc[-1][t]/h_monthly.iloc[-13][t])-1) - beta*((h_monthly.iloc[-1][bench]/h_monthly.iloc[-13][bench])-1) > 0):
                    survs.append(t)
            
            if not survs: sel_tickers = ['VT']
            else:
                mets = []
                for t in survs:
                    try:
                        s_score = (calculate_sortino_arithmetic(h_daily[t], 3)[0] + calculate_sortino_arithmetic(h_daily[t], 12)[0]) / 2
                        mets.append({'t': t, 'S': s_score * calculate_fip(h_daily[t])})
                    except: continue
                sel_tickers = pd.DataFrame(mets).set_index('t').sort_values('S', ascending=False).head(2).index.tolist() if mets else ['VT']
        
        # 結算報酬與摩擦成本
        r_raw = monthly_ret.loc[next_date, sel_tickers].mean()
        penalty = (len(set(sel_tickers) - set(prev_tickers)) / len(sel_tickers)) * tc_rate if prev_tickers else 0.0
        portfolio_log.append({'Date': next_date, 'Strategy': r_raw - penalty})
        prev_tickers = sel_tickers
        
    progress_bar.empty()
    
    # 分析結果
    res = pd.DataFrame(portfolio_log).set_index('Date')
    res['Eq'] = (1 + res['Strategy']).cumprod()
    res['DD'] = res['Eq'] / res['Eq'].cummax() - 1
    
    b_ret = monthly_ret['VT'].loc[res.index]
    b_eq = (1 + b_ret).cumprod()
    b_dd = b_eq / b_eq.cummax() - 1
    
    yrs = len(res) / 12
    cagr, b_cagr = res['Eq'].iloc[-1]**(1/yrs)-1, b_eq.iloc[-1]**(1/yrs)-1
    mdd, b_mdd = res['DD'].min(), b_dd.min()
    
    s_down = np.where(res['Strategy'] < 0, res['Strategy'], 0)
    b_down = np.where(b_ret < 0, b_ret, 0)
    sortino = (res['Strategy'].mean() * 12) / (np.sqrt(np.mean(s_down**2)) * np.sqrt(12) if np.any(s_down<0) else 1e-6)
    b_sortino = (b_ret.mean() * 12) / (np.sqrt(np.mean(b_down**2)) * np.sqrt(12) if np.any(b_down<0) else 1e-6)
    
    roll5, b_roll5 = "N/A", "N/A"
    if len(res) >= 60:
        roll5 = f"{res['Eq'].rolling(60).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(1/5)-1).mean():.2%}"
        b_roll5 = f"{b_eq.rolling(60).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(1/5)-1).mean():.2%}"

    def metric(l, v_s, v_b, fmt="{:.2%}"):
        st.markdown(f"<div><p style='font-size:14px;color:#888;margin:0;'>{l}</p><span style='font-size:24px;font-weight:bold;'>{fmt.format(v_s) if isinstance(v_s, float) else v_s}</span><span style='font-size:14px;color:gray;margin-left:8px;'>(VT: {fmt.format(v_b) if isinstance(v_b, float) else v_b})</span></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric("CAGR", cagr, b_cagr)
    with c2: metric("Max DD", mdd, b_mdd)
    with c3: metric("Sortino", sortino, b_sortino, "{:.2f}")
    with c4: metric("Avg 5Y Rolling", roll5, b_roll5, "{}")
    st.divider()

    df_chart = pd.DataFrame({'Date': res.index, 'Strategy': res['Eq']-1, 'VT': b_eq-1}).melt('Date', var_name='Asset', value_name='Ret')
    st.altair_chart(alt.Chart(df_chart).mark_line().encode(x='Date', y=alt.Y('Ret', axis=alt.Axis(format='%')), color=alt.Color('Asset', scale=alt.Scale(range=['#FFD700', '#00B4D8'])), tooltip=['Date', 'Asset', alt.Tooltip('Ret', format='.2%')]).properties(title='累積報酬率'), use_container_width=True)
