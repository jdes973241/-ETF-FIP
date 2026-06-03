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
st.title("🛡️ 多重資產因子動能輪動策略 (Live & Backtest)")
st.markdown("""
**策略邏輯摘要 (Robustness Optimized v6 — Regime 已驗證定稿)：**
*(註：實盤儀表板採用 EQLT 與 DFEV，歷史回測採用 QUAL 與 DFEVX 替代以延長測試區間)*
1.  **市場狀態 (Regime)**：計算 12 檔股票因子的 **13612W 加權動能** (12×1M + 4×3M + 2×6M + 1×12M)。若轉負標的 **>= 75%** (⌈檔數×0.75⌉，12 檔時為 9 檔)，則全面避險。
    *（13612W 已經研究池 25 種動能定義 × 4 閾值檢定：實際閾值鄰域 Sharpe 排名第 1、鄰域平均第 2/25、無候選定義可顯著勝出，故沿用。）*
2.  **避險模式 (Risk-Off)**：**TLT 與 GLD 等權 50/50** 持有（已移除原動能擇時參數）。
3.  **進攻模式 (Risk-On)**：
    * **評分**：**非重疊純報酬動能 NRet_0_6_6_12** — ([0–6]月年化報酬 + [6–12]月年化報酬) / 2，兩段非重疊，**無 FIP**。
    * **配置**：持有 **前 2 名**，等權重。
""")

# 重抓數據按鈕（置於資料下載前，確保任何標的缺失導致判讀異常時仍可一鍵重抓）
col_refresh, _ = st.columns([1, 4])
with col_refresh:
    if st.button("🔄 重新抓取數據（清除快取）", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
st.caption("若發現部分標的數據缺失或判讀異常，請點此清除快取並重新下載。")

# ==========================================
# 核心邏輯函數
# ==========================================
def calculate_daily_beta(asset, bench, daily_df, lookback=252):
    subset = daily_df[[asset, bench]].dropna().tail(lookback)
    if len(subset) < lookback * 0.8: return 1.0
    cov = np.cov(subset[asset], subset[bench])
    return cov[0, 1] / cov[1, 1]

def calculate_return_nonoverlap(daily_series, start_month, end_month):
    """
    [v5 進攻核心] 非重疊區間年化報酬 (NRet)
    切片 [start_month, end_month]（單位：月），用日報酬累積後年化。
    用於 NRet_0_6_6_12 = ([0-6] + [6-12]) / 2
    """
    start_day = int(start_month * 21)
    end_day = int(end_month * 21)
    if end_day <= start_day:
        return -999.0
    full_series = daily_series.dropna()
    if len(full_series) < end_day + 5:
        return -999.0
    if start_day == 0:
        subset = full_series.iloc[-end_day:]
    else:
        subset = full_series.iloc[-end_day:-start_day]
    if len(subset) < (end_day - start_day) * 0.5:
        return -999.0
    cum_ret = (1 + subset).prod() - 1
    years = len(subset) / 252
    if years <= 0 or (1 + cum_ret) <= 0:
        return -0.99
    return (1 + cum_ret) ** (1/years) - 1

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

# 實盤儀表板維持 EQLT 與 DFEV
live_assets_map = {
    'IMOM': 'EFA', 'IVAL': 'EFA', 'IDHQ': 'EFA', 'ISCF': 'EFA', 
    'QMOM': 'VTI', 'QVAL': 'VTI', 'SPHQ': 'VTI', 'FDM': 'VTI',  
    'PIE': 'EEM',  'DFEV': 'EEM', 'EWX': 'EEM', 'EQLT': 'EEM' 
}

# 回測資產採用 QUAL 與 DFEVX 替代
backtest_assets = [
    'IMOM', 'IVAL', 'IDHQ', 'ISCF', 
    'QMOM', 'QVAL', 'SPHQ', 'FDM',  
    'PIE',  'DFEVX', 'EWX', 'QUAL'          
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
# 第一階段：市場狀態判斷 (Regime Filter)
# ==========================================
st.subheader("1️⃣ 第一階段：市場狀態判斷 (Regime Filter)")

regime_stats = []
neg_count = 0 
valid_count = 0
missing_tickers = []   # 資料缺失防呆：記錄 NaN / 無資料標的

for ticker in equity_tickers:
    try:
        p_now = monthly_prices.loc[cutoff_date, ticker]
        # 13612W 加權動能：12×1M + 4×3M + 2×6M + 1×12M
        r_1 = (p_now / monthly_prices.iloc[-2][ticker]) - 1
        r_3 = (p_now / monthly_prices.iloc[-4][ticker]) - 1
        r_6 = (p_now / monthly_prices.iloc[-7][ticker]) - 1
        r_12 = (p_now / monthly_prices.iloc[-13][ticker]) - 1
        ticker_w_mom = 12 * r_1 + 4 * r_3 + 2 * r_6 + r_12

        # 防呆：資料缺失 (NaN) 標的標記 ⚪ 並排除於統計外，
        # 避免 NaN 被誤判為轉負紅燈（造成視覺與 neg_count 不一致的判讀異常）
        if np.isnan(ticker_w_mom):
            regime_stats.append({
                'Ticker': ticker, 'Status': '⚪',
                'W_Mom (13612W)': np.nan,
                '1M': r_1, '3M': r_3, '6M': r_6, '12M': r_12
            })
            missing_tickers.append(ticker)
            continue

        status_icon = "🟢" if ticker_w_mom > 0 else "🔴"
        if ticker_w_mom < 0:
            neg_count += 1
        valid_count += 1

        regime_stats.append({
            'Ticker': ticker,
            'Status': status_icon,
            'W_Mom (13612W)': ticker_w_mom,
            '1M': r_1, '3M': r_3, '6M': r_6, '12M': r_12
        })
    except Exception:
        # 完全無資料：標記 ⚪，排除於統計外
        regime_stats.append({
            'Ticker': ticker, 'Status': '⚪',
            'W_Mom (13612W)': np.nan,
            '1M': np.nan, '3M': np.nan, '6M': np.nan, '12M': np.nan
        })
        missing_tickers.append(ticker)
        continue

# 資料缺失警告（防呆）
if missing_tickers:
    st.warning(
        f"⚠️ 偵測到 {len(missing_tickers)} 檔資料缺失：{', '.join(missing_tickers)}。"
        f"已標記 ⚪ 並排除於 Regime 統計外，避免誤判為轉負。"
        f"建議點上方「🔄 重新抓取數據」補回後再判讀。"
    )

# Thr_75% 動態閾值：⌈有效檔數 × 0.75⌉
THRESHOLD_RATIO = 0.75
THRESHOLD_N = int(np.ceil(valid_count * THRESHOLD_RATIO)) if valid_count > 0 else 1
is_bull_market = neg_count < THRESHOLD_N

col1, col2 = st.columns([1, 2])
col1.metric("轉弱標的數量 (Count < 0)", f"{neg_count} / {valid_count}", delta_color="inverse")
status_text = "🐂 牛市 (進攻模式)" if is_bull_market else "🐻 熊市 (避險模式)"
status_color = "green" if is_bull_market else "red"
col2.markdown(f"### 市場狀態: :{status_color}[{status_text}]")
col2.caption(f"避險觸發條件：轉弱標的數量 >= {THRESHOLD_N} (⌈{valid_count} × 75%⌉)")

with st.expander("查看全市場 12 檔 ETF 動能細節 (13612W 加權)"):
    df_regime = pd.DataFrame(regime_stats)
    df_regime.index = range(1, len(df_regime) + 1)   # 編號從 1 開始
    cols = ['Ticker', 'Status', 'W_Mom (13612W)', '1M', '3M', '6M', '12M']
    st.dataframe(
        df_regime[cols].style.format(
            "{:.2%}", subset=['W_Mom (13612W)', '1M', '3M', '6M', '12M'], na_rep="—"
        )
    )

st.divider()

# ==========================================
# 第二階段：策略分支
# ==========================================

if not is_bull_market:
    # 🐻 避險模式
    st.header("2️⃣ 第二階段 (A)：避險模式 (Risk-Off)")
    st.info("全市場動能轉弱，啟動避險：**TLT 與 GLD 等權 50/50** 持有（已移除動能擇時，等權經驗證最穩健）。")
    
    hedge_stats = []
    for asset in safe_pool:
        try:
            # 動能僅供參考顯示，配置固定等權
            p_now = monthly_prices.loc[cutoff_date, asset]
            p_3m = monthly_prices.iloc[-4][asset]
            p_12m = monthly_prices.iloc[-13][asset]
            r_3m = (p_now / p_3m) - 1
            r_12m = (p_now / p_12m) - 1
            hedge_stats.append({'Asset': asset, '3M Return': r_3m, '12M Return': r_12m, 'Weight': '50%'})
        except:
            st.warning(f"缺少 {asset} 數據")

    df_hedge = pd.DataFrame(hedge_stats)
    st.dataframe(df_hedge.style.format({'3M Return': '{:.2%}', '12M Return': '{:.2%}'}), use_container_width=False)
    st.success(f"🛡️ 本月建議持倉: **TLT 50% + GLD 50%** (等權)")

else:
    # 🐂 進攻模式
    st.header("2️⃣ 第二階段 (B)：進攻模式 (Risk-On)")
    
    survivors = equity_tickers  
        
    # --- Scoring & Ranking ---
    st.subheader("排名：非重疊純報酬動能 NRet_0_6_6_12")
    st.caption("NRet = ([0-6]月年化報酬 + [6-12]月年化報酬) / 2，兩段非重疊，無 FIP")
    
    metrics_list = []

    for ticker in survivors:
        try:
            price = monthly_prices.loc[cutoff_date, ticker]
            
            p_6m = monthly_prices.iloc[-7][ticker]
            raw_ret_6m = (price / p_6m) - 1
            p_12m = monthly_prices.iloc[-13][ticker]
            raw_ret_12m = (price / p_12m) - 1
            
            # NRet_0_6_6_12：兩段非重疊年化報酬平均
            nret_0_6 = calculate_return_nonoverlap(daily_ret[ticker], 0, 6)
            nret_6_12 = calculate_return_nonoverlap(daily_ret[ticker], 6, 12)
            if nret_0_6 == -999.0 or nret_6_12 == -999.0:
                continue
            score = (nret_0_6 + nret_6_12) / 2
            
            metrics_list.append({
                'Ticker': ticker,
                'Total_Score': score,
                'Price': price,
                'Raw_Ret_6M': raw_ret_6m,
                'Raw_Ret_12M': raw_ret_12m,
                'NRet_[0-6] (Ann)': nret_0_6,
                'NRet_[6-12] (Ann)': nret_6_12,
            })
        except: continue
    
    rank_df = pd.DataFrame(metrics_list).set_index('Ticker')
    rank_df = rank_df.sort_values(by='Total_Score', ascending=False)
    
    top_N = 2
    top_tickers = rank_df.head(top_N).index.tolist()
    
    st.dataframe(
        rank_df.style.format({
            'Total_Score': '{:.2%}',
            'Price': '{:.2f}',
            'Raw_Ret_6M': '{:.2%}',
            'Raw_Ret_12M': '{:.2%}',
            'NRet_[0-6] (Ann)': '{:.2%}',
            'NRet_[6-12] (Ann)': '{:.2%}',
        }).background_gradient(subset=['Total_Score'], cmap='Greens'),
        use_container_width=True
    )
    
    # --- 2.3 資金配置 ---
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
# PART 2: 歷史回測分析 (還原原始圖表邏輯)
# ==========================================
st.markdown("---")
st.header("⏳ 歷史回測分析 (Backtest)")
st.caption("回測設定 (v6)：12 檔股票因子，使用 DFEVX (長歷史)，替換 EQLT 為 QUAL。基準為 VT。Regime=13612W+Thr75%；避險=TLT/GLD 等權；進攻=NRet_0_6_6_12。已計入單次換倉 0.15% 摩擦成本。")

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
    
    prev_tickers = []
    transaction_cost_rate = 0.0015

    for i in range(start_idx, len(dates) - 1):
        curr_date = dates[i]
        next_date = dates[i+1]
        step = i - start_idx
        progress_bar.progress(min(step / total_steps, 1.0))
        
        hist_daily = daily_ret.loc[:curr_date]
        hist_monthly = monthly_prices.loc[:curr_date]
        
        # 1. 避險判斷 (13612W 加權 + Thr_75%)
        neg_count = 0
        valid_count_bt = 0
        for t in backtest_assets:
            try:
                p_now = hist_monthly.iloc[-1][t]
                r_1 = (p_now / hist_monthly.iloc[-2][t]) - 1
                r_3 = (p_now / hist_monthly.iloc[-4][t]) - 1
                r_6 = (p_now / hist_monthly.iloc[-7][t]) - 1
                r_12 = (p_now / hist_monthly.iloc[-13][t]) - 1
                w_mom = 12 * r_1 + 4 * r_3 + 2 * r_6 + r_12
                valid_count_bt += 1
                if w_mom < 0: neg_count += 1
            except: continue
            
        threshold_n_bt = int(np.ceil(valid_count_bt * 0.75))
        is_bear = neg_count >= threshold_n_bt
        selected_tickers = []
        
        if is_bear:
            # Risk-Off: TLT + GLD 等權 50/50
            selected_tickers = ['TLT', 'GLD']
        else:
            # 2. 進攻選股 (NRet_0_6_6_12 純報酬)
            survivors = backtest_assets
            metrics = []
            for t in survivors:
                try:
                    nret_0_6 = calculate_return_nonoverlap(hist_daily[t], 0, 6)
                    nret_6_12 = calculate_return_nonoverlap(hist_daily[t], 6, 12)
                    if nret_0_6 == -999.0 or nret_6_12 == -999.0:
                        continue
                    score = (nret_0_6 + nret_6_12) / 2
                    metrics.append({'ticker': t, 'Score': score})
                except: continue
            
            if metrics:
                m_df = pd.DataFrame(metrics).set_index('ticker')
                selected_tickers = m_df.sort_values('Score', ascending=False).head(2).index.tolist()
            else:
                selected_tickers = ['VT']
        
        raw_final_ret = monthly_ret.loc[next_date, selected_tickers].mean()
        
        turnover_penalty = 0.0
        if prev_tickers:
            new_assets = set(selected_tickers) - set(prev_tickers)
            if new_assets:
                turnover_penalty = (len(new_assets) / len(selected_tickers)) * transaction_cost_rate
                
        final_ret = raw_final_ret - turnover_penalty
        portfolio_log.append({'Date': next_date, 'Strategy': final_ret})
        prev_tickers = selected_tickers
        
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
    
    strat_downside = np.where(res_df['Strategy'] < 0, res_df['Strategy'], 0)
    down_std = np.sqrt(np.mean(strat_downside**2)) * np.sqrt(12) if np.any(strat_downside < 0) else 1e-6
    sortino = (res_df['Strategy'].mean() * 12) / down_std
    sharpe = (res_df['Strategy'].mean() * 12) / (res_df['Strategy'].std() * np.sqrt(12))
    roll5y = res_df['Equity'].rolling(60).apply(lambda x: (x.iloc[-1]/x.iloc[0])**(1/5) - 1).mean()
    
    b_cagr = (bench_equity.iloc[-1]) ** (1/years) - 1
    b_mdd = bench_dd.min()
    bench_downside = np.where(bench_ret < 0, bench_ret, 0)
    b_down_std = np.sqrt(np.mean(bench_downside**2)) * np.sqrt(12) if np.any(bench_downside < 0) else 1e-6
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
