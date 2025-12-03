import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import zscore
from datetime import datetime, timedelta

# ==========================================
# 頁面設定
# ==========================================
st.set_page_config(page_title="多重資產動能策略", layout="wide")
st.title("🛡️ 多重資產因子動能輪動策略 (Final Optimized)")
st.markdown("""
**策略邏輯摘要：**
1.  **市場狀態 (Regime)**：計算 12 檔股票因子的平均動能。若 < 0 則全面避險；若 > 0 則進攻。
2.  **避險模式 (Risk-Off)**：比較 **TLT** 與 **GLD** 的 12 個月報酬，全倉持有強者。
3.  **進攻模式 (Risk-On)**：
    * **濾網**：Alpha (1M 或 12M > 0)。
    * **排名**：動能 (3+6+9+12M) 75% + 品質 (FIP) 25%。
    * **配置**：持有前 3 名，等權重。
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
    if len(subset) < lookback * 0.5: return np.nan
    return (subset > 0).sum() / len(subset)

@st.cache_data(ttl=3600)
def load_and_process_data():
    # 1. 定義資產池
    assets_map = {
        # 國際已開發
        'IMOM': 'EFA', 'IVAL': 'EFA', 'IDHQ': 'EFA', 'GWX': 'EFA',
        # 美國
        'QMOM': 'VTI', 'QVAL': 'VTI', 'SPHQ': 'VTI', 'SCHA': 'VTI',
        # 新興市場
        'PIE': 'EEM',  'DFEV': 'EEM', 'DEHP': 'EEM', 'EEMS': 'EEM'
    }
    
    equity_tickers = list(assets_map.keys())
    benchmarks = list(set(assets_map.values()))
    
    # 避險池 (Hedge Assets)
    safe_pool = ['TLT', 'GLD']
    
    all_symbols = list(set(equity_tickers + benchmarks + safe_pool))

    # 2. 下載數據
    # 抓取 3 年數據以確保有足夠的移動平均和 Beta 計算緩衝
    start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    with st.spinner('正在下載最新市場數據...'):
        raw_data = yf.download(all_symbols, start=start_date, end=end_date, progress=False, auto_adjust=False)
    
    if 'Adj Close' in raw_data.columns:
        prices = raw_data['Adj Close']
    elif 'Close' in raw_data.columns:
        prices = raw_data['Close']
    else:
        return None, None, None, None, None, None, "❌ 嚴重錯誤: 無法下載價格資料"

    prices = prices.astype(float).ffill() # 填補空值
    
    if prices.empty:
        return None, None, None, None, None, None, "❌ 錯誤: 下載的數據為空。"

    # 檢查數據新鮮度
    last_dt = prices.index[-1]
    today = datetime.now()
    if (today - last_dt).days > 7:
        st.warning(f"⚠️ 注意：最新數據日期為 {last_dt.strftime('%Y-%m-%d')}，可能非即時數據。")

    # 3. 智能月結算日期處理
    monthly_prices = prices.resample('ME').last()
    
    current_date = datetime.now().date()
    last_idx = monthly_prices.index[-1]
    
    # 檢查本月是否已結束
    next_day = current_date + timedelta(days=1)
    if last_idx.month == current_date.month and last_idx.year == current_date.year:
         if next_day.month == current_date.month: 
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
    
    return monthly_ret, daily_ret, monthly_prices, assets_map, safe_pool, cutoff_date, msg

# ==========================================
# 執行計算與顯示
# ==========================================
data_pack = load_and_process_data()

if data_pack[0] is None:
    st.error(data_pack[6])
    st.stop()

monthly_ret, daily_ret, monthly_prices, assets_map, safe_pool, cutoff_date, status_msg = data_pack
equity_tickers = list(assets_map.keys())

# --- 側邊欄：市場快照 ---
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

periods = [3, 6, 9, 12]
regime_stats = []
mom_sum = 0
valid_count = 0

for ticker in equity_tickers:
    try:
        p_now = monthly_prices.loc[cutoff_date, ticker]
        ticker_avg_mom = 0
        p_vals = []
        
        for p in periods:
            p_prev = monthly_prices.iloc[-1-p][ticker] 
            r = (p_now / p_prev) - 1
            ticker_avg_mom += r
            p_vals.append(r)
            
        ticker_avg_mom /= 4
        regime_stats.append({
            'Ticker': ticker,
            'Avg_Mom': ticker_avg_mom,
            '3M': p_vals[0], '6M': p_vals[1], '9M': p_vals[2], '12M': p_vals[3]
        })
        
        if not np.isnan(ticker_avg_mom):
            mom_sum += ticker_avg_mom
            valid_count += 1
    except Exception as e:
        continue

universe_mom = mom_sum / valid_count if valid_count > 0 else 0
is_bull_market = universe_mom > 0

col1, col2 = st.columns([1, 2])
col1.metric("全市場平均動能", f"{universe_mom:.2%}", delta_color="normal")
status_text = "🐂 牛市 (進攻模式)" if is_bull_market else "🐻 熊市 (避險模式)"
status_color = "green" if is_bull_market else "red"
col2.markdown(f"### 市場狀態: :{status_color}[{status_text}]")

with st.expander("查看全市場 12 檔 ETF 動能細節"):
    st.dataframe(pd.DataFrame(regime_stats).style.format("{:.2%}", subset=['Avg_Mom', '3M', '6M', '9M', '12M']))

st.divider()

# ==========================================
# 第二階段：策略分支
# ==========================================

if not is_bull_market:
    # ==========================
    # 🐻 避險模式 (Risk-Off)
    # ==========================
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
    # ==========================
    # 🐂 進攻模式 (Risk-On)
    # ==========================
    st.header("2️⃣ 第二階段 (B)：進攻模式 (Risk-On)")
    
    # --- 2.1 初階濾網 (Alpha Filter) ---
    st.subheader("篩選：Alpha 濾網")
    st.caption("條件：(1M Alpha > 0) OR (12M Alpha > 0)")
    
    survivors = []
    filter_data = []
    
    for ticker in equity_tickers:
        bench = assets_map[ticker]
        try:
            # 計算 Beta (最近 252 日)
            beta = calculate_daily_beta(ticker, bench, daily_ret, lookback=252)
            
            # 1M 數據
            r_asset_1m = monthly_ret.loc[cutoff_date, ticker]
            r_bench_1m = monthly_ret.loc[cutoff_date, bench]
            alpha_1m = r_asset_1m - (beta * r_bench_1m)
            
            # 12M 數據
            p_now = monthly_prices.loc[cutoff_date, ticker]
            p_12m = monthly_prices.iloc[-13][ticker]
            r_asset_12m = (p_now / p_12m) - 1
            
            p_b_now = monthly_prices.loc[cutoff_date, bench]
            p_b_12m = monthly_prices.iloc[-13][bench]
            r_bench_12m = (p_b_now / p_b_12m) - 1
            
            alpha_12m = r_asset_12m - (beta * r_bench_12m)
            
            is_pass = (alpha_1m > 0) or (alpha_12m > 0)
            
            if is_pass:
                survivors.append(ticker)
                
            filter_data.append({
                'Ticker': ticker,
                'Pass': '✅' if is_pass else '',
                '1M Alpha': alpha_1m,
                '12M Alpha': alpha_12m,
                'Beta': beta
            })
        except Exception as e:
            continue
            
    df_filter = pd.DataFrame(filter_data)
    st.dataframe(df_filter.style.format({
        '1M Alpha': '{:.2%}', '12M Alpha': '{:.2%}', 'Beta': '{:.2f}'
    }).map(lambda x: 'color: green' if x > 0 else 'color: red', subset=['1M Alpha', '12M Alpha']))
    
    if not survivors:
        st.error("⚠️ 沒有標的通過 Alpha 濾網。建議轉為持有備用資產 (VT) 或現金。")
        st.stop()
        
    # --- 2.2 總分排名 (Scoring & Ranking) ---
    st.subheader("排名：綜合動能 (75%) + 品質 (25%)")
    
    # 準備計算 Z-Score 的數據集
    metrics_df = pd.DataFrame(index=survivors)
    for ticker in survivors:
        try:
            p_now = monthly_prices.loc[cutoff_date, ticker]
            for p in periods:
                p_prev = monthly_prices.iloc[-1-p][ticker]
                r = (p_now / p_prev) - 1
                metrics_df.loc[ticker, f'R_{p}M'] = r
            
            fip = calculate_fip(daily_ret[ticker])
            metrics_df.loc[ticker, 'FIP'] = fip
        except: continue
        
    # 計算 Z-Score
    z_df = pd.DataFrame(index=survivors)
    mom_z_cols = []
    for p in periods:
        col_name = f'Z_{p}M'
        z_df[col_name] = zscore(metrics_df[f'R_{p}M'], ddof=1, nan_policy='omit')
        mom_z_cols.append(col_name)
    
    z_df['Avg_Mom_Z'] = z_df[mom_z_cols].mean(axis=1)
    z_df['Z_FIP'] = zscore(metrics_df['FIP'], ddof=1, nan_policy='omit')
    
    # 計算分數與貢獻
    z_df['Mom_Contrib (75%)'] = z_df['Avg_Mom_Z'] * 0.75
    z_df['FIP_Contrib (25%)'] = z_df['Z_FIP'] * 0.25
    z_df['Total_Score'] = z_df['Mom_Contrib (75%)'] + z_df['FIP_Contrib (25%)']
    
    # 排序
    z_df = z_df.sort_values(by='Total_Score', ascending=False)
    top_3 = z_df.head(3).index.tolist()
    
    # 將總分合併回原始數據以便顯示
    metrics_df['Total_Score'] = z_df['Total_Score']
    metrics_df = metrics_df.loc[z_df.index]

    # --- 使用 Tabs 切換視圖 ---
    tab_z, tab_raw = st.tabs(["📊 標準化數據 (Z-Score & 貢獻)", "🔢 原始數據 (報酬率 & FIP)"])

    with tab_z:
        st.caption("此表顯示經過標準化 (Z-Score) 後的分數，用於最終排名。")
        z_display_cols = ['Total_Score', 'Mom_Contrib (75%)', 'FIP_Contrib (25%)', 'Avg_Mom_Z', 'Z_FIP']
        
        st.dataframe(
            z_df[z_display_cols],
            use_container_width=True,
            column_config={
                "Total_Score": st.column_config.NumberColumn("總分", format="%.2f"),
                "Mom_Contrib (75%)": st.column_config.NumberColumn("動能貢獻", format="%.2f", help="動能 Z 分數 x 0.75"),
                "FIP_Contrib (25%)": st.column_config.NumberColumn("品質貢獻", format="%.2f", help="FIP Z 分數 x 0.25"),
                "Avg_Mom_Z": st.column_config.NumberColumn("原始動能 Z", format="%.2f"),
                "Z_FIP": st.column_config.NumberColumn("原始 FIP Z", format="%.2f"),
            }
        )

    with tab_raw:
        st.caption("此表顯示未經處理的原始報酬率與 FIP 百分比。")
        
        # 關鍵修正：建立一個副本並乘以 100 以顯示正確百分比
        display_raw_df = metrics_df.copy()
        pct_cols = ['FIP'] + [f'R_{p}M' for p in periods]
        display_raw_df[pct_cols] = display_raw_df[pct_cols] * 100
        
        raw_display_cols = ['Total_Score', 'FIP'] + [f'R_{p}M' for p in periods]
        
        st.dataframe(
            display_raw_df[raw_display_cols],
            use_container_width=True,
            column_config={
                "Total_Score": st.column_config.NumberColumn("總分", format="%.2f"),
                "FIP": st.column_config.NumberColumn("FIP (正報酬天數)", format="%.2f%%"),
                "R_3M": st.column_config.NumberColumn("3M 報酬", format="%.2f%%"),
                "R_6M": st.column_config.NumberColumn("6M 報酬", format="%.2f%%"),
                "R_9M": st.column_config.NumberColumn("9M 報酬", format="%.2f%%"),
                "R_12M": st.column_config.NumberColumn("12M 報酬", format="%.2f%%"),
            }
        )
    
    # --- 2.3 資金配置 (Allocation) ---
    st.subheader("🏆 最終資金配置 (Top 3 等權重)")
    
    cols = st.columns(len(top_3))
    for i, ticker in enumerate(top_3):
        with cols[i]:
            st.success(f"**{ticker}**")
            st.markdown("#### 33.3%")
            try:
                name = yf.Ticker(ticker).info.get('longName', '')
                st.caption(name)
            except: pass

    st.divider()
    st.write("🔗 快速連結:")
    c_links = st.columns(len(top_3))
    for i, ticker in enumerate(top_3):
        with c_links[i]:
            st.link_button(f"{ticker} Analysis", f"https://finance.yahoo.com/quote/{ticker}")
