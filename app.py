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
st.title("🛡️ 多重資產因子動能輪動策略 (Live & Backtest)")
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
    # A. 即時監控用 (Live)
    live_assets_map = {
        'IMOM': 'EFA', 'IVAL': 'EFA', 'IDHQ': 'EFA', 'GWX': 'EFA',
        'QMOM': 'VTI', 'QVAL': 'VTI', 'SPHQ': 'VTI', 'SCHA': 'VTI',
        'PIE': 'EEM',  'DFEV': 'EEM', 'DEHP': 'EEM', 'EEMS': 'EEM'
    }
    
    # B. 回測用 (Backtest): DFEV->DFEVX, 移除 DEHP
    backtest_assets = [
        'IMOM', 'IVAL', 'IDHQ', 'GWX',
        'QMOM', 'QVAL', 'SPHQ', 'SCHA',
        'PIE',  'DFEVX', 'EEMS' # 無 DEHP
    ]
    
    # C. 避險與基準
    safe_pool = ['TLT', 'GLD']
    others = ['VT'] # Benchmark
    
    # 合併所有需要下載的代碼
    all_symbols = list(set(list(live_assets_map.keys()) + list(live_assets_map.values()) + backtest_assets + safe_pool + others))

    # 2. 下載數據 (抓取最長歷史以供回測)
    start_date = '2000-01-01'
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    with st.spinner('正在下載所有歷史數據 (Live & Backtest)...'):
        raw_data = yf.download(all_symbols, start=start_date, end=end_date, progress=False, auto_adjust=False)
    
    if 'Adj Close' in raw_data.columns:
        prices = raw_data['Adj Close']
    elif 'Close' in raw_data.columns:
        prices = raw_data['Close']
    else:
        return None, None, None, None, None, None, None, None, "❌ 嚴重錯誤: 無法下載價格資料"

    prices = prices.astype(float).ffill() # 填補空值
    
    if prices.empty:
        return None, None, None, None, None, None, None, None, "❌ 錯誤: 下載的數據為空。"

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
    
    return monthly_ret, daily_ret, monthly_prices, live_assets_map, backtest_assets, safe_pool, cutoff_date, msg

# ==========================================
# 執行計算與顯示
# ==========================================
data_pack = load_and_process_data()

if data_pack[0] is None:
    st.error(data_pack[7])
    st.stop()

monthly_ret, daily_ret, monthly_prices, live_assets_map, backtest_tickers, safe_pool, cutoff_date, status_msg = data_pack
equity_tickers = list(live_assets_map.keys())

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
        
    # --- Scoring & Ranking ---
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
    
    # 合併原始數據
    metrics_df['Total_Score'] = z_df['Total_Score']
    metrics_df = metrics_df.loc[z_df.index]

    # Tabs 切換
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
        # 手動乘 100
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

# ==========================================
# PART 2: 歷史回測分析 (Historical Backtest)
# ==========================================
st.markdown("---")
st.header("⏳ 歷史回測分析 (Backtest)")
st.caption("回測設定：使用 DFEVX (長歷史版本)、無 DEHP。基準為 VT。")

if st.button("🚀 開始回測 (Run Backtest)"):
    # 1. 確定回測起始點 (需所有回測標的都有數據)
    # 我們需要預留 12個月 + 1個月
    check_tickers = backtest_tickers + safe_pool + ['VT']
    valid_starts = prices[check_tickers].apply(lambda x: x.first_valid_index())
    latest_start = valid_starts.max()
    warmup_days = 365 + 30
    required_start = latest_start + timedelta(days=warmup_days)
    
    start_idx = monthly_prices.index.searchsorted(required_start)
    
    if start_idx >= len(monthly_prices):
        st.error(f"數據不足，無法進行回測。最晚數據起始日: {latest_start.date()}")
        st.stop()
        
    st.info(f"回測區間: {monthly_prices.index[start_idx].date()} 至 {monthly_prices.index[-1].date()}")
    
    portfolio_log = []
    dates = monthly_prices.index
    progress_bar = st.progress(0)
    total_steps = len(dates) - 1 - start_idx
    
    # 為 Alpha Filter 建立一個 backtest 專用的 assets map
    # 這裡簡單處理：若無對應，預設 VTI
    bt_assets_map = {t: live_assets_map.get(t, 'VTI') for t in backtest_tickers}
    # 修正 DFEVX 對應
    bt_assets_map['DFEVX'] = 'EEM' 

    for i in range(start_idx, len(dates) - 1):
        curr_date = dates[i]
        next_date = dates[i+1]
        step = i - start_idx
        progress_bar.progress(min(step / total_steps, 1.0))
        
        hist_daily = daily_ret.loc[:curr_date]
        hist_monthly = monthly_prices.loc[:curr_date]
        hist_monthly_ret = monthly_ret.loc[:curr_date]
        
        # A. 判斷市場狀態 (Count >= 6 on Backtest Tickers)
        # 注意：回測時只使用回測池中的 11 檔來判斷
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
            # 避險: TLT vs GLD (12M)
            best_safe = 'TLT'
            best_ret = -999
            for asset in ['TLT', 'GLD']:
                try:
                    p_now = hist_monthly.iloc[-1][asset]
                    p_prev = hist_monthly.iloc[-1-12][asset]
                    r = (p_now / p_prev) - 1
                    if r > best_ret:
                        best_ret = r
                        best_safe = asset
                except: pass
            selected_tickers = [best_safe]
            
        else:
            # 進攻: Top 3
            survivors = []
            for t in backtest_tickers:
                bench = bt_assets_map.get(t, 'VTI')
                try:
                    # Beta
                    subset = hist_daily[[t, bench]].tail(252).dropna()
                    if len(subset) > 200:
                        cov = np.cov(subset[t], subset[bench])
                        beta = cov[0, 1] / cov[1, 1]
                    else: beta = 1.0
                    
                    # Alpha Check
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
    
    # Stats
    total_ret = res_df['Equity'].iloc[-1] - 1
    years = len(res_df) / 12
    cagr = (res_df['Equity'].iloc[-1]) ** (1/years) - 1
    mdd = res_df['DD'].min()
    
    neg_rets = res_df.loc[res_df['Strategy'] < 0, 'Strategy']
    down_std = neg_rets.std() * np.sqrt(12) if len(neg_rets) > 0 else 1e-6
    sortino = (res_df['Strategy'].mean() * 12) / down_std
    
    # Display Stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CAGR (年化)", f"{cagr:.2%}")
    c2.metric("MDD (最大回撤)", f"{mdd:.2%}")
    c3.metric("Sortino Ratio", f"{sortino:.2f}")
    c4.metric("總報酬率", f"{total_ret:.2%}")
    
    # Charts
    st.subheader("📈 權益曲線 (Strategy vs VT)")
    chart_data = pd.DataFrame({
        'Strategy': res_df['Equity'],
        'Benchmark (VT)': bench_equity
    })
    st.line_chart(chart_data)
    
    st.subheader("📉 回撤圖 (Drawdown)")
    st.area_chart(res_df['DD'], color='#ff4b4b')
