import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import zscore
from datetime import datetime, timedelta

# ==========================================
# 頁面設定
# ==========================================
st.set_page_config(page_title="因子動能策略監控", layout="wide")
st.title("📊 因子動能與 FIP 策略儀表板")

# ==========================================
# 核心邏輯
# ==========================================
def calculate_daily_beta(asset, bench, daily_df, lookback=252):
    subset = daily_df[[asset, bench]].dropna().tail(lookback)
    if len(subset) < lookback * 0.8: return 1.0
    cov = np.cov(subset[asset], subset[bench])
    return cov[0, 1] / cov[1, 1]

@st.cache_data(ttl=3600)
def load_and_process_data():
    assets_map = {
        'IMOM': 'EFA', 'IVAL': 'EFA', 'IDHQ': 'EFA', 'GWX': 'EFA',
        'QMOM': 'VTI', 'QVAL': 'VTI', 'SPHQ': 'VTI', 'SCHA': 'VTI',
        'PIE': 'EEM',  'DFEV': 'EEM', 'DEHP': 'EEM', 'EEMS': 'EEM'
    }
    tickers = list(assets_map.keys())
    benchmarks = list(set(assets_map.values()))
    all_symbols = tickers + benchmarks

    # 下載較長區間以確保計算無誤
    start_date = (datetime.now() - timedelta(days=365*3 + 30)).strftime('%Y-%m-%d')
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # 下載數據
    raw_data = yf.download(all_symbols, start=start_date, end=end_date, progress=False, auto_adjust=False)
    
    if 'Adj Close' in raw_data.columns:
        daily_adj_close = raw_data['Adj Close']
    elif 'Close' in raw_data.columns:
        daily_adj_close = raw_data['Close']
    else:
        return None, None, None, None, None, None, "❌ 嚴重錯誤: 無法下載價格資料"

    daily_adj_close = daily_adj_close.astype(float)
    
    if daily_adj_close.empty:
        return None, None, None, None, None, None, "❌ 錯誤: 下載的數據為空。"

    last_dt = daily_adj_close.index[-1]
    today = datetime.now()
    days_diff = (today - last_dt).days
    
    if days_diff > 5:
        return None, None, None, None, None, None, f"❌ 數據過舊警報！最新資料日期為 {last_dt.strftime('%Y-%m-%d')}，已超過 {days_diff} 天未更新。"

    monthly_prices = daily_adj_close.resample('ME').last()

    # --- 智能日期切割 ---
    last_idx = monthly_prices.index[-1]
    current_date = datetime.now().date()
    next_month = last_idx.replace(day=28) + timedelta(days=4)
    last_day_of_current_month = (next_month - timedelta(days=next_month.day)).date()
    
    cutoff_date = last_idx
    # 預設訊息，防止狀態欄空白
    msg = f"✅ 資料日期正常 (最新資料: {last_idx.strftime('%Y-%m-%d')})"

    if last_idx.month == current_date.month and last_idx.year == current_date.year:
        is_calendar_end = (current_date == last_day_of_current_month)
        is_friday_end = (
            current_date.weekday() == 4 and 
            last_day_of_current_month.weekday() in [5, 6] and
            (last_day_of_current_month - current_date).days <= 2
        )
        
        if is_calendar_end or is_friday_end:
            msg = "✅ 本月交易已結束 (或為月底)，使用本月最新數據。"
        else:
            msg = "⚠️ 本月尚未結束，自動退回上個月底計算。"
            monthly_prices = monthly_prices.iloc[:-1]
            cutoff_date = monthly_prices.index[-1]

    daily_adj_close = daily_adj_close.loc[:cutoff_date]
    monthly_ret = monthly_prices.pct_change().dropna()
    daily_ret = daily_adj_close.pct_change().dropna()
    
    return monthly_ret, daily_ret, monthly_prices, assets_map, start_date, cutoff_date, msg

# ==========================================
# 執行計算與顯示
# ==========================================
data_pack = load_and_process_data()

if data_pack[0] is None:
    st.error(data_pack[6])
    st.stop()

monthly_ret, daily_ret, monthly_prices, assets_map, start_str, cutoff_date, status_msg = data_pack
tickers = list(assets_map.keys())

# --- 側邊欄檢查 ---
with st.sidebar:
    st.header("🛡️ 數據源健康度檢查")
    def get_safe_price(ticker):
        try:
            df = yf.download(ticker, period='5d', progress=False, auto_adjust=False)
            if df.empty: return 0.0
            if 'Adj Close' in df.columns: val = df['Adj Close']
            elif 'Close' in df.columns: val = df['Close']
            else: return 0.0
            if isinstance(val, pd.DataFrame): val = val.iloc[:, 0]
            return val.iloc[-1].item()
        except: return 0.0
    
    vti_price = get_safe_price('VTI')
    eem_price = get_safe_price('EEM')
    st.metric("VTI (美股基準)", f"{vti_price:.2f}")
    st.metric("EEM (新興市場)", f"{eem_price:.2f}")
    st.caption(f"即時數據驗證時間: {datetime.now().strftime('%H:%M')}")
    st.divider()
    st.info("資料源: Yahoo Finance")

# --- 主畫面 ---
st.info(f"**系統狀態**: {status_msg}")
col_k1, col_k2 = st.columns(2)
col_k1.metric("分析基準日", cutoff_date.strftime('%Y-%m-%d'))
col_k2.caption(f"策略更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# --- 第一階段：因子動能 ---
st.header("1️⃣ 第一階段：因子動能篩選")
factor_stats = []
survivors = []
current_idx = monthly_ret.index[-1]

for ticker in tickers:
    bench = assets_map[ticker]
    try:
        beta = calculate_daily_beta(ticker, bench, daily_ret)
        
        r_asset_1m = monthly_ret.loc[current_idx, ticker]
        r_bench_1m = monthly_ret.loc[current_idx, bench]
        factor_1m = r_asset_1m - (beta * r_bench_1m)
        
        p_now = monthly_prices.loc[current_idx, ticker]
        p_12m = monthly_prices.iloc[-13][ticker]
        r_asset_12m = (p_now / p_12m) - 1
        p_b_now = monthly_prices.loc[current_idx, bench]
        p_b_12m = monthly_prices.iloc[-13][bench]
        r_bench_12m = (p_b_now / p_b_12m) - 1
        factor_12m = r_asset_12m - (beta * r_bench_12m)
        
        is_pass = (factor_1m > 0) and (factor_12m > 0)
        if is_pass: survivors.append(ticker)
        
        factor_stats.append({
            'Ticker': ticker, 
            '通過?': '✅' if is_pass else '', 
            '1M Factor': factor_1m, 
            '12M Factor': factor_12m, 
            'Beta': beta
        })
    except:
        continue

df_factor = pd.DataFrame(factor_stats)

if not df_factor.empty:
    # 手動轉為百分比數值
    df_factor['1M Factor'] = df_factor['1M Factor'] * 100
    df_factor['12M Factor'] = df_factor['12M Factor'] * 100

    def color_pos_neg(val):
        color = '#28a745' if val > 0 else '#dc3545'
        return f'color: {color}'

    styler = df_factor.style.format({
        '1M Factor': '{:.2f}%',
        '12M Factor': '{:.2f}%',
        'Beta': '{:.2f}'
    }).map(color_pos_neg, subset=['1M Factor', '12M Factor'])

    st.dataframe(styler, use_container_width=True, hide_index=True)

if not survivors:
    st.error("❌ 沒有標的通過第一階段，建議持有現金 (SGOV/BIL)。")
else:
    st.success(f"✅ 晉級標的: {', '.join(survivors)}")

    # --- 第二階段：相對動能 + FIP ---
    st.divider()
    st.header("2️⃣ 第二階段：相對動能 + FIP 總分")
    
    lookbacks = [3, 6, 9, 12]
    z_scores_raw = pd.DataFrame(index=tickers)
    
    display_raw_metrics = pd.DataFrame(index=tickers)
    
    all_prices = monthly_prices[tickers]

    # Z-Score 計算
    for lb in lookbacks:
        p_now = all_prices.iloc[-1]
        p_prev = all_prices.iloc[-1 - lb]
        period_rets = (p_now / p_prev) - 1
        
        display_raw_metrics[f'{lb}M(%)'] = period_rets
        
        z_vals = zscore(period_rets, ddof=1, nan_policy='omit')
        z_scores_raw[f'Z_{lb}M'] = pd.Series(z_vals, index=tickers)

    # Daily FIP
    last_252d_daily_ret = daily_ret[tickers].tail(252)
    fip_daily_score = (last_252d_daily_ret > 0).sum() / last_252d_daily_ret.count()
    
    display_raw_metrics['FIP(%)'] = fip_daily_score
    
    z_fip_daily = zscore(fip_daily_score, ddof=1, nan_policy='omit')
    z_scores_raw['Z_FIP'] = pd.Series(z_fip_daily, index=tickers)

    # 總分計算
    final_df = z_scores_raw.loc[survivors].copy()
    raw_df = display_raw_metrics.loc[survivors].copy()
    
    final_df['Mom_Score'] = final_df[[f'Z_{lb}M' for lb in lookbacks]].sum(axis=1)
    final_df['FIP_Score'] = final_df['Z_FIP']
    final_df['Total_Score'] = final_df['Mom_Score'] + final_df['FIP_Score']
    
    final_df = final_df.sort_values(by='Total_Score', ascending=False)
    
    if not final_df.empty:
        winner = final_df.index[0]

        # A. 視覺化
        st.subheader("📊 得分結構拆解")
        chart_data = final_df[['Mom_Score', 'FIP_Score']]
        chart_data.columns = ['相對動能 (Mom)', '品質 (FIP)']
        st.bar_chart(chart_data, height=300)

        # B. 詳解表
        st.subheader("🧮 詳細數據表 (含原始報酬與 FIP)")
        st.caption("此表顯示計算出的總分，以及各回顧期的「原始報酬率」供參考。")
        
        cols_to_show = ['Total_Score', 'FIP(%)', '3M(%)', '6M(%)', '9M(%)', '12M(%)']
        merged_display = pd.concat([final_df[['Total_Score']], raw_df], axis=1)
        merged_display = merged_display.loc[final_df.index]
        
        # [核心修正] 將原始數據乘以 100，以便在前端顯示為百分比
        merged_display[['FIP(%)', '3M(%)', '6M(%)', '9M(%)', '12M(%)']] *= 100
        
        st.dataframe(
            merged_display[cols_to_show],
            use_container_width=True,
            column_config={
                "Total_Score": st.column_config.ProgressColumn("總分", format="%.2f", min_value=-10, max_value=10),
                # [核心修正] 修正格式語法為 %.2f%%
                "FIP(%)": st.column_config.NumberColumn("FIP (正報酬天數)", format="%.2f%%"),
                "3M(%)": st.column_config.NumberColumn("3M 報酬", format="%.2f%%"),
                "6M(%)": st.column_config.NumberColumn("6M 報酬", format="%.2f%%"),
                "9M(%)": st.column_config.NumberColumn("9M 報酬", format="%.2f%%"),
                "12M(%)": st.column_config.NumberColumn("12M 報酬", format="%.2f%%"),
            }
        )

        # C. 最終贏家
        st.divider()
        st.header(f"🏆 最終贏家: :red[{winner}]")
        
        col_w1, col_w2, col_w3 = st.columns(3)
        col_w1.metric("總分", f"{final_df.loc[winner, 'Total_Score']:.2f}")
        col_w2.metric("動能得分", f"{final_df.loc[winner, 'Mom_Score']:.2f}")
        col_w3.metric("FIP 得分", f"{final_df.loc[winner, 'FIP_Score']:.2f}")
        
        st.markdown("### 🔍 執行前最後確認")
        col_link1, col_link2 = st.columns(2)
        with col_link1:
            st.link_button(f"前往 TradingView ({winner})", f"https://www.tradingview.com/chart/?symbol={winner}")
        with col_link2:
            st.link_button(f"前往 Yahoo Finance ({winner})", f"https://finance.yahoo.com/quote/{winner}")
