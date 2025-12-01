import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import zscore
from datetime import datetime, timedelta

# ==========================================
# 頁面設定 (手機優化)
# ==========================================
st.set_page_config(page_title="因子動能策略監控", layout="wide")
st.title("📊 因子動能與 FIP 策略儀表板")

# ==========================================
# 核心邏輯函數
# ==========================================
def calculate_daily_beta(asset, bench, daily_df, lookback=252):
    subset = daily_df[[asset, bench]].dropna().tail(lookback)
    if len(subset) < lookback * 0.8: return 1.0
    cov = np.cov(subset[asset], subset[bench])
    return cov[0, 1] / cov[1, 1]

@st.cache_data(ttl=3600) # 設定快取 1 小時，避免重複下載
def load_and_process_data():
    assets_map = {
        'IMOM': 'EFA', 'IVAL': 'EFA', 'IDHQ': 'EFA', 'GWX': 'EFA',
        'QMOM': 'VTI', 'QVAL': 'VTI', 'SPHQ': 'VTI', 'SCHA': 'VTI',
        'PIE': 'EEM',  'DFEV': 'EEM', 'DEHP': 'EEM', 'EEMS': 'EEM'
    }
    tickers = list(assets_map.keys())
    benchmarks = list(set(assets_map.values()))
    all_symbols = tickers + benchmarks

    # 設定資料長度
    start_date = (datetime.now() - timedelta(days=365*3 + 30)).strftime('%Y-%m-%d')
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # 下載
    raw_data = yf.download(all_symbols, start=start_date, end=end_date, progress=False, auto_adjust=False)
    
    if 'Adj Close' in raw_data.columns:
        daily_adj_close = raw_data['Adj Close']
    elif 'Close' in raw_data.columns:
        daily_adj_close = raw_data['Close']
    else:
        st.error("無法下載價格資料")
        return None, None, None, None

    daily_adj_close = daily_adj_close.astype(float)
    monthly_prices = daily_adj_close.resample('ME').last()

    # --- 智能日期切割 ---
    last_idx = monthly_prices.index[-1]
    current_date = datetime.now().date()
    next_month = last_idx.replace(day=28) + timedelta(days=4)
    last_day_of_current_month = (next_month - timedelta(days=next_month.day)).date()
    
    cutoff_date = last_idx
    msg = ""

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
if data_pack[0] is not None:
    monthly_ret, daily_ret, monthly_prices, assets_map, start_str, cutoff_date, status_msg = data_pack
    tickers = list(assets_map.keys())

    # 1. 資訊顯示
    st.info(f"**狀態更新**: {status_msg}")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("資料起始日", start_str)
    with col2:
        st.metric("分析基準日 (Cutoff)", cutoff_date.strftime('%Y-%m-%d'))
    st.caption(f"最後更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 2. 因子動能篩選
    factor_stats = []
    survivors = []
    current_idx = monthly_ret.index[-1]

    for ticker in tickers:
        bench = assets_map[ticker]
        try:
            beta = calculate_daily_beta(ticker, bench, daily_ret)
            
            # 1M Pure
            r_asset_1m = monthly_ret.loc[current_idx, ticker]
            r_bench_1m = monthly_ret.loc[current_idx, bench]
            factor_1m = r_asset_1m - (beta * r_bench_1m)
            
            # 12M Pure
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
                '標的': ticker, 
                '基準': bench,
                'Beta': round(beta, 2),
                '1M 因子報酬': f"{factor_1m:.2%}", 
                '12M 因子報酬': f"{factor_12m:.2%}", 
                '結果': '✅ 通過' if is_pass else '❌ 淘汰'
            })
        except:
            continue

    st.subheader("1. 因子動能篩選 (去除 Beta 後)")
    df_factor = pd.DataFrame(factor_stats).set_index('標的')
    st.dataframe(df_factor, use_container_width=True)

    if not survivors:
        st.error("沒有標的通過第一階段篩選，建議持有現金 (SGOV/BIL)。")
    else:
        st.success(f"晉級第二階段標的 ({len(survivors)}): {', '.join(survivors)}")

        # 3. 相對動能 + FIP 計算
        lookbacks = [3, 6, 9, 12]
        z_scores_all = pd.DataFrame(index=tickers)
        display_metrics = pd.DataFrame(index=survivors)
        all_prices = monthly_prices[tickers]

        # 相對動能 Z-Score
        for lb in lookbacks:
            p_now = all_prices.iloc[-1]
            p_prev = all_prices.iloc[-1 - lb]
            period_rets = (p_now / p_prev) - 1
            z_vals = zscore(period_rets, ddof=1, nan_policy='omit')
            z_scores_all[f'Z_{lb}M'] = pd.Series(z_vals, index=tickers)
            display_metrics[f'{lb}M 報酬'] = period_rets[survivors]

        # Daily FIP
        last_252d_daily_ret = daily_ret[tickers].tail(252)
        fip_daily_score = (last_252d_daily_ret > 0).sum() / last_252d_daily_ret.count()
        z_fip_daily = zscore(fip_daily_score, ddof=1, nan_policy='omit')
        z_scores_all['Z_FIP'] = pd.Series(z_fip_daily, index=tickers)
        display_metrics['FIP (日正報酬%)'] = fip_daily_score[survivors]

        # 總分計算
        final_z_scores = z_scores_all.loc[survivors].copy()
        final_z_scores['總分 (Total Z)'] = final_z_scores.sum(axis=1)

        # 整理最終表格
        final_df = pd.concat([display_metrics, final_z_scores[['總分 (Total Z)']]], axis=1)
        # 格式化顯示百分比
        for col in final_df.columns:
            if '報酬' in col or 'FIP' in col:
                final_df[col] = final_df[col].apply(lambda x: f"{x:.2%}")
        
        final_df = final_df.sort_values(by='總分 (Total Z)', ascending=False)

        st.subheader("2. 最終排名 (相對動能 + FIP)")
        st.dataframe(final_df, use_container_width=True)

        # 4. 最終贏家
        winner = final_df.index[0]
        winner_score = final_df.loc[winner, '總分 (Total Z)']
        winner_fip = final_df.loc[winner, 'FIP (日正報酬%)']
        
        st.divider()
        st.header(f"🏆 本月最終贏家: :red[{winner}]")
        st.metric(label="總分", value=f"{winner_score:.4f}")
        st.write(f"該標的在過去一年中，有 **{winner_fip}** 的交易日是上漲的，顯示出極佳的動能品質。")
