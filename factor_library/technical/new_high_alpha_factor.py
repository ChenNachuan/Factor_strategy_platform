import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Optional, List
import warnings

# 路径：把项目根目录加入 sys.path，便于使用绝对包导入
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_manager.data import DataManager


def get_index_components(data_manager, index_code='000852.SH', trade_date=None):
    """
    获取指定指数的成分股列表
    
    Parameters
    ----------
    data_manager : DataManager
        数据管理器实例
    index_code : str
        指数代码，默认为中证1000 (000852.SH)
        可选：000300.SH (沪深300), 000905.SH (中证500), 000016.SH (上证50)
    trade_date : Optional[str]
        指定日期，格式 YYYY-MM-DD 或 YYYYMMDD
        如果为None，使用最新一期数据
    
    Returns
    -------
    List[str]
        成分股代码列表
    """
    # 直接从raw_data加载指数权重数据（该数据不需要清洗）
    from pathlib import Path
    import pandas as pd
    
    raw_data_path = Path(__file__).resolve().parent.parent.parent / 'data_manager' / 'raw_data' / 'index_weight_data.parquet'
    
    try:
        index_weights = pd.read_parquet(raw_data_path)
    except Exception as e:
        warnings.warn(f"无法加载 index_weight 数据: {e}\n请先运行 data_manager/data_loader/index_weight_data_loader.py")
        return []
    
    if index_weights is None or index_weights.empty:
        warnings.warn(f"index_weight 数据为空，请先运行 data_manager/data_loader/index_weight_data_loader.py")
        return []
    
    # 筛选指定指数
    index_data = index_weights[index_weights['index_code'] == index_code].copy()
    
    if index_data.empty:
        warnings.warn(f"未找到指数 {index_code} 的权重数据")
        return []
    
    # 如果指定了日期，筛选该日期的数据
    if trade_date is not None:
        # 转换日期格式
        if '-' in trade_date:
            trade_date = trade_date.replace('-', '')
        index_data = index_data[index_data['trade_date'] == trade_date]
        
        if index_data.empty:
            # 如果指定日期没有数据，使用最接近的日期
            warnings.warn(f"指定日期 {trade_date} 没有数据，使用最新一期数据")
            index_data = index_weights[index_weights['index_code'] == index_code].copy()
            latest_date = index_data['trade_date'].max()
            index_data = index_data[index_data['trade_date'] == latest_date]
    else:
        # 使用最新一期数据
        latest_date = index_data['trade_date'].max()
        index_data = index_data[index_data['trade_date'] == latest_date]
    
    # 提取成分股代码
    components = index_data['con_code'].unique().tolist()
    
    print(f"✅ 获取指数 {index_code} 成分股:")
    print(f"   日期: {index_data['trade_date'].iloc[0] if not index_data.empty else 'N/A'}")
    print(f"   成分股数量: {len(components)}")
    
    return components


def calculate_new_high_alpha_factor(
    data_manager: DataManager,
    start_date: str,
    end_date: str,
    stock_codes: Optional[List[str]] = None,
    high_window: int = 240,
    volume_ma_window: int = 10,
    lookback_window: int = 60,
    n1_threshold: int = 20,
    n2_threshold: int = 50,
    # 估值筛选参数
    pe_min: float = 0.0,
    pe_max: float = 150.0,
    # 市值筛选参数
    use_marketcap_filter: bool = True,
    # 回踩确认参数
    enable_pullback: bool = True,
    pullback_window: int = 10,
    pullback_min: float = 0.03,
    pullback_max: float = 0.08,
) -> pd.DataFrame:
    """
    计算创新高精选Alpha因子 (v3) - 增强版
    
    该版本在v2基础上增加了以下改进：
    1. **估值筛选**：使用PE-TTM过滤亏损和高估值股票
    2. **市值筛选**：优先选择市值排名前50%的股票（稳定性代理）
    3. **回踩确认机制**：要求突破后回调3-8%但未跌破前高才产生最终信号
    
    **关键改进**：
    - 使用 daily_basic 中的换手率替代成交额进行放量确认
    - 使用 PE-TTM 和市值作为基本面筛选，替代分析师预测EPS
    - 引入回踩确认机制，提高信号质量
    
    **默认股票池**：中证1000成分股（000852.SH）

    Parameters
    ----------
    data_manager : DataManager
        数据管理器实例。
    start_date, end_date : str
        回测周期。
    stock_codes : Optional[List[str]]
        股票池。如果为None，则自动使用中证1000成分股。
    high_window : int
        新高窗口期，默认240天。
    volume_ma_window : int
        换手率均线窗口，默认10天。
    lookback_window : int
        前期表现回看窗口，默认60天。
    n1_threshold, n2_threshold : int
        动态筛选样本数阈值。
    pe_min : float
        PE-TTM最小值，默认0（排除亏损股）。
    pe_max : float
        PE-TTM最大值，默认150（排除高估值股）。
    use_marketcap_filter : bool
        是否使用市值筛选，默认True。
    enable_pullback : bool
        是否启用回踩确认机制，默认True。
    pullback_window : int
        回踩观察窗口，默认10天。
    pullback_min : float
        最小回调幅度，默认3%。
    pullback_max : float
        最大回调幅度，默认8%。

    Returns
    -------
    pd.DataFrame
        MultiIndex (trade_date, ts_code) with single column 'factor'.
        因子值为1（入选）或NaN（未入选）。
    """
    # 步骤 1: 初始信号生成
    print("\n" + "=" * 60)
    print("步骤 1: 初始信号生成")
    
    # 1.1 确定选股域
    if stock_codes is None:
        print("未指定股票池，使用中证1000成分股...")
        stock_codes = get_index_components(data_manager, index_code='000852.SH')
        
        if not stock_codes:
            warnings.warn("无法获取中证1000成分股，将使用全市场股票。请确保已运行 index_weight_data_loader.py")
            stock_codes = None
        else:
            print(f"✅ 成功加载中证1000成分股，共 {len(stock_codes)} 只股票")

    # 1.2 加载数据
    buffer_days = max(high_window, lookback_window) * 2
    start_date_extended = (pd.to_datetime(start_date) - pd.Timedelta(days=buffer_days)).strftime('%Y-%m-%d')
    
    # 加载日线数据
    daily = data_manager.load_data('daily', start_date=start_date_extended, end_date=end_date, stock_codes=stock_codes)
    if daily is None or daily.empty:
        raise ValueError("无法加载日行情数据。")
    
    # 加载 daily_basic 数据以获取换手率
    daily_basic = data_manager.load_data('daily_basic', start_date=start_date_extended, end_date=end_date, stock_codes=stock_codes)
    if daily_basic is None or daily_basic.empty:
        raise ValueError("无法加载 daily_basic 数据。")
    
    daily = daily.copy()
    daily['trade_date'] = pd.to_datetime(daily['trade_date'])
    daily = daily.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    
    # 合并换手率、PE-TTM和市值数据
    daily_basic = daily_basic.copy()
    daily_basic['trade_date'] = pd.to_datetime(daily_basic['trade_date'])
    daily = pd.merge(
        daily,
        daily_basic[['ts_code', 'trade_date', 'turnover_rate', 'pe_ttm', 'total_mv']],
        on=['ts_code', 'trade_date'],
        how='left'
    )
    
    print(f"✅ 成功加载数据，包含换手率字段: {'turnover_rate' in daily.columns}")
    print(f"   包含PE-TTM字段: {'pe_ttm' in daily.columns}")
    print(f"   包含市值字段: {'total_mv' in daily.columns}")
    print(f"   数据时间范围: {daily['trade_date'].min()} ~ {daily['trade_date'].max()}")
    print(f"   股票数量: {daily['ts_code'].nunique()}")

    # 1.3 识别创新高事件
    # 过去240个交易日（不含当日）的收盘价新高
    rolling_max_close = daily.groupby('ts_code')['close'].transform(
        lambda x: x.rolling(window=high_window, min_periods=high_window).max().shift(1)
    )
    is_new_high = daily['close'] > rolling_max_close
    
    initial_count = is_new_high.sum()
    print(f"识别到 {initial_count} 个初始创新高事件。")
    
    # 步骤 2: 时序筛选
    print("\n" + "=" * 60)
    print("步骤 2: 时序筛选（剔除假突破样本）")
    
    # 2.1 剔除涨停样本
    is_limit_up_today = daily['pct_chg'] > 9.8
    daily['next_open'] = daily.groupby('ts_code')['open'].shift(-1)
    is_limit_up_next_open = (daily['next_open'] / daily['close'] - 1) > 0.098
    
    valid_mask = is_new_high & (~is_limit_up_today) & (~is_limit_up_next_open)
    print(f"剔除涨停后，剩余 {valid_mask.sum()} 个事件。")

    # 2.2 确认放量突破（使用换手率）
    print("计算换手率确认指标...")
    daily['turnover_ma'] = daily.groupby('ts_code')['turnover_rate'].transform(
        lambda x: x.rolling(window=volume_ma_window).mean()
    )
    
    # 找到前期高点对应的换手率MA
    def get_prev_high_turnover_ma(group):
        """对单只股票计算前期高点的换手率MA"""
        result = pd.Series(index=group.index, dtype=float)
        
        for i in range(len(group)):
            if i < high_window:
                result.iloc[i] = np.nan
                continue
            
            # 获取前期窗口（i-high_window 到 i-1）
            window_close = group['close'].iloc[i - high_window : i]
            window_turnover_ma = group['turnover_ma'].iloc[i - high_window : i]
            
            # 找到窗口内最高价对应的索引
            if not window_close.empty:
                max_idx = window_close.idxmax()
                result.iloc[i] = window_turnover_ma.loc[max_idx]
            else:
                result.iloc[i] = np.nan
                
        return result
    
    daily['prev_high_turnover_ma'] = daily.groupby('ts_code', group_keys=False).apply(
        get_prev_high_turnover_ma
    ).values
    
    is_volume_breakthrough = daily['turnover_ma'] > daily['prev_high_turnover_ma']
    valid_mask = valid_mask & is_volume_breakthrough.fillna(False)
    
    # 2.3 估值筛选：过滤PE-TTM不在合理区间的股票
    print(f"应用估值筛选: {pe_min} < PE-TTM < {pe_max}...")
    is_pe_valid = (daily['pe_ttm'] > pe_min) & (daily['pe_ttm'] < pe_max)
    valid_mask = valid_mask & is_pe_valid.fillna(False)
    
    effective_new_high = daily[valid_mask].copy()
    print(f"估值筛选后，每日有效创新高样本池构建完成，共 {len(effective_new_high)} 个事件。")

    # 步骤 3: 计算截面筛选所需指标
    print("\n" + "=" * 60)
    print("步骤 3: 计算截面筛选所需指标")
    
    daily['prior_return'] = daily.groupby('ts_code')['close'].transform(
        lambda x: x.pct_change(periods=lookback_window).shift(1)
    )
    
    # 计算前期平均换手率
    daily['prior_turnover'] = daily.groupby('ts_code')['turnover_rate'].transform(
        lambda x: x.rolling(window=lookback_window).mean().shift(1)
    )
    
    # 提取市值数据（已经在daily_basic中）
    # total_mv 字段已经在前面merge时加入

    effective_new_high = pd.merge(
        effective_new_high[['trade_date', 'ts_code', 'close']],  # 保留close用于回踩确认
        daily[['trade_date', 'ts_code', 'prior_return', 'prior_turnover', 'total_mv']],
        on=['trade_date', 'ts_code'],
        how='left'
    )

    # 步骤 4: 动态截面筛选与最终因子生成
    print("\n" + "=" * 60)
    print("步骤 4: 动态截面筛选（加入市值筛选）")
    
    final_selection = []
    
    for date, group in effective_new_high.groupby('trade_date'):
        n = len(group)
        if n == 0:
            continue
            
        pr_median = group['prior_return'].quantile(0.5)
        pt_median = group['prior_turnover'].quantile(0.5)
        
        # 市值筛选：只选择市值排名前50%的股票
        if use_marketcap_filter:
            mv_median = group['total_mv'].quantile(0.5)
            group = group[group['total_mv'] >= mv_median]
        
        selected_group = None
        
        if n < n1_threshold:
            # 情况一：N < 20，仅使用指标A（前期涨跌幅）
            selected_group = group[group['prior_return'] <= pr_median]
        else:
            # 情况二和三：N >= 20，综合使用指标A和指标B
            # 使用PE-TTM和市值替代EPS
            selected_group = group[
                (group['prior_return'] <= pr_median) &
                (group['prior_turnover'] <= pt_median)
            ]
            
        if selected_group is not None and not selected_group.empty:
            final_selection.append(selected_group)

    if not final_selection:
        print("⚠️ 警告：没有找到任何最终入选的股票。")
        return pd.DataFrame(columns=['factor']).rename_axis(['trade_date', 'ts_code'])
        
    final_df = pd.concat(final_selection, ignore_index=True)
    
    # 步骤 5: 回踩确认机制（可选）
    if enable_pullback:
        print("\n" + "=" * 60)
        print(f"步骤 5: 回踩确认机制（观察窗口={pullback_window}天，回调范围={pullback_min*100:.1f}%-{pullback_max*100:.1f}%）")
        
        # 为每个初选信号创建观察池
        final_df = final_df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        final_df['signal_high'] = final_df['close']  # 记录信号产生时的价格（新高价格）
        
        # 合并后续价格数据以进行回踩检测
        pullback_confirmed = []
        
        for idx, row in final_df.iterrows():
            ts_code = row['ts_code']
            signal_date = row['trade_date']
            signal_high = row['signal_high']
            
            # 获取该股票在信号日之后pullback_window天内的数据
            future_data = daily[
                (daily['ts_code'] == ts_code) &
                (daily['trade_date'] > signal_date) &
                (daily['trade_date'] <= signal_date + pd.Timedelta(days=pullback_window * 2))  # 留足自然日
            ].head(pullback_window).copy()
            
            if future_data.empty:
                continue
            
            # 计算回调幅度（相对于新高价格）
            future_data['pullback_pct'] = (future_data['close'] - signal_high) / signal_high
            
            # 找到是否存在符合条件的回踩：
            # 1. 回调幅度在 -pullback_max 到 -pullback_min 之间（即下跌3%-8%）
            # 2. 没有跌破前高（这里简化为没有跌破信号日价格的90%，可以根据需要调整）
            valid_pullback = future_data[
                (future_data['pullback_pct'] >= -pullback_max) &
                (future_data['pullback_pct'] <= -pullback_min) &
                (future_data['close'] >= signal_high * 0.92)  # 未跌破前高太多
            ]
            
            if not valid_pullback.empty:
                # 取第一个符合条件的回踩日期作为最终信号日期
                confirm_date = valid_pullback.iloc[0]['trade_date']
                pullback_confirmed.append({
                    'trade_date': confirm_date,
                    'ts_code': ts_code,
                    'signal_date': signal_date,
                    'pullback_pct': valid_pullback.iloc[0]['pullback_pct']
                })
        
        if pullback_confirmed:
            pullback_df = pd.DataFrame(pullback_confirmed)
            pullback_df['factor'] = 1.0
            factor_data = pullback_df[['trade_date', 'ts_code', 'factor']].set_index(['trade_date', 'ts_code'])
            print(f"✅ 回踩确认完成！从 {len(final_df)} 个初选信号中确认了 {len(pullback_df)} 个回踩信号。")
        else:
            print("⚠️ 警告：没有找到任何符合回踩条件的信号。")
            return pd.DataFrame(columns=['factor']).rename_axis(['trade_date', 'ts_code'])
    else:
        # 不使用回踩确认，直接使用截面筛选结果
        final_df['factor'] = 1.0
        factor_data = final_df[['trade_date', 'ts_code', 'factor']].set_index(['trade_date', 'ts_code'])
    
    print(f"\n✅ 创新高精选 Alpha 因子计算完成！共 {len(factor_data)} 条有效记录。")
    
    return factor_data.loc[factor_data.index.get_level_values('trade_date') >= pd.to_datetime(start_date)]



def run_new_high_alpha_backtest(
    start_date: str = '2022-01-01',
    end_date: str = '2023-12-31',
    stock_codes: Optional[List[str]] = None,
    high_window: int = 240,
    volume_ma_window: int = 10,
    lookback_window: int = 60,
    rebalance_freq: str = 'weekly',
    transaction_cost: float = 0.0003,
    # 新增参数
    pe_min: float = 0.0,
    pe_max: float = 150.0,
    use_marketcap_filter: bool = True,
    enable_pullback: bool = True,
    pullback_window: int = 10,
    pullback_min: float = 0.03,
    pullback_max: float = 0.08,
) -> dict:
    """
    运行新版创新高精选Alpha因子（v3 - 增强版）的回测。
    
    由于因子是二元值（1/NaN），此回测采用Long-Only策略，
    即每日等权持有所有因子值为1的股票。
    
    **v3新增功能**：
    - PE-TTM估值筛选
    - 市值筛选（优先大市值）
    - 回踩确认机制
    
    **默认股票池**：中证1000成分股

    Parameters
    ----------
    start_date, end_date : str
        回测周期。
    stock_codes : Optional[List[str]]
        股票池。如果为None，则自动使用中证1000成分股。
    high_window, volume_ma_window, lookback_window : int
        因子计算所需参数。
    rebalance_freq : str
        调仓频率，用于估算交易成本。
    transaction_cost : float
        单边交易成本。
    pe_min, pe_max : float
        PE-TTM筛选区间。
    use_marketcap_filter : bool
        是否使用市值筛选。
    enable_pullback : bool
        是否启用回踩确认机制。
    pullback_window : int
        回踩观察窗口。
    pullback_min, pullback_max : float
        回踩幅度区间。

    Returns
    -------
    dict
        包含回测结果的字典。
    """
    # 初始化数据管理器
    data_manager = DataManager()
    
    # 计算因子
    factor_data = calculate_new_high_alpha_factor(
        data_manager=data_manager,
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_codes,
        high_window=high_window,
        volume_ma_window=volume_ma_window,
        lookback_window=lookback_window,
        pe_min=pe_min,
        pe_max=pe_max,
        use_marketcap_filter=use_marketcap_filter,
        enable_pullback=enable_pullback,
        pullback_window=pullback_window,
        pullback_min=pullback_min,
        pullback_max=pullback_max,
    )
    
    if factor_data.empty:
        print("因子数据为空，无法回测。")
        return {
            'factor_data': None,
            'portfolio_returns': None,
            'performance_metrics': {},
            'analysis_results': {}
        }
        
    # 准备收益率数据
    stock_list = factor_data.index.get_level_values('ts_code').unique().tolist()
    
    stock_data = data_manager.load_data(
        'daily',
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_list
    )
    if stock_data is None or stock_data.empty:
        raise ValueError("无法加载用于回测的股票数据。")

    stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'])
    stock_data = stock_data.sort_values(['ts_code', 'trade_date'])
    # 计算下一日收益率：(明天的收盘价 / 今天的收盘价) - 1
    stock_data['next_close'] = stock_data.groupby('ts_code')['close'].shift(-1)
    stock_data['next_return'] = (stock_data['next_close'] / stock_data['close']) - 1
    
    # 合并数据
    combined = pd.merge(
        factor_data.reset_index(),
        stock_data[['trade_date', 'ts_code', 'next_return']],
        on=['trade_date', 'ts_code'],
        how='inner'
    )
    
    # 去除NaN值
    combined = combined.dropna(subset=['next_return'])
    
    # 计算 Long-Only 组合每日收益
    portfolio_returns = combined.groupby('trade_date')['next_return'].mean()
    
    # 模拟交易成本
    # 获取调仓日
    if rebalance_freq == 'daily':
        rebalance_dates = portfolio_returns.index
    else:
        freq_map = {'weekly': 'W-MON', 'monthly': 'MS'}
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=freq_map.get(rebalance_freq))
    
    # 近似换手率成本
    if rebalance_dates is not None and len(portfolio_returns) > 0:
        cost_impact = len(rebalance_dates) * transaction_cost / len(portfolio_returns)
        portfolio_returns -= cost_impact
    
    # 计算业绩指标
    cum_returns = (1 + portfolio_returns).cumprod()
    total_return = cum_returns.iloc[-1] - 1 if not cum_returns.empty else 0
    
    days = len(portfolio_returns)
    annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
    
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    running_max = cum_returns.cummax()
    drawdown = cum_returns / running_max - 1
    max_drawdown = drawdown.min() if not drawdown.empty else 0

    # IC分析：计算因子值与未来收益的相关性
    # 对于二元因子（1/NaN），我们计算每日入选股票的平均收益 vs 全市场平均收益的差异
    ic_series = None
    ic_mean = None
    ic_std = None
    icir = None
    ic_positive_ratio = None
    
    try:
        # 按日期分组计算IC
        ic_list = []
        for date in combined['trade_date'].unique():
            date_data = combined[combined['trade_date'] == date]
            if len(date_data) >= 2:  # 至少需要2个样本
                # 因子值为1的股票
                factor_returns = date_data['next_return']
                
                # 加载当天全市场数据计算基准
                all_stocks = stock_data[stock_data['trade_date'] == date]['next_return']
                if len(all_stocks) > 0:
                    market_avg = all_stocks.mean()
                    factor_avg = factor_returns.mean()
                    # IC定义为因子选股的超额收益
                    ic = factor_avg - market_avg
                    ic_list.append({'trade_date': date, 'ic': ic})
        
        if ic_list:
            ic_series = pd.DataFrame(ic_list).set_index('trade_date')['ic']
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            icir = ic_mean / ic_std if ic_std > 0 else 0
            ic_positive_ratio = (ic_series > 0).mean()
    except Exception as e:
        print(f"⚠️ IC计算失败: {e}")
        ic_series = None

    return {
        'factor_data': factor_data,
        'portfolio_returns': portfolio_returns,
        'performance_metrics': {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
        },
        'analysis_results': {
            'ic_series': ic_series,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'icir': icir,
            'ic_positive_ratio': ic_positive_ratio,
        }
    }



def main():
    """主函数：演示新版创新高精选Alpha因子计算和回测（v3 - 增强版）"""
    print("=" * 60)
    print("创新高精选 Alpha 因子 (v3 - 增强版)")
    print("股票池: 中证1000成分股")
    print("新功能: PE-TTM筛选 + 市值筛选 + 回踩确认")
    print("=" * 60)

    try:
        # 配置参数
        config = {
            'start_date': '2024-01-01',
            'end_date': '2025-09-30',
            'high_window': 240,
            'volume_ma_window': 10,
            'lookback_window': 60,
            'rebalance_freq': 'weekly',
            'transaction_cost': 0.0003,
            # v3新增参数
            'pe_min': 0.0,
            'pe_max': 150.0,
            'use_marketcap_filter': True,
            'enable_pullback': True,
            'pullback_window': 10,
            'pullback_min': 0.03,
            'pullback_max': 0.08,
        }

        print("\n回测配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        # 运行回测
        results = run_new_high_alpha_backtest(**config)

        # 打印结果
        print("\n" + "=" * 60)
        print("回测结果")
        print("=" * 60)

        if results['portfolio_returns'] is not None:
            metrics = results['performance_metrics']
            print(f"\n📊 业绩指标:")
            print(f"  总收益率: {metrics['total_return']:.2%}")
            print(f"  年化收益率: {metrics['annualized_return']:.2%}")
            print(f"  年化波动率: {metrics['volatility']:.2%}")
            print(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
            print(f"  最大回撤: {metrics['max_drawdown']:.2%}")

            # IC分析结果
            analysis = results['analysis_results']
            if analysis.get('ic_series') is not None:
                print(f"\n📊 IC分析:")
                print(f"  IC均值: {analysis['ic_mean']:.4f}")
                print(f"  IC标准差: {analysis['ic_std']:.4f}")
                print(f"  ICIR: {analysis['icir']:.4f}")
                print(f"  IC>0占比: {analysis['ic_positive_ratio']:.2%}")
            else:
                print(f"\n⚠️ IC分析: 数据不足，无法计算")

            print(f"\n📈 因子覆盖:")
            factor_data = results['factor_data']
            print(f"  有效因子记录数: {len(factor_data)}")
            print(f"  覆盖股票数: {factor_data.index.get_level_values('ts_code').nunique()}")
            print(f"  覆盖交易日数: {factor_data.index.get_level_values('trade_date').nunique()}")
        else:
            print("⚠️ 回测失败，无可用数据。")

    except Exception as e:
        print(f"\n❌ 运行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
