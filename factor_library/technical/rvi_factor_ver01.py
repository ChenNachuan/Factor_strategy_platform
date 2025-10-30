"""
RVI（Relative Vigor Index）相对活力指标因子

本模块实现了基于RVI技术指标的量化选股因子，包括多种因子变体和组合策略。

**RVI指标简介**：
RVI通过比较收盘价相对开盘价的位置与价格波动范围，衡量价格变动的"活力"或"动能"。
它假设在上涨趋势中，收盘价倾向于接近最高价；在下跌趋势中，收盘价倾向于接近最低价。

**指标优势**：
- 综合考虑开盘价、收盘价、最高价、最低价四个价格
- 类似MACD，通过双线交叉产生交易信号
- 对短期价格动能敏感，适合捕捉趋势转折点

**因子类型**：
1. 基础因子
   - value: RVI原始值（动能强度）
   - cross: 金叉/死叉信号（交易时机）
   - diff: RVI与信号线差值（偏离度）
   - strength: 交叉强度（突破力度）

2. 组合因子（提高信号质量）
   - rvi_volume: RVI金叉 + 成交量放大（双重确认）
   - rvi_trend: RVI金叉 + 价格在均线上方（顺势交易）

**使用示例**：
    >>> from data_manager.data import DataManager
    >>> 
    >>> # 初始化数据管理器
    >>> dm = DataManager()
    >>> 
    >>> # 计算基础交叉因子
    >>> factor = calculate_rvi_factor(
    ...     data_manager=dm,
    ...     start_date='2020-01-01',
    ...     end_date='2023-12-31',
    ...     factor_type='cross'
    ... )
    >>> 
    >>> # 运行回测
    >>> results = run_rvi_factor_backtest(
    ...     start_date='2020-01-01',
    ...     end_date='2023-12-31',
    ...     factor_type='rvi_volume',
    ...     rebalance_freq='weekly'
    ... )
    >>> 
    >>> # 查看业绩
    >>> print(results['performance_metrics'])

**主要函数**：
- calculate_rvi_factor: 计算RVI因子
- run_rvi_factor_backtest: 运行回测并计算业绩指标
- main: 演示多策略对比

**依赖项**：
- pandas, numpy: 数据处理
- data_manager.data: 数据加载
- backtest_engine.engine: 回测引擎

**参考文献**：
- Dorsey, John F. "The Relative Vigor Index." 
  Technical Analysis of Stocks & Commodities, 1995.

Author: Investment Assignment
Date: 2025-10-30
Version: 2.0 (增强版，包含组合因子)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Optional, List

# 路径设置
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_manager.data import DataManager

def calculate_rvi_factor(
    data_manager: DataManager,
    start_date: str,
    end_date: str,
    stock_codes: Optional[List[str]] = None,
    period: int = 10,
    signal_period: int = 4,
    factor_type: str = 'cross',
    volume_ma_period: int = 20,
    trend_ma_period: int = 20
) -> pd.DataFrame:
    """
    计算RVI因子（Relative Vigor Index - 相对活力指标）
    
    **因子原理**：
    RVI通过比较收盘价相对开盘价的位置与当日价格波动范围，衡量价格变动的"活力"。
    当收盘价接近最高价时，表明多头力量强劲；当接近最低价时，表明空头占优。
    RVI指标与MACD类似，通过RVI线与信号线的交叉产生交易信号。
    
    **计算公式**：
    1. Vigor = (Close - Open) / (High - Low)  # 价格活力
    2. Numerator = WMA(Vigor, 4)  # 分子：Vigor的4期加权移动平均
    3. Denominator = WMA(High - Low, 4)  # 分母：振幅的4期加权移动平均
    4. RVI = Numerator / Denominator  # 相对活力指标
    5. Signal = WMA(RVI, 4)  # 信号线：RVI的4期加权移动平均
    
    其中WMA权重为：(1, 2, 2, 1) / 6
    
    **因子逻辑**：
    - RVI > 0: 多头力量占优（收盘价 > 开盘价）
    - RVI < 0: 空头力量占优（收盘价 < 开盘价）
    - 金叉（RVI上穿Signal）: 动能转强，买入信号（看涨）
    - 死叉（RVI下穿Signal）: 动能转弱，卖出信号（看跌）
    - |RVI|越大: 价格变动活力越强
    
    **因子方向**：
    对于不同的factor_type：
    - 'value'/'diff': 高因子值 = 高RVI = 强多头动能（做多高值）
    - 'cross': 金叉(+1) = 买入信号，死叉(-1) = 卖出信号（做多金叉）
    - 'rvi_volume': 金叉+放量 = 强确认信号（做多高值）
    - 'rvi_trend': 金叉+趋势向上 = 顺势交易（做多高值）
    
    **因子特性**：
    - IC通常为正（高RVI对应未来正收益）
    - 适合短周期交易（日频、周频）
    - 对突发性行情敏感
    - 在震荡市中容易产生假信号，需要结合成交量或趋势过滤
    
    Parameters
    ----------
    data_manager : DataManager
        数据管理器实例
    start_date : str
        开始日期，格式 'YYYY-MM-DD'
    end_date : str
        结束日期，格式 'YYYY-MM-DD'
    stock_codes : Optional[List[str]]
        股票代码列表，如为 None 则使用所有可用股票
    period : int
        RVI计算周期，默认10（注：实际使用固定的4期加权MA）
    signal_period : int
        信号线周期，默认4（对RVI进行4期加权移动平均）
    factor_type : str
        因子类型，决定因子值的计算方式：
        - 'value': RVI原始值（连续值，范围通常在-1到1之间）
        - 'cross': 交叉信号（离散值：金叉=1, 死叉=-1, 无信号=0）
        - 'diff': RVI与信号线的差值（RVI - Signal，衡量偏离度）
        - 'strength': 交叉强度（交叉时刻的RVI变化率，衡量突破力度）
        - 'rvi_volume': RVI+成交量组合（金叉且放量，因子值=RVI×放量倍数）
        - 'rvi_trend': RVI+趋势组合（金叉且价格在均线上，因子值=RVI×价格强度）
    volume_ma_period : int
        成交量均线周期，默认20天（仅用于'rvi_volume'类型）
    trend_ma_period : int
        趋势均线周期，默认20天（仅用于'rvi_trend'类型）

    Returns
    -------
    DataFrame
        MultiIndex (trade_date, ts_code) with single column 'factor'.
        因子值含义取决于factor_type：
        - 'value': 连续值，范围约[-1, 1]，越高表示多头动能越强
        - 'cross': 离散值，1=金叉（买入）, -1=死叉（卖出）, 0=无信号
        - 'diff': 连续值，正值表示RVI在信号线上方（强势）
        - 'strength': 连续值，交叉时的动量变化率
        - 'rvi_volume'/'rvi_trend': 连续值，组合信号强度
        
    Examples
    --------
    >>> # 基础交叉策略
    >>> factor = calculate_rvi_factor(
    ...     data_manager=dm,
    ...     start_date='2020-01-01',
    ...     end_date='2023-12-31',
    ...     factor_type='cross'
    ... )
    >>> 
    >>> # 成交量确认策略
    >>> factor = calculate_rvi_factor(
    ...     data_manager=dm,
    ...     start_date='2020-01-01',
    ...     end_date='2023-12-31',
    ...     factor_type='rvi_volume',
    ...     volume_ma_period=20
    ... )
    
    Notes
    -----
    - RVI指标对价格的开盘价、收盘价、最高价、最低价四个价格敏感
    - 涨跌停板会导致High=Low，已做除零保护（返回0）
    - 金叉信号配合放量确认，可以提高信号质量，减少假突破
    - 建议与趋势指标（如MA）或波动率指标（如ATR）结合使用
    - 在强趋势市场中效果较好，在震荡市场中需谨慎使用
    
    References
    ----------
    - Dorsey, John F. "The Relative Vigor Index." Technical Analysis of Stocks & Commodities, 1995.
    - 类似指标：RSI（基于涨跌幅）、Stochastic（基于价格位置）
    """
    print(f"\n{'='*60}")
    print("RVI (Relative Vigor Index) 因子计算")
    print(f"{'='*60}")
    print(f"因子类型: {factor_type}")
    print(f"计算周期: RVI={period}, 信号线={signal_period}")
    if factor_type == 'rvi_volume':
        print(f"成交量MA周期: {volume_ma_period}")
    elif factor_type == 'rvi_trend':
        print(f"趋势MA周期: {trend_ma_period}")
    
    # 参数验证
    print(f"\n步骤 1: 参数验证")
    try:
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        if start_date_dt >= end_date_dt:
            raise ValueError(f"开始日期({start_date})必须早于结束日期({end_date})")
        print(f"  ✅ 日期范围: {start_date} ~ {end_date}")
    except Exception as e:
        raise ValueError(f"❌ 日期格式错误: {e}")
    
    if period < 1:
        raise ValueError(f"❌ period必须大于0，当前值: {period}")
    if signal_period < 1:
        raise ValueError(f"❌ signal_period必须大于0，当前值: {signal_period}")
    if volume_ma_period < 1:
        raise ValueError(f"❌ volume_ma_period必须大于0，当前值: {volume_ma_period}")
    if trend_ma_period < 1:
        raise ValueError(f"❌ trend_ma_period必须大于0，当前值: {trend_ma_period}")
    
    valid_factor_types = ['value', 'cross', 'diff', 'strength', 'rvi_volume', 'rvi_trend']
    if factor_type not in valid_factor_types:
        raise ValueError(
            f"❌ 不支持的factor_type: '{factor_type}'\n"
            f"   支持的类型: {', '.join(valid_factor_types)}"
        )
    print(f"  ✅ 参数验证通过")
    
    # 步骤2: 确定股票池
    print(f"\n步骤 2: 确定股票池")
    if stock_codes is None:
        print("  未指定股票池，使用全市场股票...")
        try:
            all_daily = data_manager.load_data('daily', start_date=start_date, end_date=end_date, cleaned=True)
            if all_daily is None or all_daily.empty:
                print("  ⚠️  警告：无法获取市场数据，使用默认示例股票")
                stock_codes = ['000001.SZ', '000002.SZ', '000858.SZ', '600000.SH', '600036.SH', '600519.SH']
            else:
                stock_codes = all_daily['ts_code'].unique().tolist()
        except Exception as e:
            print(f"  ⚠️  加载市场数据失败: {e}，使用默认股票池")
            stock_codes = ['000001.SZ', '000002.SZ', '000858.SZ', '600000.SH', '600036.SH', '600519.SH']
        print(f"  ✅ 股票池: {len(stock_codes)} 只股票")
    else:
        if not isinstance(stock_codes, list) or len(stock_codes) == 0:
            raise ValueError("❌ stock_codes必须是非空列表")
        print(f"  ✅ 使用指定股票池: {len(stock_codes)} 只股票")
        if len(stock_codes) <= 10:
            print(f"     股票列表: {stock_codes}")
        else:
            print(f"     示例: {stock_codes[:5]} ...")

    # 步骤3: 计算数据缓冲期（确保有足够历史数据）
    print(f"\n步骤 3: 加载历史数据")
    buffer_days = max(period, signal_period, volume_ma_period, trend_ma_period) * 3
    start_date_extended = (pd.to_datetime(start_date) - pd.Timedelta(days=buffer_days)).strftime('%Y-%m-%d')
    print(f"  请求日期范围: {start_date_extended} ~ {end_date}")
    print(f"  缓冲天数: {buffer_days} 天（确保指标计算完整性）")
    
    # 加载日线数据（OHLC + 成交量）
    daily = data_manager.load_data('daily', start_date=start_date_extended, end_date=end_date, stock_codes=stock_codes)
    if daily is None or daily.empty:
        raise ValueError(
            f'❌ 无法获取日行情数据（daily）\n'
            f'   请检查：\n'
            f'   1. 数据管理器是否正确初始化\n'
            f'   2. 日期范围是否合理: {start_date_extended} ~ {end_date}\n'
            f'   3. 股票代码是否正确: {stock_codes[:5]}...'
        )
    
    # 立即进行日期处理和排序（参考pe_factor.py和new_high_alpha_factor.py）
    daily = daily.copy()
    daily['trade_date'] = pd.to_datetime(daily['trade_date'])
    daily = daily.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    
    print(f"  ✅ 成功加载日线数据")
    print(f"     记录数: {len(daily):,} 条")
    print(f"     时间跨度: {len(daily) / len(stock_codes) if stock_codes else 0:.0f} 天")
    
    # 检查数据量是否充足
    if len(daily) < buffer_days:
        print(f"  ⚠️  警告: 数据量({len(daily)})可能不足以计算因子（建议>{buffer_days}）")

    # 步骤4: 加载daily_basic数据（换手率等）
    if factor_type == 'rvi_volume':
        print(f"\n步骤 4: 加载换手率数据（用于{factor_type}因子）")
        try:
            daily_basic = data_manager.load_data(
                'daily_basic', 
                start_date=start_date_extended, 
                end_date=end_date, 
                stock_codes=stock_codes
            )
            
            if daily_basic is not None and not daily_basic.empty:
                # 检查是否有turnover_rate字段
                if 'turnover_rate' in daily_basic.columns:
                    print(f"  ✅ 成功加载daily_basic数据: {len(daily_basic):,} 条记录")
                    
                    # 立即处理日期（参考new_high_alpha_factor.py）
                    daily_basic = daily_basic.copy()
                    daily_basic['trade_date'] = pd.to_datetime(daily_basic['trade_date'])
                    
                    # 合并换手率数据到daily
                    daily_basic_subset = daily_basic[['ts_code', 'trade_date', 'turnover_rate']].copy()
                    
                    # 合并前记录原始行数
                    original_len = len(daily)
                    daily = pd.merge(daily, daily_basic_subset, on=['ts_code', 'trade_date'], how='left')
                    
                    if len(daily) != original_len:
                        print(f"  ⚠️  警告: 合并后行数变化 ({original_len} → {len(daily)})")
                    else:
                        print(f"  ✅ 换手率数据合并完成，保持 {len(daily):,} 条记录")
                    
                    # 检查turnover_rate的有效性
                    valid_turnover = daily['turnover_rate'].notna().sum()
                    if valid_turnover > 0:
                        print(f"     换手率有效数据: {valid_turnover:,} 条 ({valid_turnover/len(daily)*100:.1f}%)")
                    else:
                        print(f"  ⚠️  turnover_rate全部为空，将使用vol字段")
                else:
                    print("  ⚠️  daily_basic中缺少turnover_rate字段，将使用vol")
            else:
                print("  ⚠️  无法加载daily_basic数据，将使用vol字段进行成交量分析")
        except Exception as e:
            print(f"  ⚠️  加载daily_basic数据时出错: {e}")
            print("     将使用vol字段进行成交量分析")

    # 步骤5: 数据质量检查
    print(f"\n步骤 5: 数据质量检查")
    
    # 数据统计
    print(f"  ✅ 数据预处理完成")
    print(f"     时间范围: {daily['trade_date'].min().date()} ~ {daily['trade_date'].max().date()}")
    print(f"     覆盖股票: {daily['ts_code'].nunique()} 只")
    print(f"     总记录数: {len(daily):,} 条")
    
    # 检查必需字段
    required_fields = ['open', 'high', 'low', 'close', 'vol']
    missing_fields = [f for f in required_fields if f not in daily.columns]
    if missing_fields:
        raise ValueError(
            f"❌ 数据缺少必需字段: {missing_fields}\n"
            f"   当前字段: {list(daily.columns)}"
        )
    
    # 检查缺失值和异常值
    print(f"\n  字段质量详情:")
    quality_issues = []
    for field in required_fields:
        missing_count = daily[field].isna().sum()
        missing_pct = missing_count / len(daily) * 100 if len(daily) > 0 else 0
        zero_count = (daily[field] == 0).sum()
        zero_pct = zero_count / len(daily) * 100 if len(daily) > 0 else 0
        
        status = "✅" if missing_pct < 1 and zero_pct < 5 else "⚠️"
        print(f"    {status} {field}: 缺失{missing_pct:.2f}%, 零值{zero_pct:.2f}%")
        
        if missing_pct > 5:
            quality_issues.append(f"{field}缺失值过多({missing_pct:.1f}%)")
        
        # 检查负值
        if field in ['open', 'high', 'low', 'close', 'vol']:
            negative_count = (daily[field] < 0).sum()
            if negative_count > 0:
                print(f"       ⚠️  {negative_count} 个负值（异常）")
                quality_issues.append(f"{field}有{negative_count}个负值")
    
    # 检查OHLC逻辑
    logic_errors = (
        (daily['high'] < daily['low']) |
        (daily['close'] > daily['high']) |
        (daily['close'] < daily['low']) |
        (daily['open'] > daily['high']) |
        (daily['open'] < daily['low'])
    ).sum()
    if logic_errors > 0:
        print(f"    ⚠️  OHLC逻辑: 发现 {logic_errors} 条异常（如high<low）")
        quality_issues.append(f"OHLC逻辑错误({logic_errors}条)")
    else:
        print(f"    ✅ OHLC逻辑: 全部正常")
    
    # 过滤掉关键字段缺失的记录
    before_filter = len(daily)
    daily = daily.dropna(subset=['open', 'high', 'low', 'close'])
    after_filter = len(daily)
    filtered_count = before_filter - after_filter
    
    if filtered_count > 0:
        print(f"\n  已过滤 {filtered_count} 条关键字段缺失的记录")
    
    if daily.empty:
        raise ValueError(
            "❌ 过滤后数据为空！可能原因:\n"
            "   1. OHLC数据缺失过多\n"
            "   2. 日期范围内无有效数据"
        )
    
    print(f"  ✅ 最终有效记录: {len(daily):,} 条")
    
    # 警告：如果数据质量问题较多
    if len(quality_issues) > 3:
        print(f"\n  ⚠️  数据质量警告：")
        for issue in quality_issues[:5]:  # 最多显示5个问题
            print(f"     - {issue}")
        if len(quality_issues) > 5:
            print(f"     ...及其他 {len(quality_issues)-5} 个问题")
        print("     建议检查数据源或缩小日期/股票范围")
    
    # 步骤6: 计算RVI及信号线
    print(f"\n步骤 6: 计算RVI指标")
    print(f"  计算方式: 加权移动平均（权重1:2:2:1）")
    print(f"  RVI周期: {period}, 信号线周期: {signal_period}")
    
    # ============ 辅助函数定义（提取公共逻辑，避免重复） ============
    
    def weighted_ma_4(series):
        """计算4期加权移动平均，权重为(1,2,2,1)/6"""
        if len(series) < 4:
            return np.nan
        return (series.iloc[-4] + 2*series.iloc[-3] + 2*series.iloc[-2] + series.iloc[-1]) / 6
    
    def calculate_stock_rvi(group):
        """对单只股票计算完整的RVI指标"""
        df = group.sort_values('trade_date').copy()
        
        if len(df) < period + signal_period:
            return df[['trade_date']].assign(RVI=np.nan, Signal=np.nan)
        
        # 1. 计算Vigor（价格活力）
        df['Vigor'] = np.where(
            df['high'] != df['low'],
            (df['close'] - df['open']) / (df['high'] - df['low']),
            0.0
        )
        
        # 2. 计算Numerator: Vigor的4期加权MA
        df['Num'] = df['Vigor'].rolling(window=4, min_periods=4).apply(weighted_ma_4, raw=False)
        
        # 3. 计算Denominator: 振幅的4期加权MA
        df['Range'] = df['high'] - df['low']
        df['Den'] = df['Range'].rolling(window=4, min_periods=4).apply(weighted_ma_4, raw=False)
        
        # 4. 计算RVI
        df['RVI'] = np.where(
            (df['Den'].notna()) & (df['Den'] != 0),
            df['Num'] / df['Den'],
            0.0
        )
        
        # 5. 计算Signal线: RVI的4期加权MA
        df['Signal'] = df['RVI'].rolling(window=signal_period, min_periods=signal_period).apply(
            weighted_ma_4, raw=False
        )
        
        return df[['trade_date', 'RVI', 'Signal']]
    
    def add_prev_values(df):
        """添加前一期的RVI和Signal值（避免重复代码）"""
        df['RVI_prev'] = df.groupby('ts_code')['RVI'].shift(1)
        df['Signal_prev'] = df.groupby('ts_code')['Signal'].shift(1)
        return df
    
    def detect_golden_cross(df):
        """检测金叉信号（提取公共逻辑）"""
        return (df['RVI_prev'] <= df['Signal_prev']) & (df['RVI'] > df['Signal'])
    
    def detect_death_cross(df):
        """检测死叉信号（提取公共逻辑）"""
        return (df['RVI_prev'] >= df['Signal_prev']) & (df['RVI'] < df['Signal'])
    
    # ============ RVI计算 ============
    print(f"  开始计算 {daily['ts_code'].nunique()} 只股票的RVI值...")
    try:
        rvi_results = daily.groupby('ts_code', group_keys=False).apply(calculate_stock_rvi)
    except Exception as e:
        raise RuntimeError(
            f"❌ 计算RVI指标时出错: {e}\n"
            f"   可能原因:\n"
            f"   1. 数据量不足（需要至少{period + signal_period}个交易日）\n"
            f"   2. 数据存在异常值\n"
            f"   3. 内存不足"
        )
    
    # 重建索引并合并
    rvi_results = rvi_results.reset_index()
    
    # 检查RVI计算结果
    if rvi_results.empty:
        raise ValueError("❌ RVI计算结果为空，请检查数据质量和参数设置")
    
    daily = pd.merge(
        daily,
        rvi_results[['ts_code', 'trade_date', 'RVI', 'Signal']],
        on=['ts_code', 'trade_date'],
        how='left'
    )
    
    rvi_computed = daily['RVI'].notna().sum()
    rvi_coverage = rvi_computed / len(daily) * 100
    print(f"  ✅ RVI计算完成")
    print(f"     有效记录: {rvi_computed:,} / {len(daily):,} ({rvi_coverage:.1f}%)")
    
    if rvi_coverage < 50:
        print(f"  ⚠️  警告：RVI覆盖率较低({rvi_coverage:.1f}%)，可能导致因子有效性不足")
        print(f"     建议：增加历史数据量或减小period/signal_period参数")
    
    # 步骤7: 根据factor_type生成因子值
    print(f"\n步骤 7: 生成 '{factor_type}' 类型因子值")
    
    if factor_type == 'value':
        # 返回RVI原始值
        daily['factor'] = daily['RVI']
        
    elif factor_type == 'cross':
        # 检测金叉/死叉信号（使用提取的公共函数）
        daily = add_prev_values(daily)
        golden_cross = detect_golden_cross(daily)
        death_cross = detect_death_cross(daily)
        
        daily['factor'] = 0.0
        daily.loc[golden_cross, 'factor'] = 1.0   # 金叉=1
        daily.loc[death_cross, 'factor'] = -1.0   # 死叉=-1
        
        print(f"   金叉信号: {golden_cross.sum():,} 个")
        print(f"   死叉信号: {death_cross.sum():,} 个")
        
    elif factor_type == 'diff':
        # RVI与信号线的差值
        daily['factor'] = daily['RVI'] - daily['Signal']
        
    elif factor_type == 'strength':
        # 交叉强度：交叉时的RVI变化率（使用提取的公共函数）
        daily['RVI_change'] = daily.groupby('ts_code')['RVI'].pct_change()
        daily = add_prev_values(daily)
        
        # 金叉或死叉时刻（复用检测函数）
        is_cross = detect_golden_cross(daily) | detect_death_cross(daily)
        
        daily['factor'] = 0.0
        daily.loc[is_cross, 'factor'] = daily.loc[is_cross, 'RVI_change']
        
    elif factor_type == 'rvi_volume':
        # RVI+成交量组合（使用提取的公共函数）
        # 优先使用turnover_rate，如果没有则使用vol
        if 'turnover_rate' in daily.columns and daily['turnover_rate'].notna().sum() > 0:
            volume_field = 'turnover_rate'
            print(f"   使用字段: turnover_rate（换手率）")
        else:
            volume_field = 'vol'
            print(f"   使用字段: vol（成交量）")
        
        # 计算成交量/换手率均线
        daily['volume_ma'] = daily.groupby('ts_code')[volume_field].transform(
            lambda x: x.rolling(window=volume_ma_period, min_periods=volume_ma_period).mean()
        )
        
        # 检测金叉（复用函数）
        daily = add_prev_values(daily)
        golden_cross = detect_golden_cross(daily)
        
        # 放量确认
        volume_confirm = daily[volume_field] > daily['volume_ma']
        
        # 组合信号
        combined_signal = golden_cross & volume_confirm
        
        # 因子值 = RVI × 放量倍数
        volume_ratio = daily[volume_field] / daily['volume_ma']
        daily['factor'] = 0.0
        daily.loc[combined_signal, 'factor'] = (
            daily.loc[combined_signal, 'RVI'] * 
            volume_ratio.loc[combined_signal]
        )
        
        print(f"   金叉信号: {golden_cross.sum():,} 个")
        print(f"   放量确认: {volume_confirm.sum():,} 个")
        print(f"   组合信号: {combined_signal.sum():,} 个")
        
    elif factor_type == 'rvi_trend':
        # RVI+趋势组合（使用提取的公共函数）
        # 计算价格均线
        daily['price_ma'] = daily.groupby('ts_code')['close'].transform(
            lambda x: x.rolling(window=trend_ma_period, min_periods=trend_ma_period).mean()
        )
        
        # 检测金叉（复用函数）
        daily = add_prev_values(daily)
        golden_cross = detect_golden_cross(daily)
        
        # 趋势确认
        trend_confirm = daily['close'] > daily['price_ma']
        
        # 组合信号
        combined_signal = golden_cross & trend_confirm
        
        # 因子值 = RVI × (1 + 价格相对强度)
        price_strength = (daily['close'] - daily['price_ma']) / daily['price_ma']
        daily['factor'] = 0.0
        daily.loc[combined_signal, 'factor'] = (
            daily.loc[combined_signal, 'RVI'] * 
            (1 + price_strength.loc[combined_signal])
        )
        
        print(f"   金叉信号: {golden_cross.sum():,} 个")
        print(f"   趋势确认: {trend_confirm.sum():,} 个")
        print(f"   组合信号: {combined_signal.sum():,} 个")
        
    else:
        raise ValueError(
            f"不支持的factor_type: {factor_type}，"
            f"支持的类型: 'value', 'cross', 'diff', 'strength', 'rvi_volume', 'rvi_trend'"
        )
    
    # 步骤8: 构建因子数据
    print(f"\n步骤 8: 构建因子DataFrame")
    factor_data = daily[['trade_date', 'ts_code', 'factor']].copy()
    
    # 统计因子有效性
    total_records = len(factor_data)
    valid_records = factor_data['factor'].notna().sum()
    valid_pct = valid_records / total_records * 100 if total_records > 0 else 0
    
    print(f"  总记录数: {total_records:,}")
    print(f"  有效因子: {valid_records:,} ({valid_pct:.1f}%)")
    
    factor_data = factor_data.dropna(subset=['factor'])
    
    if factor_data.empty:
        print("\n" + "="*60)
        print("❌ 错误：因子数据为空")
        print("="*60)
        print("\n可能原因:")
        print(f"  1. 数据量不足")
        print(f"     - 当前时间范围: {daily['trade_date'].min().date()} ~ {daily['trade_date'].max().date()}")
        print(f"     - 建议: 至少提供 {period + signal_period + max(volume_ma_period, trend_ma_period)} 个交易日数据")
        print(f"  2. 参数设置不当")
        print(f"     - 当前参数: period={period}, signal_period={signal_period}")
        print(f"  3. 数据质量问题")
        print(f"     - 建议: 检查OHLC数据完整性")
        if factor_type in ['rvi_volume', 'rvi_trend']:
            print(f"  4. 组合因子条件过严")
            param_name = 'volume_ma_period' if factor_type=='rvi_volume' else 'trend_ma_period'
            param_value = volume_ma_period if factor_type=='rvi_volume' else trend_ma_period
            print(f"     - 当前{param_name}={param_value}")
            print(f"     - 建议: 尝试调整参数或使用'cross'类型观察基础信号")
        print("="*60 + "\n")
        
        # 返回空DataFrame而不是抛出异常
        return pd.DataFrame(columns=['factor']).rename_axis(['trade_date', 'ts_code'])
    
    # 检查因子值的合理性
    factor_stats = factor_data['factor'].describe()
    print(f"\n  因子值统计:")
    print(f"    均值: {factor_stats['mean']:.4f}")
    print(f"    标准差: {factor_stats['std']:.4f}")
    print(f"    最小值: {factor_stats['min']:.4f}")
    print(f"    中位数: {factor_stats['50%']:.4f}")
    print(f"    最大值: {factor_stats['max']:.4f}")
    
    # 检查极端值
    if factor_type in ['value', 'diff', 'rvi_volume', 'rvi_trend']:
        extreme_threshold = 10  # 极端值阈值
        extreme_count = ((factor_data['factor'].abs() > extreme_threshold)).sum()
        if extreme_count > 0:
            extreme_pct = extreme_count / len(factor_data) * 100
            print(f"   ⚠️ 极端值(|factor|>{extreme_threshold}): {extreme_count} 个 ({extreme_pct:.2f}%)")
            if extreme_pct > 5:
                print(f"   警告: 极端值比例过高，可能存在数据异常")
    
    # 步骤9: 设置MultiIndex
    print(f"\n步骤 9: 设置MultiIndex格式")
    try:
        factor = factor_data.set_index(['trade_date', 'ts_code'])
        factor.index.names = ['trade_date', 'ts_code']
        print(f"  ✅ MultiIndex设置成功")
    except Exception as e:
        raise RuntimeError(f"❌ 构建MultiIndex失败: {e}")
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print("✅ RVI因子计算完成")
    print(f"{'='*60}")
    print(f"\n📊 因子配置:")
    print(f"  因子类型: {factor_type}")
    print(f"  RVI周期: {period}")
    print(f"  信号线周期: {signal_period}")
    if factor_type == 'rvi_volume':
        print(f"  成交量MA周期: {volume_ma_period}")
        print(f"\n💡 策略说明:")
        print(f"  金叉 + 成交量放大确认")
        print(f"  因子逻辑: RVI上穿信号线且换手率 > MA{volume_ma_period}")
        print(f"  因子值含义: RVI × 放量倍数（放量越多权重越大）")
    elif factor_type == 'rvi_trend':
        print(f"  趋势MA周期: {trend_ma_period}")
        print(f"\n💡 策略说明:")
        print(f"  金叉 + 价格趋势确认")
        print(f"  因子逻辑: RVI上穿信号线且价格 > MA{trend_ma_period}")
    
    print(f"\n📈 因子覆盖:")
    print(f"  有效记录数: {len(factor):,}")
    print(f"  覆盖股票数: {factor.index.get_level_values('ts_code').nunique()}")
    print(f"  覆盖交易日数: {factor.index.get_level_values('trade_date').nunique()}")
    
    if factor_type == 'cross':
        golden_count = (factor['factor'] == 1.0).sum()
        death_count = (factor['factor'] == -1.0).sum()
        total_signals = golden_count + death_count
        print(f"\n📉 交叉信号统计:")
        golden_pct = golden_count/total_signals*100 if total_signals > 0 else 0
        death_pct = death_count/total_signals*100 if total_signals > 0 else 0
        print(f"  金叉（买入）: {golden_count} 次 ({golden_pct:.1f}%)")
        print(f"  死叉（卖出）: {death_count} 次 ({death_pct:.1f}%)")
        print(f"  总信号数: {total_signals}")
    elif factor_type in ['rvi_volume', 'rvi_trend']:
        positive_signals = (factor['factor'] > 0).sum()
        negative_signals = (factor['factor'] < 0).sum()
        total = len(factor)
        print(f"\n📉 组合信号统计:")
        print(f"  正信号（做多）: {positive_signals} ({positive_signals/total*100:.1f}%)")
        print(f"  负信号（做空）: {negative_signals} ({negative_signals/total*100:.1f}%)")
        print(f"  无信号占比: {(total - positive_signals - negative_signals) / total * 100:.1f}%)")
    elif factor_type in ['value', 'diff']:
        positive = (factor['factor'] > 0).sum()
        negative = (factor['factor'] <= 0).sum()
        total = len(factor)
        factor_stats = factor['factor'].describe()
        print(f"\n📉 因子值分布:")
        print(f"  正值（多头动能）: {positive} ({positive/total*100:.1f}%)")
        print(f"  负值/零值: {negative} ({negative/total*100:.1f}%)")
        print(f"  25%分位数: {factor_stats['25%']:.4f}")
        print(f"  75%分位数: {factor_stats['75%']:.4f}")
    elif factor_type in ['value', 'diff']:
        print(f"\n因子值统计:")
        print(f"   最小值: {factor['factor'].min():.4f}")
        print(f"   25%分位: {factor['factor'].quantile(0.25):.4f}")
        print(f"   中位数: {factor['factor'].median():.4f}")
        print(f"   75%分位: {factor['factor'].quantile(0.75):.4f}")
        print(f"   最大值: {factor['factor'].max():.4f}")
        print(f"   均值: {factor['factor'].mean():.4f}")
        print(f"   标准差: {factor['factor'].std():.4f}")
    
    # 过滤到指定日期范围（移除缓冲期数据）
    print(f"\n步骤 10: 过滤到目标日期范围")
    print(f"  目标范围: {start_date} ~ {end_date}")
    original_count = len(factor)
    factor = factor[factor.index.get_level_values('trade_date') >= pd.to_datetime(start_date)]
    factor = factor[factor.index.get_level_values('trade_date') <= pd.to_datetime(end_date)]
    filtered_count = len(factor)
    removed_count = original_count - filtered_count
    print(f"  过滤前: {original_count:,} 条")
    print(f"  过滤后: {filtered_count:,} 条")
    if removed_count > 0:
        print(f"  移除缓冲期数据: {removed_count:,} 条 ({removed_count/original_count*100:.1f}%)")
    print(f"{'='*60}\n")
    
    return factor

def run_rvi_factor_backtest(
    start_date: str = '2024-01-01',
    end_date: str = '2024-02-29',
    stock_codes: Optional[List[str]] = None,
    period: int = 10,
    signal_period: int = 4,
    factor_type: str = 'cross',
    volume_ma_period: int = 20,
    trend_ma_period: int = 20,
    rebalance_freq: str = 'weekly',
    transaction_cost: float = 0.0003,
    long_direction: str = 'high'
) -> dict:
    """
    使用 BacktestEngine 主路径运行 RVI 因子策略回测，并集成 PerformanceAnalyzer 计算 IC。
    
    **策略说明**：
    根据不同的factor_type采用不同的交易策略：
    - 'cross': 交叉策略 - 金叉做多，死叉做空（或平仓）
    - 'value': 动量策略 - 做多高RVI股票（动能强）
    - 'rvi_volume': 放量突破策略 - 金叉+放量双重确认
    - 'rvi_trend': 顺势策略 - 金叉+趋势向上，顺势而为
    
    **因子特性**：
    - RVI是趋势跟踪指标，IC通常为正
    - 金叉信号适合做多（long_direction='high'）
    - 短周期效果更好（日频、周频调仓）
    - 成交量或趋势过滤可以提高信号质量
    
    Parameters
    ----------
    start_date : str
        回测开始日期，格式 'YYYY-MM-DD'
    end_date : str
        回测结束日期，格式 'YYYY-MM-DD'
    stock_codes : Optional[List[str]]
        股票代码列表，如为 None 则使用所有可用股票
    period : int
        RVI计算周期，默认10（注：实际使用4期加权MA）
    signal_period : int
        信号线周期，默认4
    factor_type : str
        因子类型，决定策略逻辑：
        - 'value': RVI原始值策略
        - 'cross': 金叉/死叉交易信号策略（推荐）
        - 'diff': RVI与信号线差值策略
        - 'strength': 交叉强度策略
        - 'rvi_volume': RVI+成交量组合策略（提高信号质量）
        - 'rvi_trend': RVI+趋势组合策略（顺势交易）
    volume_ma_period : int
        成交量均线周期，默认20天（仅用于'rvi_volume'类型）
    trend_ma_period : int
        趋势均线周期，默认20天（仅用于'rvi_trend'类型）
    rebalance_freq : str
        调仓频率：'daily'（日频）, 'weekly'（周频）, 'monthly'（月频）
        建议：RVI适合周频或更高频率
    transaction_cost : float
        单边交易费用，默认 0.03%
    long_direction : str
        多头方向：'high' 或 'low'
        - 'high': 做多高因子值（推荐，因为RVI是动量指标）
          * cross类型：做多金叉信号
          * value/diff类型：做多高RVI值
        - 'low': 做多低因子值（反向策略，不推荐）
          * cross类型：做多死叉信号
          * value/diff类型：做多低RVI值
        
    Returns
    -------
    dict
        包含以下键值的回测结果字典：
        - 'factor_data': pd.DataFrame
            因子数据，MultiIndex (trade_date, ts_code)
        - 'portfolio_returns': pd.DataFrame
            组合收益率时间序列，包含 'Long_Only' 列
        - 'performance_metrics': dict
            业绩指标字典：
            * 'total_return': 总收益率
            * 'annualized_return': 年化收益率
            * 'volatility': 年化波动率
            * 'sharpe_ratio': 夏普比率
            * 'max_drawdown': 最大回撤
        - 'analysis_results': dict
            分析结果字典：
            * 'metrics': 完整业绩指标DataFrame
            * 'ic_series': IC时间序列（因子与收益相关性）
    
    Examples
    --------
    >>> # 基础交叉策略回测
    >>> results = run_rvi_factor_backtest(
    ...     start_date='2020-01-01',
    ...     end_date='2023-12-31',
    ...     factor_type='cross',
    ...     rebalance_freq='weekly'
    ... )
    >>> print(f"夏普比率: {results['performance_metrics']['sharpe_ratio']:.2f}")
    >>> 
    >>> # 成交量确认策略
    >>> results = run_rvi_factor_backtest(
    ...     start_date='2020-01-01',
    ...     end_date='2023-12-31',
    ...     factor_type='rvi_volume',
    ...     volume_ma_period=20,
    ...     rebalance_freq='weekly'
    ... )
    
    Notes
    -----
    - 使用 BacktestEngine 进行回测，自动处理调仓、交易成本等
    - IC分析基于Spearman相关系数，衡量因子预测能力
    - 对于'cross'类型，金叉数量较少时可能导致持仓不足
    - 建议先用短时间段测试参数，再进行全样本回测
    
    See Also
    --------
    calculate_rvi_factor : RVI因子计算函数
    """
    print("\n" + "=" * 60)
    print(f"RVI因子回测 - {factor_type}模式")
    print("=" * 60)
    
    # 参数验证
    try:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        if start_dt >= end_dt:
            raise ValueError(f"开始日期必须早于结束日期")
    except Exception as e:
        raise ValueError(f"日期格式错误: {e}")
    
    if rebalance_freq not in ['daily', 'weekly', 'monthly']:
        raise ValueError(f"不支持的调仓频率: {rebalance_freq}，支持: daily, weekly, monthly")
    
    if transaction_cost < 0 or transaction_cost > 0.1:
        raise ValueError(f"交易成本异常: {transaction_cost}，应在0-0.1之间")
    
    if long_direction not in ['high', 'low']:
        raise ValueError(f"不支持的做多方向: {long_direction}，支持: high, low")
    
    # 初始化数据管理器
    try:
        data_manager = DataManager()
    except Exception as e:
        raise RuntimeError(f"初始化DataManager失败: {e}")
    
    # 计算因子
    print(f"\n计算RVI因子 (周期={period}, 信号线={signal_period})...")
    try:
        factor_data = calculate_rvi_factor(
            data_manager=data_manager,
            start_date=start_date,
            end_date=end_date,
            stock_codes=stock_codes,
            period=period,
            signal_period=signal_period,
            factor_type=factor_type,
            volume_ma_period=volume_ma_period,
            trend_ma_period=trend_ma_period
        )
    except Exception as e:
        print(f"\n❌ 因子计算失败: {e}")
        raise
    
    if factor_data.empty:
        print("\n⚠️ 因子数据为空，无法回测")
        print("请检查:")
        print("1. 日期范围是否有足够的数据")
        print("2. 股票代码是否正确")
        print("3. factor_type参数是否合理")
        return {
            'factor_data': None,
            'portfolio_returns': None,
            'performance_metrics': {},
            'analysis_results': {}
        }

    # 使用 BacktestEngine
    from backtest_engine.engine import BacktestEngine
    
    print("\n初始化回测引擎...")
    try:
        engine = BacktestEngine(
            data_manager=data_manager,
            fee=transaction_cost,
            long_direction=long_direction,
            rebalance_freq=rebalance_freq,
            factor_name='factor',
        )
    except Exception as e:
        raise RuntimeError(f"初始化BacktestEngine失败: {e}")
    
    # 设置因子数据
    engine.factor_data = factor_data
    
    # 准备收益率数据
    print("准备收益率数据...")
    stock_list = factor_data.index.get_level_values('ts_code').unique().tolist()
    
    try:
        stock_data = data_manager.load_data(
            'daily',
            start_date=start_date,
            end_date=end_date,
            stock_codes=stock_list
        )
    except Exception as e:
        raise RuntimeError(f"加载回测数据失败: {e}")
    
    if stock_data is None or stock_data.empty:
        raise ValueError("无法加载回测所需的股票数据")
    
    # 计算次日收益率
    stock_data = stock_data.sort_values(['ts_code', 'trade_date'])
    stock_data['next_return'] = stock_data.groupby('ts_code')['close'].pct_change().shift(-1)
    
    # 合并因子和收益率
    factor_reset = factor_data.reset_index()
    stock_subset = stock_data[['ts_code', 'trade_date', 'next_return']].copy()
    
    engine.combined_data = pd.merge(
        factor_reset,
        stock_subset,
        on=['ts_code', 'trade_date'],
        how='inner'
    )
    engine.combined_data.dropna(subset=['factor', 'next_return'], inplace=True)
    
    if engine.combined_data.empty:
        raise ValueError(
            "合并因子和收益率数据后为空\n"
            "可能原因:\n"
            "1. 因子日期和收益率日期不匹配\n"
            "2. 股票代码不匹配\n"
            "3. next_return全部为NaN"
        )
    
    print(f"   有效数据: {len(engine.combined_data):,} 条")
    
    # 运行回测
    print("\n运行回测...")
    try:
        portfolio_returns = engine.run()
    except Exception as e:
        raise RuntimeError(f"回测执行失败: {e}")
    
    # 获取调仓日期（用于统计调仓次数）
    rebalance_dates = engine._get_rebalance_dates()
    rebalance_count = len(rebalance_dates)
    print(f"✅ 回测完成，共执行 {rebalance_count} 次调仓")
    
    # 计算业绩指标
    if not isinstance(portfolio_returns, pd.DataFrame) or 'Long_Only' not in portfolio_returns.columns:
        print("⚠️ 回测结果格式异常")
        return {'factor_data': factor_data, 'portfolio_returns': portfolio_returns}
    
    series = portfolio_returns['Long_Only']
    cum = (1 + series).cumprod()
    total_return = float(cum.iloc[-1] - 1) if len(cum) else np.nan
    trading_days = len(series)
    annualized_return = float(cum.iloc[-1] ** (252 / trading_days) - 1) if trading_days > 0 else np.nan
    volatility = float(series.std() * np.sqrt(252))
    sharpe_ratio = float(annualized_return / volatility) if volatility > 0 and not np.isnan(annualized_return) else 0.0
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    max_drawdown = float(drawdown.min()) if not drawdown.empty else np.nan
    
    # 获取性能分析
    analyzer = engine.get_performance_analysis()
    metrics_df = analyzer.calculate_metrics()
    ic_series = analyzer.ic_series
    
    # 打印结果
    print("\n" + "=" * 60)
    print("回测结果 (Long_Only)")
    print("=" * 60)
    print(f"总收益率: {total_return:.2%}")
    print(f"年化收益率: {annualized_return:.2%}")
    print(f"年化波动率: {volatility:.2%}")
    print(f"夏普比率: {sharpe_ratio:.3f}")
    print(f"最大回撤: {max_drawdown:.2%}")
    print(f"\n调仓统计:")
    print(f"  调仓次数: {rebalance_count}")
    print(f"  调仓频率: {rebalance_freq}")
    print(f"  平均持仓天数: {len(series) / rebalance_count:.1f}" if rebalance_count > 0 else "  平均持仓天数: N/A")
    
    if ic_series is not None and len(ic_series) > 0:
        print(f"\nIC分析:")
        print(f"  IC均值: {ic_series.mean():.4f}")
        print(f"  IC标准差: {ic_series.std():.4f}")
        print(f"  ICIR: {ic_series.mean() / ic_series.std():.4f}" if ic_series.std() > 0 else "  ICIR: N/A")
        print(f"  IC>0占比: {(ic_series > 0).mean():.2%}")
    
    return {
        'factor_data': factor_data,
        'portfolio_returns': portfolio_returns,
        'performance_metrics': {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'rebalance_count': rebalance_count,
            'rebalance_freq': rebalance_freq,
            'trading_days': trading_days,
        },
        'analysis_results': {
            'metrics': metrics_df,
            'ic_series': ic_series
        }
    }
    
def main():
    """
    主函数：演示RVI因子的多种策略回测及对比分析
    
    **演示内容**：
    1. RVI + 成交量组合因子：金叉信号 + 放量确认
    2. RVI + 趋势组合因子：金叉信号 + 价格在均线上方
    3. 纯RVI交叉因子：仅使用金叉/死叉信号（作为基准）
    
    **策略对比**：
    - 展示三种策略的夏普比率、年化收益、最大回撤
    - 评估成交量和趋势过滤对信号质量的影响
    - 帮助选择最适合当前市场环境的策略
    
    **配置说明**：
    - 回测周期：2015-01-01 至 2025-09-30（约10年）
    - 调仓频率：周频（weekly）
    - 交易成本：单边0.03%
    - 做多方向：high（做多金叉或高因子值）
    
    **预期结果**：
    - 组合因子（成交量/趋势过滤）通常表现更稳定
    - 纯交叉信号在震荡市中可能频繁交易
    - 不同市场环境下，最优策略可能不同
    
    **因子类型说明**：
    1. 'value'      - RVI原始值（连续动量指标）
    2. 'cross'      - 金叉/死叉信号（离散交易信号）
    3. 'diff'       - RVI与信号线差值（偏离度指标）
    4. 'strength'   - 交叉强度（突破力度指标）
    5. 'rvi_volume' - RVI+成交量组合（双重确认）✨
    6. 'rvi_trend'  - RVI+趋势组合（顺势交易）✨
    
    Notes
    -----
    - 可修改config参数测试不同配置
    - 建议先用短周期测试，确认参数后再全样本回测
    - 成交量和趋势过滤可以减少假信号，但可能降低信号数量
    - IC分析可以评估因子的预测能力
    
    Raises
    ------
    Exception
        如果数据加载失败或回测过程出错
    """
    print("=" * 60)
    print("RVI组合因子策略演示")
    print("=" * 60)

    try:
        # 演示1: RVI+成交量组合因子
        print("\n" + "=" * 60)
        print("【演示1】RVI + 成交量组合因子")
        print("=" * 60)
        print("策略逻辑: RVI金叉 + 成交量放大")
        
        config_volume = {
            'start_date': '2015-01-01',
            'end_date': '2025-09-30',
            'period': 10,
            'signal_period': 4,
            'factor_type': 'rvi_volume',
            'volume_ma_period': 20,
            'rebalance_freq': 'weekly',
            'transaction_cost': 0.0003,
            'long_direction': 'high',
        }

        print("\n配置参数:")
        for key, value in config_volume.items():
            print(f"  {key}: {value}")

        try:
            results_volume = run_rvi_factor_backtest(**config_volume)
        except Exception as e:
            print(f"\n❌ 演示1执行失败: {e}")
            results_volume = {'performance_metrics': None}

        if results_volume.get('performance_metrics'):
            print("\n策略表现:")
            metrics = results_volume['performance_metrics']
            print(f"  夏普比率: {metrics['sharpe_ratio']:.3f}")
            print(f"  总收益: {metrics['total_return']:.2%}")
            print(f"  年化收益: {metrics['annualized_return']:.2%}")
            print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
            print(f"  调仓次数: {metrics['rebalance_count']}")
        else:
            print("\n⚠️ 未能获取业绩指标")

        # 演示2: RVI+趋势组合因子
        print("\n" + "=" * 60)
        print("【演示2】RVI + 趋势组合因子")
        print("=" * 60)
        print("策略逻辑: RVI金叉 + 价格在均线上方")
        
        config_trend = {
            'start_date': '2015-01-01',
            'end_date': '2025-09-30',
            'period': 10,
            'signal_period': 4,
            'factor_type': 'rvi_trend',
            'trend_ma_period': 20,
            'rebalance_freq': 'weekly',
            'transaction_cost': 0.0003,
            'long_direction': 'high',
        }

        print("\n配置参数:")
        for key, value in config_trend.items():
            print(f"  {key}: {value}")

        try:
            results_trend = run_rvi_factor_backtest(**config_trend)
        except Exception as e:
            print(f"\n❌ 演示2执行失败: {e}")
            results_trend = {'performance_metrics': None}

        if results_trend.get('performance_metrics'):
            print("\n策略表现:")
            metrics = results_trend['performance_metrics']
            print(f"  夏普比率: {metrics['sharpe_ratio']:.3f}")
            print(f"  总收益: {metrics['total_return']:.2%}")
            print(f"  年化收益: {metrics['annualized_return']:.2%}")
            print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
            print(f"  调仓次数: {metrics['rebalance_count']}")
        else:
            print("\n⚠️ 未能获取业绩指标")

        # 演示3: 纯RVI交叉因子（对比）
        print("\n" + "=" * 60)
        print("【演示3】纯RVI交叉因子（对比基准）")
        print("=" * 60)
        print("策略逻辑: 仅使用RVI金叉信号")
        
        config_cross = {
            'start_date': '2015-01-01',
            'end_date': '2025-09-30',
            'period': 10,
            'signal_period': 4,
            'factor_type': 'cross',
            'rebalance_freq': 'weekly',
            'transaction_cost': 0.0003,
            'long_direction': 'high',
        }

        try:
            results_cross = run_rvi_factor_backtest(**config_cross)
        except Exception as e:
            print(f"\n❌ 演示3执行失败: {e}")
            results_cross = {'performance_metrics': None}

        if results_cross.get('performance_metrics'):
            print("\n策略表现:")
            metrics = results_cross['performance_metrics']
            print(f"  夏普比率: {metrics['sharpe_ratio']:.3f}")
            print(f"  总收益: {metrics['total_return']:.2%}")
            print(f"  年化收益: {metrics['annualized_return']:.2%}")
            print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
            print(f"  调仓次数: {metrics['rebalance_count']}")
        else:
            print("\n⚠️ 未能获取业绩指标")

        # 对比总结
        print("\n" + "=" * 60)
        print("策略对比总结")
        print("=" * 60)
        
        if all([results_volume.get('performance_metrics'), 
                results_trend.get('performance_metrics'), 
                results_cross.get('performance_metrics')]):
            
            print(f"\n{'策略类型':<20} {'夏普比率':>10} {'年化收益':>10} {'最大回撤':>10} {'调仓次数':>10}")
            print("-" * 70)
            
            m1 = results_volume['performance_metrics']
            print(f"{'RVI+成交量':<20} {m1['sharpe_ratio']:>10.3f} {m1['annualized_return']:>9.2%} {m1['max_drawdown']:>9.2%} {m1['rebalance_count']:>10}")
            
            m2 = results_trend['performance_metrics']
            print(f"{'RVI+趋势':<20} {m2['sharpe_ratio']:>10.3f} {m2['annualized_return']:>9.2%} {m2['max_drawdown']:>9.2%} {m2['rebalance_count']:>10}")
            
            m3 = results_cross['performance_metrics']
            print(f"{'纯RVI交叉':<20} {m3['sharpe_ratio']:>10.3f} {m3['annualized_return']:>9.2%} {m3['max_drawdown']:>9.2%} {m3['rebalance_count']:>10}")

        print("\n" + "=" * 60)
        print("所有因子类型:")
        print("=" * 60)
        print("1. 'value'      - RVI原始值")
        print("2. 'cross'      - 金叉/死叉信号")
        print("3. 'diff'       - RVI与信号线差值")
        print("4. 'strength'   - 交叉强度")
        print("5. 'rvi_volume' - RVI+成交量组合 ✨")
        print("6. 'rvi_trend'  - RVI+趋势组合 ✨")
        print("\n✅ RVI组合因子策略演示完成!")

    except Exception as e:
        print(f"\n❌ 演示运行失败: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
