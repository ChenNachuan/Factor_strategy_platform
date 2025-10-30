"""
OBV (On-Balance Volume) 增强版因子

优化内容（参考 pe_factor.py, rsi_factor.py, new_high_alpha_factor.py）：
=================================================================

1. 数据质量筛选 ✅
   - 过滤 ST 股票（通过涨跌幅限制判断）
   - 过滤停牌数据（成交量/成交额为0）
   - 过滤涨跌停数据（±9.8%）
   - 过滤低流动性股票（换手率过低）
   - 过滤低价股（价格 < 1元）
   - 过滤异常涨跌幅（|涨跌幅| > 30%）
   - 成交量异常处理：
     * 零成交量数据
     * 异常放量（>均值+10σ，限制到均值+5σ）
     * 异常缩量（<均值1%）
     * 量价一致性检查（成交额/成交量 vs 价格）

2. OBV 计算优化 ✅
   - 使用百分比变化而非绝对变化（避免价格量级影响）
   - 设置变化阈值过滤噪音（0.01%）
   - 价格平稳时延续前一日方向
   - 从0开始累积（标准做法）
   - 数据质量检查（NaN值、异常变化）
   - 成交量验证：
     * 确保成交量为正
     * 检查数据覆盖率（>50%）
     * 限制单日极端OBV变化（<50倍中位数）
     * 成交量稳定性监控

3. 异常值处理 ✅
   - 使用 MAD (Median Absolute Deviation) 方法（比标准差更稳健）
   - 三层防护：Winsorize → 标准化 → 截尾
   - 按日期分组处理，避免时序偏差
   - 实时监控：偏度、峰度、缺失率

4. 数值稳定性 ✅
   - 所有除法操作添加最小值保护（1e-10）
   - 标准化前检查样本数（≥10）和标准差（≥1e-8）
   - 自动检测并替换无穷值/NaN
   - 百分比变化限制在合理范围（±1000%）
   - 安全标准化函数（safe_standardize）

5. 输出增强 ✅
   - 详细的数据筛选统计
   - OBV 计算质量检查
   - 子因子贡献分析
   - 数据质量评估报告

6. 回测空值保护 ✅
   - 因子计算异常捕获
   - 数据加载验证（None/Empty检查）
   - 列存在性验证
   - 有效值比例检查
   - 合并操作异常处理
   - 业绩指标安全计算
   - IC 分析异常保护
   - 完整的错误返回机制

作者：参考业界最佳实践
版本：v2.0 (Enhanced)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Optional, List

# 路径：把项目根目录加入 sys.path，便于使用绝对包导入
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_manager.data import DataManager


def calculate_obv_factor(
    data_manager: DataManager,
    start_date: str,
    end_date: str,
    stock_codes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    计算基础 OBV (On-Balance Volume) 因子，并进行标准化处理。
    
    Returns
    -------
    DataFrame
        MultiIndex (trade_date, ts_code) with single column 'factor'.
    """
    # 股票池
    if stock_codes is None:
        all_daily = data_manager.load_data('daily', start_date=start_date, end_date=end_date, cleaned=True)
        if all_daily is None or all_daily.empty:
            stock_codes = ['000001.SZ', '000002.SZ', '000858.SZ', '600000.SH', '600036.SH', '600519.SH']
        else:
            stock_codes = all_daily['ts_code'].unique().tolist()

    # 日线数据
    daily = data_manager.load_data('daily', start_date=start_date, end_date=end_date, stock_codes=stock_codes)
    if daily is None or daily.empty:
        raise ValueError('无法获取日行情数据')
    
    # 统一日期为 datetime 并排序
    daily = daily.copy()
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], errors='coerce')
    if daily['trade_date'].isna().any():
        daily['trade_date'] = pd.to_datetime(daily['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
    daily = daily.dropna(subset=['trade_date'])
    daily = daily.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

    # 计算 OBV
    result_parts = []
    for code in daily['ts_code'].unique():
        stock_data = daily[daily['ts_code'] == code].sort_values('trade_date').copy()
        
        # === 优化的 OBV 计算（与增强版保持一致）===
        stock_data['prev_close'] = stock_data['close'].shift(1)
        
        # 使用百分比变化判断方向
        price_change_pct = (stock_data['close'] - stock_data['prev_close']) / stock_data['prev_close']
        price_threshold = 0.0001  # 0.01% 的变化阈值
        
        def determine_direction(change_pct, prev_direction):
            if pd.isna(change_pct):
                return 0
            elif change_pct > price_threshold:
                return 1
            elif change_pct < -price_threshold:
                return -1
            else:
                return prev_direction if not pd.isna(prev_direction) else 0
        
        # 计算方向
        directions = [0]
        for i in range(1, len(stock_data)):
            direction = determine_direction(
                price_change_pct.iloc[i], 
                directions[-1]
            )
            directions.append(direction)
        
        stock_data['direction'] = directions
        
        # 计算 OBV
        obv = (stock_data['direction'] * stock_data['vol']).cumsum()
        
        # 创建结果 DataFrame
        result = pd.DataFrame({
            'trade_date': stock_data['trade_date'],
            'ts_code': code,
            'factor': obv
        })
        result_parts.append(result)
    
    # 合并所有股票的结果
    combined = pd.concat(result_parts, axis=0)
    
    # 对每个交易日的 OBV 进行截面标准化
    combined = combined.set_index(['trade_date', 'ts_code'])
    combined = combined.sort_index()
    
    # 按日期分组进行标准化
    grouped = combined.groupby('trade_date')
    combined['factor'] = grouped['factor'].transform(lambda x: (x - x.mean()) / x.std())
    
    return combined[['factor']]


def calculate_obv_advanced_factor(
    data_manager: DataManager,
    start_date: str,
    end_date: str,
    stock_codes: Optional[List[str]] = None,
    trend_period: int = 20,
    divergence_period: int = 20,
    rank_period: int = 120,
    # 数据质量筛选参数
    filter_st: bool = True,
    filter_suspend: bool = True,
    filter_limit: bool = True,
    min_turnover_rate: float = 0.01,
    min_price: float = 1.0,
) -> pd.DataFrame:
    """
    计算增强版 OBV 综合因子，包括趋势类和价格背离因子。
    
    **因子组成**：
    1. OBV 趋势斜率因子：线性回归斜率，衡量资金流入/流出速度
    2. OBV 变化率因子：OBV 的百分比变化，衡量累积量能增长
    3. OBV 相对强度因子：OBV 在历史区间的分位数排名
    4. 量价背离因子：OBV 趋势与价格趋势的差异度
    5. OBV 突破因子：OBV 突破历史高点的强度
    
    **选股逻辑**：
    - 高因子值 = 资金持续流入 + 量价配合良好 + OBV 突破
    - 适合捕捉主力建仓和趋势启动信号
    
    **数据质量筛选**（参考 new_high_alpha_factor.py）：
    - 过滤 ST/ST* 股票（高风险）
    - 过滤停牌日数据（无交易）
    - 过滤涨跌停日数据（无法正常交易）
    - 过滤低流动性股票（换手率过低）
    - 过滤低价股（价格过低，易操纵）
    
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
    trend_period : int
        趋势计算周期，默认 20 天
    divergence_period : int
        背离计算周期，默认 20 天
    rank_period : int
        相对强度计算周期，默认 120 天
    filter_st : bool
        是否过滤 ST 股票，默认 True
    filter_suspend : bool
        是否过滤停牌数据，默认 True
    filter_limit : bool
        是否过滤涨跌停数据，默认 True
    min_turnover_rate : float
        最小换手率阈值（%），默认 0.01%（过滤极低流动性）
    min_price : float
        最小价格阈值（元），默认 1.0 元（过滤低价股）
    
    Returns
    -------
    DataFrame
        MultiIndex (trade_date, ts_code) with columns:
        - factor: 综合因子（加权平均）
        - obv_slope: OBV 斜率因子
        - obv_change: OBV 变化率因子
        - obv_rank: OBV 相对强度因子
        - obv_divergence: 量价背离因子
        - obv_breakthrough: OBV 突破因子
    """
    print(f"\n{'='*60}")
    print("OBV 增强版综合因子计算")
    print(f"{'='*60}")
    
    # 股票池
    if stock_codes is None:
        print("未指定股票池，使用全市场股票...")
        all_daily = data_manager.load_data('daily', start_date=start_date, end_date=end_date, cleaned=True)
        if all_daily is None or all_daily.empty:
            stock_codes = ['000001.SZ', '000002.SZ', '000858.SZ', '600000.SH', '600036.SH', '600519.SH']
        else:
            stock_codes = all_daily['ts_code'].unique().tolist()
        print(f"✅ 股票池: {len(stock_codes)} 只股票")
    else:
        print(f"✅ 使用指定股票池: {len(stock_codes)} 只股票")
    
    # 加载日线数据（需要足够的历史数据）
    buffer_days = max(trend_period, divergence_period, rank_period) * 3
    start_date_extended = (pd.to_datetime(start_date) - pd.Timedelta(days=buffer_days)).strftime('%Y-%m-%d')
    
    daily = data_manager.load_data('daily', start_date=start_date_extended, end_date=end_date, stock_codes=stock_codes)
    if daily is None or daily.empty:
        raise ValueError('无法获取日行情数据')
    
    # 统一日期为 datetime 并排序
    daily = daily.copy()
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], errors='coerce')
    if daily['trade_date'].isna().any():
        daily['trade_date'] = pd.to_datetime(daily['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
    daily = daily.dropna(subset=['trade_date'])
    daily = daily.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    
    print(f"✅ 成功加载数据")
    print(f"   数据时间范围: {daily['trade_date'].min()} ~ {daily['trade_date'].max()}")
    print(f"   原始数据量: {len(daily):,} 条记录")
    
    # === 数据质量筛选 ===
    print(f"\n步骤 1: 数据质量筛选...")
    original_count = len(daily)
    
    # 1.1 过滤 ST 股票
    if filter_st:
        # ST 股票通常在股票名称中包含 ST/ST*/退市等标识
        # 或者通过 ts_code 判断（如果有 name 字段更准确）
        # 简化处理：通过涨跌幅限制判断（ST 股票涨跌幅限制为 ±5%）
        st_mask = daily['pct_chg'].abs() <= 5.5  # 留出容错空间
        # 更严格的判断：连续多日涨跌幅都在 ±5% 以内的可能是 ST
        daily['is_likely_st'] = daily.groupby('ts_code')['pct_chg'].transform(
            lambda x: (x.abs() <= 5.5).rolling(5).mean() > 0.8
        )
        st_count = daily['is_likely_st'].sum()
        daily = daily[~daily['is_likely_st']].copy()
        print(f"   ✓ 过滤 ST 股票: 剔除 {st_count:,} 条记录")
    
    # 1.2 过滤停牌数据
    if filter_suspend:
        # 停牌判断：成交量为 0 或接近 0
        suspend_mask = (daily['vol'] <= 0) | (daily['amount'] <= 0)
        suspend_count = suspend_mask.sum()
        daily = daily[~suspend_mask].copy()
        print(f"   ✓ 过滤停牌数据: 剔除 {suspend_count:,} 条记录")
    
    # 1.3 过滤涨跌停数据
    if filter_limit:
        # 涨停：涨幅 > 9.8%（科创板/创业板为 19.8%）
        # 跌停：跌幅 < -9.8%
        # 为简化，统一使用 9.8% 阈值
        limit_up_mask = daily['pct_chg'] > 9.8
        limit_down_mask = daily['pct_chg'] < -9.8
        limit_count = (limit_up_mask | limit_down_mask).sum()
        daily = daily[~(limit_up_mask | limit_down_mask)].copy()
        print(f"   ✓ 过滤涨跌停数据: 剔除 {limit_count:,} 条记录")
    
    # 1.4 加载换手率和流动性数据（增强版）
    # 需要从 daily_basic 获取
    print(f"\n   === 流动性筛选（增强版）===")
    
    try:
        daily_basic = data_manager.load_data(
            'daily_basic', 
            start_date=start_date_extended, 
            end_date=end_date, 
            stock_codes=stock_codes
        )
        
        if daily_basic is not None and not daily_basic.empty:
            daily_basic = daily_basic.copy()
            daily_basic['trade_date'] = pd.to_datetime(daily_basic['trade_date'])
            
            # 合并换手率和市值数据
            daily = pd.merge(
                daily,
                daily_basic[['ts_code', 'trade_date', 'turnover_rate', 'turnover_rate_f', 'volume_ratio', 'float_share', 'total_mv']],
                on=['ts_code', 'trade_date'],
                how='left'
            )
            
            print(f"   ✓ 成功加载流动性数据")
            print(f"      - turnover_rate: 换手率（%）")
            print(f"      - turnover_rate_f: 换手率（自由流通股）")
            print(f"      - volume_ratio: 量比")
            print(f"      - float_share: 流通股本（万股）")
            print(f"      - total_mv: 总市值（万元）")
            
            # 1.5 多维度流动性筛选
            print(f"\n   多维度流动性筛选:")
            
            # 方法1: 换手率筛选（基础）
            if 'turnover_rate' in daily.columns:
                low_turnover_mask = daily['turnover_rate'] < min_turnover_rate
                low_turnover_count = low_turnover_mask.sum()
                daily = daily[~low_turnover_mask].copy()
                print(f"      ✓ 换手率过滤（<{min_turnover_rate}%）: 剔除 {low_turnover_count:,} 条")
            
            # 方法2: 自由流通换手率筛选（更严格）
            if 'turnover_rate_f' in daily.columns:
                low_float_turnover_mask = (
                    daily['turnover_rate_f'].notna() & 
                    (daily['turnover_rate_f'] < min_turnover_rate * 1.2)  # 自由流通换手率要求更高
                )
                low_float_turnover_count = low_float_turnover_mask.sum()
                daily = daily[~low_float_turnover_mask].copy()
                print(f"      ✓ 自由流通换手率过滤（<{min_turnover_rate*1.2}%）: 剔除 {low_float_turnover_count:,} 条")
            
            # 方法3: 连续低换手率筛选（识别持续低流动性）
            if 'turnover_rate' in daily.columns:
                daily['avg_turnover_5d'] = daily.groupby('ts_code')['turnover_rate'].transform(
                    lambda x: x.rolling(5, min_periods=3).mean()
                )
                persistent_low_turnover_mask = daily['avg_turnover_5d'] < min_turnover_rate * 0.5
                persistent_low_count = persistent_low_turnover_mask.sum()
                daily = daily[~persistent_low_turnover_mask].copy()
                print(f"      ✓ 持续低换手率过滤（5日均值<{min_turnover_rate*0.5}%）: 剔除 {persistent_low_count:,} 条")
                daily = daily.drop(columns=['avg_turnover_5d'], errors='ignore')
            
            # 方法4: 成交金额筛选（绝对流动性）
            # 计算日均成交金额（万元）
            daily['amount_wan'] = daily['amount']  # amount 已经是万元单位
            
            # 按股票分组计算20日平均成交金额
            daily['avg_amount_20d'] = daily.groupby('ts_code')['amount_wan'].transform(
                lambda x: x.rolling(20, min_periods=10).mean()
            )
            
            # 过滤日均成交金额过低的股票（< 1000万元）
            min_daily_amount = 1000  # 万元
            low_amount_mask = (
                daily['avg_amount_20d'].notna() & 
                (daily['avg_amount_20d'] < min_daily_amount)
            )
            low_amount_count = low_amount_mask.sum()
            daily = daily[~low_amount_mask].copy()
            print(f"      ✓ 成交金额过滤（20日均值<{min_daily_amount}万元）: 剔除 {low_amount_count:,} 条")
            
            # 方法5: 流通市值筛选（避免超小盘股）
            if 'float_share' in daily.columns and 'close' in daily.columns:
                # 计算流通市值（万元）= 流通股本（万股）× 收盘价
                daily['float_mv'] = daily['float_share'] * daily['close']
                
                # 过滤流通市值过小的股票（< 5亿元 = 50000万元）
                min_float_mv = 50000  # 万元（5亿）
                small_float_mask = (
                    daily['float_mv'].notna() & 
                    (daily['float_mv'] < min_float_mv)
                )
                small_float_count = small_float_mask.sum()
                daily = daily[~small_float_mask].copy()
                print(f"      ✓ 流通市值过滤（<{min_float_mv/10000:.1f}亿元）: 剔除 {small_float_count:,} 条")
                
                daily = daily.drop(columns=['float_mv'], errors='ignore')
            
            # 方法6: 换手率稳定性筛选（避免流动性不稳定的股票）
            if 'turnover_rate' in daily.columns:
                daily['turnover_std_20d'] = daily.groupby('ts_code')['turnover_rate'].transform(
                    lambda x: x.rolling(20, min_periods=10).std()
                )
                daily['turnover_mean_20d'] = daily.groupby('ts_code')['turnover_rate'].transform(
                    lambda x: x.rolling(20, min_periods=10).mean()
                )
                
                # 变异系数 = 标准差 / 均值
                # 过滤变异系数过大的股票（流动性不稳定）
                daily['turnover_cv'] = np.where(
                    daily['turnover_mean_20d'] > 0,
                    daily['turnover_std_20d'] / daily['turnover_mean_20d'],
                    np.nan
                )
                
                unstable_turnover_mask = (
                    daily['turnover_cv'].notna() & 
                    (daily['turnover_cv'] > 5)  # 变异系数 > 5 视为不稳定
                )
                unstable_count = unstable_turnover_mask.sum()
                daily = daily[~unstable_turnover_mask].copy()
                print(f"      ✓ 换手率稳定性过滤（变异系数>5）: 剔除 {unstable_count:,} 条")
                
                daily = daily.drop(columns=['turnover_std_20d', 'turnover_mean_20d', 'turnover_cv'], errors='ignore')
            
            # 方法7: 量比异常筛选（识别异常交易日）
            if 'volume_ratio' in daily.columns:
                # 量比过高（>10）或过低（<0.1）都可能有问题
                abnormal_volume_ratio_mask = (
                    daily['volume_ratio'].notna() & 
                    ((daily['volume_ratio'] > 10) | (daily['volume_ratio'] < 0.1))
                )
                abnormal_vr_count = abnormal_volume_ratio_mask.sum()
                daily = daily[~abnormal_volume_ratio_mask].copy()
                print(f"      ✓ 量比异常过滤（<0.1 或 >10）: 剔除 {abnormal_vr_count:,} 条")
            
            # 清理临时列
            daily = daily.drop(columns=['amount_wan', 'avg_amount_20d'], errors='ignore')
            
            # 流动性筛选总结
            print(f"\n   📊 流动性筛选总结:")
            remaining_stocks = daily['ts_code'].nunique()
            remaining_records = len(daily)
            print(f"      剩余股票数: {remaining_stocks}")
            print(f"      剩余记录数: {remaining_records:,}")
            
        else:
            print(f"   ⚠️  警告: 无法加载 daily_basic 数据，跳过流动性筛选")
            print(f"   将使用简化的流动性筛选（基于成交量）")
            
            # 简化的流动性筛选（仅基于成交量）
            # 计算20日平均成交量
            daily['avg_vol_20d'] = daily.groupby('ts_code')['vol'].transform(
                lambda x: x.rolling(20, min_periods=10).mean()
            )
            
            # 过滤平均成交量过低的股票
            vol_median = daily['avg_vol_20d'].median()
            low_vol_mask = (
                daily['avg_vol_20d'].notna() & 
                (daily['avg_vol_20d'] < vol_median * 0.1)  # 低于中位数的10%
            )
            low_vol_count = low_vol_mask.sum()
            daily = daily[~low_vol_mask].copy()
            print(f"   ✓ 低成交量过滤（<中位数10%）: 剔除 {low_vol_count:,} 条记录")
            
            daily = daily.drop(columns=['avg_vol_20d'], errors='ignore')
            
    except Exception as e:
        print(f"   ⚠️  警告: 流动性筛选失败 ({e})，跳过此步骤")
    
    # 1.6 过滤低价股
    low_price_mask = daily['close'] < min_price
    low_price_count = low_price_mask.sum()
    daily = daily[~low_price_mask].copy()
    print(f"   ✓ 过滤低价股（价格<{min_price}元）: 剔除 {low_price_count:,} 条记录")
    
    # 1.7 过滤异常涨跌幅数据
    extreme_pct_mask = daily['pct_chg'].abs() > 30  # 单日涨跌幅超过 30% 视为异常
    extreme_pct_count = extreme_pct_mask.sum()
    daily = daily[~extreme_pct_mask].copy()
    print(f"   ✓ 过滤异常涨跌幅数据（|涨跌幅|>30%）: 剔除 {extreme_pct_count:,} 条记录")
    
    # === 成交量异常处理（新增）===
    print(f"\n   成交量异常检测与处理:")
    
    # 1.8 过滤成交量为0或负数的异常数据
    zero_volume_mask = (daily['vol'] <= 0)
    zero_volume_count = zero_volume_mask.sum()
    daily = daily[~zero_volume_mask].copy()
    print(f"   ✓ 过滤零成交量数据: 剔除 {zero_volume_count:,} 条记录")
    
    # 1.9 过滤成交量异常放大（可能是数据错误或特殊事件）
    # 计算每只股票的成交量移动平均和标准差
    daily['vol_ma'] = daily.groupby('ts_code')['vol'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    daily['vol_std'] = daily.groupby('ts_code')['vol'].transform(
        lambda x: x.rolling(20, min_periods=5).std()
    )
    
    # 异常放量定义：成交量超过均值+10倍标准差
    extreme_volume_mask = (
        (daily['vol'] > daily['vol_ma'] + 10 * daily['vol_std']) & 
        (daily['vol_std'] > 0)
    )
    extreme_volume_count = extreme_volume_mask.sum()
    
    if extreme_volume_count > 0:
        print(f"   ⚠️  检测到 {extreme_volume_count:,} 个异常放量数据（>均值+10σ）")
        # 对于异常放量，限制其值而不是直接删除（可能是真实的重大事件）
        daily.loc[extreme_volume_mask, 'vol'] = (
            daily.loc[extreme_volume_mask, 'vol_ma'] + 
            5 * daily.loc[extreme_volume_mask, 'vol_std']
        )
        print(f"      已将异常值限制到均值+5σ")
    
    # 1.10 过滤成交量异常缩小（可能是数据缺失）
    # 异常缩量定义：成交量低于均值的1%（且均值>0）
    extreme_low_volume_mask = (
        (daily['vol'] < daily['vol_ma'] * 0.01) & 
        (daily['vol_ma'] > 0)
    )
    extreme_low_volume_count = extreme_low_volume_mask.sum()
    daily = daily[~extreme_low_volume_mask].copy()
    print(f"   ✓ 过滤异常缩量数据（<均值1%）: 剔除 {extreme_low_volume_count:,} 条记录")
    
    # 1.11 成交量与成交额一致性检查
    if 'amount' in daily.columns:
        # 计算隐含价格：成交额 / 成交量（手转换为股）
        daily['implied_price'] = daily['amount'] * 1000 / (daily['vol'] * 100)  # 成交额单位：千元，成交量单位：手
        
        # 检查隐含价格是否与收盘价接近（允许20%误差）
        price_mismatch_mask = (
            (daily['implied_price'].notna()) &
            (daily['close'] > 0) &
            (
                (daily['implied_price'] / daily['close'] > 1.5) | 
                (daily['implied_price'] / daily['close'] < 0.5)
            )
        )
        price_mismatch_count = price_mismatch_mask.sum()
        
        if price_mismatch_count > 0:
            print(f"   ⚠️  检测到 {price_mismatch_count:,} 个量价不一致数据")
            daily = daily[~price_mismatch_mask].copy()
            print(f"      已剔除量价不一致数据")
        
        # 清理临时列
        daily = daily.drop(columns=['implied_price'], errors='ignore')
    
    # 清理临时计算列
    daily = daily.drop(columns=['vol_ma', 'vol_std'], errors='ignore')
    
    # 筛选总结
    filtered_count = len(daily)
    filter_rate = (original_count - filtered_count) / original_count * 100
    print(f"\n   📊 筛选汇总:")
    print(f"      原始数据: {original_count:,} 条")
    print(f"      筛选后: {filtered_count:,} 条")
    print(f"      过滤比例: {filter_rate:.2f}%")
    print(f"      保留股票数: {daily['ts_code'].nunique()}")
    
    # 计算各类 OBV 衍生因子
    print(f"\n步骤 2: 计算基础 OBV 和衍生因子...")
    print(f"   OBV 计算改进:")
    print(f"      ✓ 使用百分比变化判断方向（避免价格量级影响）")
    print(f"      ✓ 设置变化阈值过滤噪音（{0.0001*100:.3f}%）")
    print(f"      ✓ 价格平稳时延续前一日方向")
    print(f"      ✓ 从0开始累积（标准做法）")
    
    result_parts = []
    
    for code, group in daily.groupby('ts_code'):
        df = group.sort_values('trade_date').copy()
        
        # === 成交量质量检查（新增）===
        # 确保成交量都是正数
        if (df['vol'] <= 0).any():
            print(f"   ⚠️  警告: 股票 {code} 仍有零成交量，已跳过")
            continue
        
        # 检查成交量的连续性（是否有大量缺失）
        expected_days = (df['trade_date'].max() - df['trade_date'].min()).days
        actual_days = len(df)
        coverage_ratio = actual_days / (expected_days / 7 * 5)  # 估算交易日数量
        
        if coverage_ratio < 0.5:  # 数据覆盖率低于50%
            print(f"   ⚠️  警告: 股票 {code} 数据覆盖率仅 {coverage_ratio*100:.1f}%，已跳过")
            continue
        
        # === 优化的 OBV 计算 ===
        # 问题1: 使用相对变化而非绝对变化，避免价格量级影响
        # 问题2: 处理价格相等的情况（沿用前一天的方向）
        # 问题3: 初始值从0开始（标准做法）
        # 问题4: 成交量异常的情况需要特殊处理
        
        # 计算价格变动
        df['prev_close'] = df['close'].shift(1)
        
        # 更精确的方向判断（使用百分比变化，避免微小价格变动的噪音）
        price_change_pct = (df['close'] - df['prev_close']) / df['prev_close']
        
        # 设定阈值，小于阈值视为无变化（过滤噪音）
        price_threshold = 0.0001  # 0.01% 的变化阈值
        
        def determine_direction(change_pct, prev_direction):
            """
            确定当日方向
            - 涨幅 > 阈值: +1
            - 跌幅 < -阈值: -1
            - 其他情况: 延续前一日方向（如果是第一天，则为0）
            """
            if pd.isna(change_pct):
                return 0
            elif change_pct > price_threshold:
                return 1
            elif change_pct < -price_threshold:
                return -1
            else:
                # 价格几乎不变，延续前一日方向
                return prev_direction if not pd.isna(prev_direction) else 0
        
        # 计算方向（使用循环以支持延续前一日方向）
        directions = [0]  # 第一天默认为0
        for i in range(1, len(df)):
            direction = determine_direction(
                price_change_pct.iloc[i], 
                directions[-1]
            )
            directions.append(direction)
        
        df['direction'] = directions
        
        # 计算 OBV（从0开始累积）
        # 成交量安全性检查
        vol_with_direction = df['direction'] * df['vol']
        
        # 检查是否有异常大的单日OBV变化
        vol_median = df['vol'].median()
        extreme_obv_change = vol_with_direction.abs() > (vol_median * 100)
        
        if extreme_obv_change.any():
            extreme_count = extreme_obv_change.sum()
            print(f"   ⚠️  警告: 股票 {code} 有 {extreme_count} 个极端OBV变化，已限制")
            # 将极端值限制到中位数的50倍
            vol_with_direction = vol_with_direction.clip(
                -vol_median * 50, 
                vol_median * 50
            )
        
        df['obv'] = vol_with_direction.cumsum()
        
        # 验证：确保 OBV 计算正确
        # 对于第一天，OBV = direction * vol
        # 对于后续天，OBV = 前一天OBV + direction * vol
        
        # 数据质量检查
        if df['obv'].isna().any():
            print(f"   ⚠️  警告: 股票 {code} 存在 NaN 值，已跳过")
            continue
        
        # 过滤异常OBV值（例如突然变化过大）
        obv_change = df['obv'].diff().abs()
        obv_median_change = obv_change.median()
        # 异常值定义：变化超过中位数的100倍
        if obv_median_change > 0:
            abnormal_mask = obv_change > (obv_median_change * 100)
            if abnormal_mask.sum() > 0:
                print(f"   ⚠️  警告: 股票 {code} 存在 {abnormal_mask.sum()} 个异常OBV变化")
        
        result_parts.append(df)
    
    if not result_parts:
        raise ValueError("所有股票的 OBV 计算都失败了")
    
    print(f"✅ 完成 {len(result_parts)} 只股票的基础 OBV 计算")
    
    # OBV 计算质量检查
    print(f"\n   OBV 计算质量检查:")
    all_obv_changes = []
    all_vol_ratios = []  # 新增：成交量比率检查
    
    for df in result_parts:
        obv_pct_change = df['obv'].pct_change().abs()
        all_obv_changes.extend(obv_pct_change.dropna().tolist())
        
        # 检查成交量的稳定性
        vol_ratio = df['vol'] / df['vol'].rolling(20).mean()
        all_vol_ratios.extend(vol_ratio.dropna().tolist())
    
    if all_obv_changes:
        all_obv_changes = pd.Series(all_obv_changes)
        print(f"      OBV 变化率中位数: {all_obv_changes.median():.4f}")
        print(f"      OBV 变化率均值: {all_obv_changes.mean():.4f}")
        print(f"      OBV 异常变化(>100%)比例: {(all_obv_changes > 1.0).mean():.2%}")
    
    if all_vol_ratios:
        all_vol_ratios = pd.Series(all_vol_ratios)
        print(f"\n      成交量稳定性检查:")
        print(f"         成交量/均值 中位数: {all_vol_ratios.median():.2f}")
        print(f"         异常放量(>5倍均值)比例: {(all_vol_ratios > 5).mean():.2%}")
        print(f"         异常缩量(<0.2倍均值)比例: {(all_vol_ratios < 0.2).mean():.2%}")
    
    # 计算衍生因子
    print(f"\n   计算 OBV 衍生因子...")
    for i, df in enumerate(result_parts):
        # === 因子 1: OBV 趋势斜率 ===
        # 使用线性回归计算斜率
        def calc_slope(series):
            if len(series) < trend_period or series.isna().any():
                return np.nan
            x = np.arange(len(series))
            y = series.values
            try:
                slope = np.polyfit(x, y, 1)[0]
                return slope
            except:
                return np.nan
        
        df['obv_slope'] = df['obv'].rolling(trend_period).apply(calc_slope, raw=False)
        
        # === 因子 2: OBV 变化率 ===
        # 使用安全的百分比变化计算，避免除以0
        obv_shifted = df['obv'].shift(trend_period)
        # 避免除以0或极小值
        df['obv_change'] = np.where(
            obv_shifted.abs() > 1e-10,  # 只有当分母足够大时才计算
            (df['obv'] - obv_shifted) / obv_shifted.abs(),
            0  # 否则返回0
        )
        
        # 限制变化率范围，避免极端值
        df['obv_change'] = df['obv_change'].clip(-10, 10)  # 限制在±1000%
        
        # === 因子 3: OBV 相对强度（分位数排名）===
        df['obv_rank'] = df['obv'].rolling(rank_period).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= trend_period else np.nan,
            raw=False
        )
        
        # === 因子 4: 量价背离度 ===
        # 计算价格趋势斜率
        def calc_price_slope(series):
            if len(series) < divergence_period or series.isna().any():
                return np.nan
            x = np.arange(len(series))
            y = series.values
            try:
                slope = np.polyfit(x, y, 1)[0]
                return slope
            except:
                return np.nan
        
        price_slope = df['close'].rolling(divergence_period).apply(calc_price_slope, raw=False)
        obv_slope_div = df['obv'].rolling(divergence_period).apply(calc_price_slope, raw=False)
        
        # 标准化后计算背离度（正值表示量能强于价格）
        # 使用更稳健的标准化方法
        def safe_normalize(series):
            """安全的标准化，避免除以0"""
            mean = series.mean()
            std = series.std()
            if pd.isna(mean) or pd.isna(std) or std < 1e-8:
                return pd.Series(0, index=series.index)
            return (series - mean) / std
        
        price_slope_norm = safe_normalize(price_slope)
        obv_slope_norm = safe_normalize(obv_slope_div)
        df['obv_divergence'] = obv_slope_norm - price_slope_norm
        
        # === 因子 5: OBV 突破强度 ===
        df['obv_high'] = df['obv'].rolling(rank_period).max()
        obv_high_shifted = df['obv_high'].shift(1)
        
        # 安全的突破强度计算
        df['obv_breakthrough'] = np.where(
            obv_high_shifted.abs() > 1e-10,
            (df['obv'] - obv_high_shifted) / obv_high_shifted.abs(),
            0
        )
        
        # 限制突破强度范围
        df['obv_breakthrough'] = df['obv_breakthrough'].clip(-5, 5)
        
        result_parts[i] = df
    
    print(f"✅ 完成所有衍生因子计算")
    
    # 合并所有股票
    combined = pd.concat(result_parts, ignore_index=True)
    
    # 过滤到指定日期范围
    combined = combined[combined['trade_date'] >= start_date].copy()
    
    print(f"\n步骤 3: 截面标准化各子因子...")
    
    # 截面标准化每个因子
    factor_cols = ['obv_slope', 'obv_change', 'obv_rank', 'obv_divergence', 'obv_breakthrough']
    
    print(f"   标准化前异常值统计:")
    for col in factor_cols:
        col_data = combined[col].dropna()
        if len(col_data) > 0:
            print(f"      {col}:")
            print(f"         均值: {col_data.mean():.4f}, 标准差: {col_data.std():.4f}")
            print(f"         最小值: {col_data.min():.4f}, 最大值: {col_data.max():.4f}")
            print(f"         极端值(|z|>5): {(abs((col_data - col_data.mean()) / (col_data.std() + 1e-8)) > 5).sum()} 个")
    
    # 第一步：处理无穷值和缺失值
    for col in factor_cols:
        # 替换无穷值为 NaN
        combined[col] = combined[col].replace([np.inf, -np.inf], np.nan)
        
        # 统计缺失值
        nan_count = combined[col].isna().sum()
        if nan_count > 0:
            print(f"   处理 {col} 的 {nan_count} 个缺失值")
    
    # 第二步：极端值处理（在标准化之前，按日期分组处理）
    print(f"\n   极端值处理（MAD 方法）:")
    for col in factor_cols:
        def winsorize_by_mad(series, n_mad=5):
            """
            使用 MAD (Median Absolute Deviation) 方法处理极端值
            比标准差方法更稳健，不受极端值影响
            """
            if series.isna().all() or len(series) < 10:
                return series
            
            median = series.median()
            mad = (series - median).abs().median()
            
            if mad == 0:
                # MAD 为 0 说明数据变化很小，使用标准差方法
                std = series.std()
                if std > 0:
                    lower = series.mean() - 5 * std
                    upper = series.mean() + 5 * std
                else:
                    return series
            else:
                # MAD 方法：极端值定义为偏离中位数超过 n_mad 个 MAD
                lower = median - n_mad * 1.4826 * mad  # 1.4826 是使 MAD 等价于标准差的系数
                upper = median + n_mad * 1.4826 * mad
            
            # Winsorize：将极端值拉回到边界
            return series.clip(lower, upper)
        
        # 按日期分组处理极端值
        combined[col] = combined.groupby('trade_date')[col].transform(
            lambda x: winsorize_by_mad(x, n_mad=5)
        )
        
        # 统计处理效果
        after_data = combined[col].dropna()
        if len(after_data) > 0:
            print(f"      {col}: 范围 [{after_data.min():.4f}, {after_data.max():.4f}]")
    
    # 第三步：截面标准化
    print(f"\n   执行截面标准化...")
    
    def safe_standardize(series, min_std=1e-8, min_samples=10):
        """
        安全的截面标准化函数
        
        Parameters
        ----------
        series : pd.Series
            待标准化的序列
        min_std : float
            最小标准差阈值，低于此值视为无变化
        min_samples : int
            最小样本数，低于此值不进行标准化
        
        Returns
        -------
        pd.Series
            标准化后的序列
        """
        # 移除 NaN 值
        valid_data = series.dropna()
        
        # 样本数不足
        if len(valid_data) < min_samples:
            return pd.Series(0, index=series.index)
        
        # 计算统计量
        mean = valid_data.mean()
        std = valid_data.std()
        
        # 检查统计量有效性
        if pd.isna(mean) or pd.isna(std):
            return pd.Series(0, index=series.index)
        
        # 标准差过小（数据几乎无变化）
        if std < min_std:
            return pd.Series(0, index=series.index)
        
        # 执行标准化
        result = (series - mean) / std
        
        # 替换可能产生的无穷值
        result = result.replace([np.inf, -np.inf], 0)
        
        return result
    
    for col in factor_cols:
        combined[col] = combined.groupby('trade_date')[col].transform(safe_standardize)
        
        # 验证标准化结果
        invalid_count = combined[col].isin([np.inf, -np.inf]).sum()
        if invalid_count > 0:
            print(f"      警告: {col} 存在 {invalid_count} 个无穷值，已清理")
            combined[col] = combined[col].replace([np.inf, -np.inf], np.nan)
    
    # 第四步：标准化后再次截尾（限制在 [-3, 3] 标准差范围内）
    print(f"   标准化后截尾处理（±3σ）...")
    clip_count = {}
    for col in factor_cols:
        original = combined[col].copy()
        combined[col] = combined[col].clip(-3, 3)
        clipped = (original != combined[col]).sum()
        clip_count[col] = clipped
        if clipped > 0:
            print(f"      {col}: 截尾 {clipped} 个值")
    
    # 第五步：最终数据质量检查
    print(f"\n   最终数据质量检查:")
    for col in factor_cols:
        col_data = combined[col].dropna()
        if len(col_data) > 0:
            # 检查是否还有异常值
            extreme_count = (col_data.abs() > 3).sum()
            inf_count = np.isinf(col_data).sum()
            nan_ratio = combined[col].isna().mean()
            
            print(f"      {col}:")
            print(f"         |z|>3: {extreme_count} 个 ({extreme_count/len(col_data)*100:.2f}%)")
            print(f"         无穷值: {inf_count} 个")
            print(f"         缺失率: {nan_ratio*100:.2f}%")
            
            if extreme_count > 0 or inf_count > 0:
                print(f"         ⚠️  警告: 仍存在异常值！")
    
    print(f"✅ 完成截面标准化")
    
    print(f"\n步骤 4: 合成综合因子...")
    
    # 合成最终因子（等权平均，可根据 IC 测试调整权重）
    # 推荐权重：趋势因子权重更高
    weights = {
        'obv_slope': 0.30,        # 趋势斜率
        'obv_change': 0.20,       # 变化率
        'obv_rank': 0.20,         # 相对强度
        'obv_divergence': 0.15,   # 量价背离
        'obv_breakthrough': 0.15, # 突破强度
    }
    
    # 检查子因子质量
    print(f"   子因子质量检查:")
    for col, weight in weights.items():
        valid_count = combined[col].notna().sum()
        valid_ratio = valid_count / len(combined)
        print(f"      {col} (权重={weight}): 有效率 {valid_ratio*100:.2f}%")
    
    combined['factor'] = sum(combined[col] * weight for col, weight in weights.items())
    
    # 处理综合因子的缺失值
    # 如果某些子因子缺失，综合因子也会是 NaN，这是合理的
    factor_nan_count = combined['factor'].isna().sum()
    if factor_nan_count > 0:
        print(f"   综合因子缺失值: {factor_nan_count} 个 ({factor_nan_count/len(combined)*100:.2f}%)")
    
    # 再次标准化综合因子
    print(f"   标准化综合因子...")
    combined['factor'] = combined.groupby('trade_date')['factor'].transform(safe_standardize)
    
    # 检查标准化后的无穷值
    factor_inf_after = combined['factor'].isin([np.inf, -np.inf]).sum()
    if factor_inf_after > 0:
        print(f"      清理 {factor_inf_after} 个无穷值")
        combined['factor'] = combined['factor'].replace([np.inf, -np.inf], np.nan)
    
    # 综合因子最终异常值处理
    print(f"   综合因子异常值处理:")
    
    # 使用 MAD 方法
    def final_winsorize(series):
        if series.isna().all() or len(series) < 10:
            return series
        median = series.median()
        mad = (series - median).abs().median()
        if mad > 0:
            lower = median - 5 * 1.4826 * mad
            upper = median + 5 * 1.4826 * mad
            return series.clip(lower, upper)
        return series
    
    combined['factor'] = combined.groupby('trade_date')['factor'].transform(final_winsorize)
    
    # 最终截尾
    original_factor = combined['factor'].copy()
    combined['factor'] = combined['factor'].clip(-3, 3)
    final_clip_count = (original_factor != combined['factor']).sum()
    print(f"      最终截尾: {final_clip_count} 个值")
    
    # 综合因子统计
    factor_data = combined['factor'].dropna()
    if len(factor_data) > 0:
        print(f"      均值: {factor_data.mean():.4f}")
        print(f"      标准差: {factor_data.std():.4f}")
        print(f"      偏度: {factor_data.skew():.4f}")
        print(f"      峰度: {factor_data.kurtosis():.4f}")
        print(f"      范围: [{factor_data.min():.4f}, {factor_data.max():.4f}]")
    
    print(f"✅ 综合因子合成完成")
    print(f"   因子权重: {weights}")
    
    # 设置 MultiIndex
    result = combined[['trade_date', 'ts_code', 'factor'] + factor_cols].set_index(['trade_date', 'ts_code'])
    
    # 统计信息
    print(f"\n{'='*60}")
    print(f"✅ OBV 增强版因子计算完成！")
    print(f"   有效记录数: {len(result):,}")
    print(f"   覆盖股票数: {result.index.get_level_values('ts_code').nunique()}")
    print(f"   覆盖交易日数: {result.index.get_level_values('trade_date').nunique()}")
    
    print(f"\n因子值统计（综合因子）:")
    factor_stats = result['factor'].describe()
    print(f"   数量: {int(factor_stats['count']):,}")
    print(f"   均值: {factor_stats['mean']:.4f}")
    print(f"   标准差: {factor_stats['std']:.4f}")
    print(f"   最小值: {factor_stats['min']:.4f}")
    print(f"   25%分位: {factor_stats['25%']:.4f}")
    print(f"   中位数: {factor_stats['50%']:.4f}")
    print(f"   75%分位: {factor_stats['75%']:.4f}")
    print(f"   最大值: {factor_stats['max']:.4f}")
    
    # 检查数据分布健康度
    print(f"\n数据质量评估:")
    skewness = result['factor'].skew()
    kurtosis = result['factor'].kurtosis()
    print(f"   偏度: {skewness:.4f} {'(正常)' if abs(skewness) < 1 else '(偏斜较大)'}")
    print(f"   峰度: {kurtosis:.4f} {'(正常)' if abs(kurtosis) < 3 else '(尖峰或厚尾)'}")
    
    # 缺失值统计
    total_possible = len(result.index.get_level_values('trade_date').unique()) * len(result.index.get_level_values('ts_code').unique())
    missing_rate = (total_possible - len(result)) / total_possible
    print(f"   缺失率: {missing_rate*100:.2f}%")
    
    print(f"{'='*60}\n")
    
    return result


def run_obv_factor_backtest(start_date: str = '2024-01-01',
                          end_date: str = '2024-02-29',
                          stock_codes: Optional[List[str]] = None,
                          rebalance_freq: str = 'weekly',
                          transaction_cost: float = 0.0003,
                          long_direction: str = 'high',
                          use_advanced: bool = True,
                          trend_period: int = 20,
                          divergence_period: int = 20,
                          rank_period: int = 120,
                          # 数据质量筛选参数
                          filter_st: bool = True,
                          filter_suspend: bool = True,
                          filter_limit: bool = True,
                          min_turnover_rate: float = 0.01,
                          min_price: float = 1.0) -> dict:
    """
    使用 BacktestEngine 主路径运行 OBV 因子策略回测。
    
    Parameters
    ----------
    start_date, end_date : str
        回测周期
    stock_codes : Optional[List[str]]
        股票池
    rebalance_freq : str
        调仓频率: 'daily', 'weekly', 'monthly'
    transaction_cost : float
        单边交易费用
    long_direction : str
        多头方向: 'high' 做多高因子值（推荐），'low' 做多低因子值
    use_advanced : bool
        是否使用增强版因子（包含趋势+背离），默认 True
    trend_period : int
        趋势计算周期
    divergence_period : int
        背离计算周期
    rank_period : int
        相对强度计算周期
    filter_st : bool
        是否过滤 ST 股票，默认 True
    filter_suspend : bool
        是否过滤停牌数据，默认 True
    filter_limit : bool
        是否过滤涨跌停数据，默认 True
    min_turnover_rate : float
        最小换手率阈值（%），默认 0.01%
    min_price : float
        最小价格阈值（元），默认 1.0 元
    
    Returns
    -------
    dict
        包含因子数据、组合收益、业绩指标和IC分析结果
    """
    # 初始化数据管理器
    data_manager = DataManager()
    
    # 使用 BacktestEngine 主路径
    from backtest_engine.engine import BacktestEngine
    
    print("\n" + "=" * 60)
    factor_type = "OBV 增强版因子" if use_advanced else "基础 OBV 因子"
    print(f"开始计算 {factor_type}...")
    
    # === 步骤 1: 计算因子（带异常处理）===
    try:
        if use_advanced:
            factor_data = calculate_obv_advanced_factor(
                data_manager=data_manager,
                start_date=start_date,
                end_date=end_date,
                stock_codes=stock_codes,
                trend_period=trend_period,
                divergence_period=divergence_period,
                rank_period=rank_period,
                filter_st=filter_st,
                filter_suspend=filter_suspend,
                filter_limit=filter_limit,
                min_turnover_rate=min_turnover_rate,
                min_price=min_price,
            )
        else:
            factor_data = calculate_obv_factor(
                data_manager=data_manager,
                start_date=start_date,
                end_date=end_date,
                stock_codes=stock_codes
            )
    except Exception as e:
        print(f"❌ 因子计算失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'factor_data': None,
            'portfolio_returns': None,
            'positions': None,
            'performance_metrics': {},
            'analysis_results': {},
            'error': str(e)
        }
    
    # === 步骤 2: 空值检查 ===
    if factor_data is None:
        print("❌ 因子数据为 None")
        return {
            'factor_data': None,
            'portfolio_returns': None,
            'positions': None,
            'performance_metrics': {},
            'analysis_results': {},
            'error': 'factor_data is None'
        }
    
    if factor_data.empty:
        print("❌ 因子数据为空")
        return {
            'factor_data': factor_data,
            'portfolio_returns': None,
            'positions': None,
            'performance_metrics': {},
            'analysis_results': {},
            'error': 'factor_data is empty'
        }
    
    # 检查因子列是否存在
    if 'factor' not in factor_data.columns:
        print("❌ 因子数据缺少 'factor' 列")
        return {
            'factor_data': factor_data,
            'portfolio_returns': None,
            'positions': None,
            'performance_metrics': {},
            'analysis_results': {},
            'error': 'Missing factor column'
        }
    
    # 检查因子值的有效性
    valid_factor_count = factor_data['factor'].notna().sum()
    total_factor_count = len(factor_data)
    
    if valid_factor_count == 0:
        print("❌ 所有因子值都是 NaN")
        return {
            'factor_data': factor_data,
            'portfolio_returns': None,
            'positions': None,
            'performance_metrics': {},
            'analysis_results': {},
            'error': 'All factor values are NaN'
        }
    
    if valid_factor_count < total_factor_count * 0.1:
        print(f"⚠️  警告: 有效因子值比例过低 ({valid_factor_count/total_factor_count*100:.1f}%)")
    
    print(f"因子值范围: [{factor_data['factor'].min():.4f}, {factor_data['factor'].max():.4f}]")
    
    # 回测前数据质量检查
    print(f"\n回测前数据质量检查:")
    factor_inf_count = np.isinf(factor_data['factor']).sum()
    factor_nan_count = factor_data['factor'].isna().sum()
    print(f"   无穷值数量: {factor_inf_count}")
    print(f"   缺失值数量: {factor_nan_count}")
    print(f"   有效值数量: {valid_factor_count} ({valid_factor_count/total_factor_count*100:.1f}%)")
    
    if factor_inf_count > 0 or factor_nan_count > 0:
        print(f"   清理异常值...")
        # 移除无穷值和缺失值
        factor_data = factor_data[~np.isinf(factor_data['factor'])].copy()
        factor_data = factor_data.dropna(subset=['factor'])
        print(f"   清理后记录数: {len(factor_data)}")
        
        if len(factor_data) == 0:
            print("❌ 清理后无有效数据")
            return {
                'factor_data': None,
                'portfolio_returns': None,
                'positions': None,
                'performance_metrics': {},
                'analysis_results': {},
                'error': 'No valid data after cleaning'
            }
    
    print("=" * 60 + "\n")
    
    # 创建回测引擎
    engine = BacktestEngine(
        data_manager=data_manager,
        fee=transaction_cost,
        long_direction=long_direction,
        rebalance_freq=rebalance_freq,
        factor_name='factor',
    )
    
    # 直接设置因子数据
    engine.factor_data = factor_data[['factor']]  # 只使用综合因子列
    
    # === 步骤 3: 准备收益率数据（带空值保护）===
    print("准备收益率数据...")
    
    stock_list = factor_data.index.get_level_values('ts_code').unique().tolist()
    
    if not stock_list:
        print("❌ 股票列表为空")
        return {
            'factor_data': factor_data,
            'portfolio_returns': None,
            'positions': None,
            'performance_metrics': {},
            'analysis_results': {},
            'error': 'Empty stock list'
        }
    
    print(f"   加载 {len(stock_list)} 只股票的数据...")
    
    try:
        stock_data = data_manager.load_data(
            'daily',
            start_date=start_date,
            end_date=end_date,
            stock_codes=stock_list
        )
    except Exception as e:
        print(f"❌ 加载股票数据失败: {e}")
        return {
            'factor_data': factor_data,
            'portfolio_returns': None,
            'positions': None,
            'performance_metrics': {},
            'analysis_results': {},
            'error': f'Failed to load stock data: {e}'
        }
    
    if stock_data is None:
        print("❌ 股票数据为 None")
        return {
            'factor_data': factor_data,
            'portfolio_returns': None,
            'positions': None,
            'performance_metrics': {},
            'analysis_results': {},
            'error': 'stock_data is None'
        }
    
    if stock_data.empty:
        print("❌ 股票数据为空")
        return {
            'factor_data': factor_data,
            'portfolio_returns': None,
            'positions': None,
            'performance_metrics': {},
            'analysis_results': {},
            'error': 'stock_data is empty'
        }
    
    print(f"   成功加载 {len(stock_data)} 条记录")
    
    # 计算次日收益率（带空值保护）
    print("   计算收益率...")
    stock_data = stock_data.sort_values(['ts_code', 'trade_date'])
    
    # 确保 close 列存在且有效
    if 'close' not in stock_data.columns:
        print("❌ 股票数据缺少 'close' 列")
        return {
            'factor_data': factor_data,
            'portfolio_returns': None,
            'positions': None,
            'performance_metrics': {},
            'analysis_results': {},
            'error': 'Missing close price column'
        }
    
    # 检查价格有效性
    valid_close_count = stock_data['close'].notna().sum()
    if valid_close_count == 0:
        print("❌ 所有收盘价都是 NaN")
        return {
            'factor_data': factor_data,
            'portfolio_returns': None,
            'positions': None,
            'performance_metrics': {},
            'analysis_results': {},
            'error': 'All close prices are NaN'
        }
    
    # 安全计算收益率
    try:
        stock_data['next_return'] = stock_data.groupby('ts_code')['close'].pct_change().shift(-1)
    except Exception as e:
        print(f"❌ 计算收益率失败: {e}")
        return {
            'factor_data': factor_data,
            'portfolio_returns': None,
            'positions': None,
            'performance_metrics': {},
            'analysis_results': {},
            'error': f'Failed to calculate returns: {e}'
        }
    
    # 检查收益率有效性
    valid_return_count = stock_data['next_return'].notna().sum()
    print(f"   有效收益率数量: {valid_return_count} ({valid_return_count/len(stock_data)*100:.1f}%)")
    
    if valid_return_count == 0:
        print("❌ 所有收益率都是 NaN")
        return {
            'factor_data': factor_data,
            'portfolio_returns': None,
            'positions': None,
            'performance_metrics': {},
            'analysis_results': {},
            'error': 'All returns are NaN'
        }
    
    # === 步骤 4: 合并因子和收益率（带空值保护）===
    print("   合并因子和收益率数据...")
    
    factor_reset = factor_data.reset_index()
    stock_subset = stock_data[['ts_code', 'trade_date', 'next_return']].copy()
    
    try:
        engine.combined_data = pd.merge(
            factor_reset[['trade_date', 'ts_code', 'factor']],
            stock_subset,
            on=['ts_code', 'trade_date'],
            how='inner'
        )
    except Exception as e:
        print(f"❌ 合并数据失败: {e}")
        return {
            'factor_data': factor_data,
            'portfolio_returns': None,
            'positions': None,
            'performance_metrics': {},
            'analysis_results': {},
            'error': f'Failed to merge data: {e}'
        }
    
    print(f"   合并后记录数: {len(engine.combined_data)}")
    
    if engine.combined_data.empty:
        print("❌ 合并后数据为空（可能是日期不匹配）")
        return {
            'factor_data': factor_data,
            'portfolio_returns': None,
            'positions': None,
            'performance_metrics': {},
            'analysis_results': {},
            'error': 'Empty data after merge'
        }
    
    # 清理缺失值
    before_clean = len(engine.combined_data)
    engine.combined_data.dropna(subset=['factor', 'next_return'], inplace=True)
    after_clean = len(engine.combined_data)
    
    if before_clean > after_clean:
        print(f"   清理缺失值: {before_clean - after_clean} 条")
    
    if engine.combined_data.empty:
        print("❌ 清理缺失值后数据为空")
        return {
            'factor_data': factor_data,
            'portfolio_returns': None,
            'positions': None,
            'performance_metrics': {},
            'analysis_results': {},
            'error': 'Empty data after cleaning NaN'
        }
    
    print(f"   最终有效记录: {len(engine.combined_data)}")
    
    # === 步骤 5: 运行回测（带异常处理）===
    print("\n开始回测...")
    
    try:
        portfolio_returns = engine.run()
    except Exception as e:
        print(f"❌ 回测运行失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'factor_data': factor_data,
            'portfolio_returns': None,
            'positions': None,
            'performance_metrics': {},
            'analysis_results': {},
            'error': f'Backtest execution failed: {e}'
        }
    
    print("回测完成！\n")
    
    # === 步骤 6: 计算业绩指标（带空值保护）===
    if portfolio_returns is None:
        print("❌ 回测结果为 None")
        return {
            'factor_data': factor_data,
            'portfolio_returns': None,
            'positions': None,
            'performance_metrics': {},
            'analysis_results': {},
            'error': 'portfolio_returns is None'
        }
    
    
    # 计算基本业绩指标（基于 Long_Only）
    if not isinstance(portfolio_returns, pd.DataFrame):
        print("❌ 回测结果不是 DataFrame")
        return {
            'factor_data': factor_data,
            'portfolio_returns': portfolio_returns,
            'positions': None,
            'performance_metrics': {},
            'analysis_results': {},
            'error': 'portfolio_returns is not a DataFrame'
        }
    
    if 'Long_Only' not in portfolio_returns.columns:
        print(f"❌ 回测结果缺少 'Long_Only' 列")
        print(f"   可用列: {portfolio_returns.columns.tolist()}")
        return {
            'factor_data': factor_data,
            'portfolio_returns': portfolio_returns,
            'positions': None,
            'performance_metrics': {},
            'analysis_results': {},
            'error': 'Missing Long_Only column'
        }
    
    series = portfolio_returns['Long_Only']
    
    # 检查收益序列有效性
    if series is None or len(series) == 0:
        print("❌ Long_Only 收益序列为空")
        return {
            'factor_data': factor_data,
            'portfolio_returns': portfolio_returns,
            'positions': None,
            'performance_metrics': {},
            'analysis_results': {},
            'error': 'Empty returns series'
        }
    
    valid_returns = series.notna().sum()
    if valid_returns == 0:
        print("❌ 所有收益率都是 NaN")
        return {
            'factor_data': factor_data,
            'portfolio_returns': portfolio_returns,
            'positions': None,
            'performance_metrics': {},
            'analysis_results': {},
            'error': 'All returns are NaN'
        }
    
    print(f"有效收益率数量: {valid_returns}/{len(series)}")
    
    # 安全计算业绩指标
    try:
        cum = (1 + series).cumprod()
        
        if len(cum) == 0 or cum.isna().all():
            raise ValueError("累积收益计算失败")
        
        total_return = float(cum.iloc[-1] - 1) if len(cum) > 0 else 0.0
        trading_days = len(series)
        
        if trading_days > 0:
            annualized_return = float(cum.iloc[-1] ** (252 / trading_days) - 1)
        else:
            annualized_return = 0.0
        
        volatility = float(series.std() * np.sqrt(252))
        
        if volatility > 0 and not np.isnan(annualized_return):
            sharpe_ratio = float(annualized_return / volatility)
        else:
            sharpe_ratio = 0.0
        
        running_max = cum.cummax()
        drawdown = cum / running_max - 1
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
        
    except Exception as e:
        print(f"⚠️  业绩指标计算失败: {e}")
        total_return = 0.0
        annualized_return = 0.0
        volatility = 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0
    
    # === 步骤 7: IC 分析（带异常处理）===
    try:
        analyzer = engine.get_performance_analysis()
        metrics_df = analyzer.calculate_metrics()
        ic_series = analyzer.ic_series
        
        analysis_results = {
            'metrics': metrics_df,
            'ic_series': ic_series
        }
    except Exception as e:
        print(f"⚠️  IC 分析失败: {e}")
        analysis_results = {
            'metrics': None,
            'ic_series': None
        }
    
    return {
        'factor_data': factor_data,
        'portfolio_returns': portfolio_returns,
        'positions': None,
        'performance_metrics': {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'rebalance_count': len(engine._get_rebalance_dates()),
        },
        'analysis_results': analysis_results,
    }


def main():
    """主函数：演示 OBV 增强版因子计算和回测"""
    print("=" * 60)
    print("OBV 增强版因子策略演示")
    print("包含: 趋势斜率 + 变化率 + 相对强度 + 量价背离 + 突破强度")
    print("新增: 数据质量筛选（ST/停牌/涨跌停/低流动性/低价股）")
    print("新增: 异常值处理（MAD方法 + Winsorize + 截尾）")
    print("=" * 60)

    try:
        # 配置参数
        config = {
            'start_date': '2015-01-01',
            'end_date': '2025-09-30',
            'rebalance_freq': 'weekly',
            'transaction_cost': 0.0003,  # 0.03% 交易费用
            'long_direction': 'high',  # 做多高因子值（资金流入强）
            'use_advanced': True,  # 使用增强版因子
            'trend_period': 20,  # 趋势计算周期
            'divergence_period': 20,  # 背离计算周期
            'rank_period': 120,  # 相对强度周期
            # 数据质量筛选参数
            'filter_st': True,  # 过滤 ST 股票
            'filter_suspend': True,  # 过滤停牌数据
            'filter_limit': True,  # 过滤涨跌停数据
            'min_turnover_rate': 0.01,  # 最小换手率 0.01%
            'min_price': 1.0,  # 最小价格 1 元
        }

        print("\n回测配置:")
        print(f"  时间范围: {config['start_date']} ~ {config['end_date']}")
        print(f"  调仓频率: {config['rebalance_freq']}")
        print(f"  交易费用: {config['transaction_cost']:.4f}")
        print(f"\n因子参数:")
        print(f"  趋势周期: {config['trend_period']} 天")
        print(f"  背离周期: {config['divergence_period']} 天")
        print(f"  相对强度周期: {config['rank_period']} 天")
        print(f"\n数据质量筛选:")
        print(f"  过滤 ST 股票: {config['filter_st']}")
        print(f"  过滤停牌数据: {config['filter_suspend']}")
        print(f"  过滤涨跌停: {config['filter_limit']}")
        print(f"  最小换手率: {config['min_turnover_rate']}%")
        print(f"  最小价格: {config['min_price']} 元")

        # 运行回测
        results = run_obv_factor_backtest(**config)
        
        # 检查回测是否成功
        if 'error' in results and results['error']:
            print(f"\n❌ 回测失败: {results['error']}")
            return
        
        if results['portfolio_returns'] is None:
            print(f"\n❌ 回测未返回有效结果")
            return

        # 结果总结（基于 Long_Only）
        print("\n" + "=" * 60)
        print("回测结果总结 (Long_Only)")
        print("=" * 60)
        
        metrics = results['performance_metrics']
        
        # 检查指标有效性
        if not metrics:
            print("⚠️  无法获取业绩指标")
            return
        
        print(f"\n📊 业绩指标:")
        print(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"  总收益: {metrics.get('total_return', 0):.2%}")
        print(f"  年化收益: {metrics.get('annualized_return', 0):.2%}")
        print(f"  年化波动: {metrics.get('volatility', 0):.2%}")
        print(f"  最大回撤: {metrics.get('max_drawdown', 0):.2%}")
        print(f"  调仓次数: {metrics.get('rebalance_count', 0)}")

        # IC 分析
        analysis_results = results.get('analysis_results', {})
        if analysis_results and analysis_results.get('ic_series') is not None:
            ic = analysis_results['ic_series']
            if len(ic) > 0:
                print(f"\n📊 IC 分析:")
                print(f"  IC 均值: {ic.mean():.4f}")
                print(f"  IC 标准差: {ic.std():.4f}")
                if ic.std() > 0:
                    print(f"  ICIR: {ic.mean() / ic.std():.4f}")
                print(f"  IC>0 占比: {(ic > 0).mean():.2%}")
            else:
                print(f"\n⚠️  IC 序列为空")
        else:
            print(f"\n⚠️  无法获取 IC 分析结果")
        
        # 子因子贡献分析
        if results['factor_data'] is not None and 'obv_slope' in results['factor_data'].columns:
            print(f"\n📊 子因子统计:")
            sub_factors = ['obv_slope', 'obv_change', 'obv_rank', 'obv_divergence', 'obv_breakthrough']
            for factor_name in sub_factors:
                if factor_name in results['factor_data'].columns:
                    factor_series = results['factor_data'][factor_name]
                    valid_count = factor_series.notna().sum()
                    
                    if valid_count > 0:
                        print(f"\n  {factor_name}:")
                        print(f"    有效值: {valid_count}")
                        print(f"    均值: {factor_series.mean():.4f}")
                        print(f"    标准差: {factor_series.std():.4f}")
                        print(f"    最小值: {factor_series.min():.4f}")
                        print(f"    最大值: {factor_series.max():.4f}")
                    else:
                        print(f"\n  {factor_name}: 无有效值")
                else:
                    print(f"\n  {factor_name}: 未找到")
        else:
            print(f"\n⚠️  无法获取子因子数据")
        
        print("\n" + "=" * 60)
        print("💡 策略说明:")
        print("  • 做多高因子值股票（资金持续流入 + 量价配合）")
        print("  • OBV 趋势因子捕捉主力建仓信号")
        print("  • 量价背离因子识别价量不匹配风险")
        print("  • 突破因子确认强势股持续性")
        print("  • 数据质量筛选确保交易可行性")
        print("\n💡 数据处理特点:")
        print("  • OBV 计算：百分比变化 + 方向延续 + 噪音过滤")
        print("  • 成交量处理：")
        print("    - 零成交量/停牌数据过滤")
        print("    - 异常放量限制（>均值+10σ → 限制到+5σ）")
        print("    - 异常缩量过滤（<均值1%）")
        print("    - 量价一致性验证")
        print("    - 单日极端OBV变化限制（<50倍中位数）")
        print("  • 异常值处理：MAD 方法（比标准差更稳健）")
        print("  • 多层防护：Winsorize → 标准化 → 截尾")
        print("  • 质量监控：偏度/峰度/缺失率/成交量稳定性")
        print("  • 数值稳定性：")
        print("    - 所有除法操作添加最小值保护（1e-10）")
        print("    - 标准化前检查样本数和标准差")
        print("    - 自动检测并替换无穷值/NaN")
        print("    - 百分比变化限制在合理范围（±1000%）")
        print("=" * 60)

        print("\n✅ OBV 增强版因子策略演示完成!")

    except Exception as e:
        print(f"\n❌ 演示运行失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
