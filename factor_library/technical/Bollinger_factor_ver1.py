import pandas as pd
import numpy as np
import sys
import warnings
from pathlib import Path
from typing import Optional, List

# 路径设置
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
        可选：
        - 000300.SH (沪深300)
        - 000905.SH (中证500)
        - 000852.SH (中证1000)
        - 000016.SH (上证50)
    trade_date : Optional[str]
        指定日期，格式 YYYY-MM-DD 或 YYYYMMDD
        如果为None，使用最新一期数据
    
    Returns
    -------
    List[str]
        成分股代码列表
    """
    # 直接从raw_data加载指数权重数据
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
            warnings.warn(f"未找到指数 {index_code} 在日期 {trade_date} 的数据，将使用最新一期")
            # 回退到最新一期
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

def calculate_bollinger_bands_factor(
    data_manager: DataManager,
    start_date: str,
    end_date: str,
    window: int = 20,
    num_std: float = 2,
    stock_codes: Optional[List[str]] = None,
    use_index_components: bool = True,
    index_code: str = '000852.SH',
    factor_type: str = 'percent_b',
    return_all_columns: bool = False,
) -> pd.DataFrame:
    """
    计算布林带因子，包括带宽(BB_Width)和%B指标
    
    Parameters
    ----------
    data_manager : DataManager
        数据管理器实例
    start_date : str
        开始日期，格式 'YYYY-MM-DD'
    end_date : str
        结束日期，格式 'YYYY-MM-DD'
    window : int
        布林带周期，默认20天
    num_std : float
        标准差倍数，默认2
    stock_codes : Optional[List[str]]
        股票代码列表，如为 None 则根据 use_index_components 参数决定
    use_index_components : bool
        是否使用指数成分股作为默认股票池，默认 True
    index_code : str
        指数代码，默认中证1000 (000852.SH)
        可选：000300.SH (沪深300), 000905.SH (中证500), 000016.SH (上证50)
    factor_type : str
        【优化6】选择作为因子的指标类型，默认 'percent_b'
        可选：'percent_b' (%B指标), 'bb_width' (带宽), 'above_upper', 'below_lower'
    return_all_columns : bool
        【优化6】是否返回所有布林带列，默认 False
        - False: 仅返回单列 'factor'，适配 BacktestEngine
        - True: 返回所有列 ['bb_width', 'percent_b', 'upper_band', 'middle_band', 'lower_band']
    
    Returns
    -------
    DataFrame
        MultiIndex (trade_date, ts_code)
        - 如果 return_all_columns=False: 返回单列 ['factor']
        - 如果 return_all_columns=True: 返回多列 ['bb_width', 'percent_b', 'upper_band', 'middle_band', 'lower_band']
    """
    print(f"\n{'='*60}")
    print("布林带因子计算")
    print(f"{'='*60}")
    
    # 【优化】股票池处理 - 优先使用指数成分股
    if stock_codes is None:
        if use_index_components:
            print(f"\n未指定股票池，尝试使用指数成分股...")
            stock_codes = get_index_components(data_manager, index_code=index_code)
            
            if not stock_codes:
                print(f"⚠️ 无法获取指数 {index_code} 成分股，改用全市场股票池")
                use_index_components = False
        
        if not use_index_components or not stock_codes:
            print(f"\n使用全市场股票池...")
            all_daily = data_manager.load_data('daily', start_date=start_date, end_date=end_date, cleaned=True)
            if all_daily is None or all_daily.empty:
                print(f"⚠️ 无法获取全市场数据，使用默认样本股票")
                stock_codes = ['000001.SZ', '000002.SZ', '000858.SZ', '600000.SH', '600036.SH', '600519.SH']
            else:
                stock_codes = all_daily['ts_code'].unique().tolist()
                print(f"✅ 获取全市场股票: {len(stock_codes)} 只")
    else:
        print(f"\n✅ 使用指定股票池: {len(stock_codes)} 只股票")

    # 【优化1】向前扩展日期以确保有足够的历史数据来计算布林带
    # 布林带需要 window 个历史数据点，为了确保 start_date 时就有有效值，
    # 需要提前加载 window * 3 个交易日的数据（考虑到周末和节假日）
    buffer_days = window * 3
    start_date_extended = pd.to_datetime(start_date) - pd.Timedelta(days=buffer_days)
    start_date_extended = start_date_extended.strftime('%Y-%m-%d')
    
    print(f"\n步骤1: 加载数据（含缓冲期）")
    print(f"   目标日期范围: {start_date} ~ {end_date}")
    print(f"   实际加载范围: {start_date_extended} ~ {end_date}")
    print(f"   缓冲天数: {buffer_days} 天（用于计算 {window} 日布林带）")
    
    # 加载日线数据
    daily = data_manager.load_data('daily', start_date=start_date_extended, end_date=end_date, stock_codes=stock_codes)
    if daily is None or daily.empty:
        raise ValueError('无法获取日行情数据')
    
    print(f"✅ 成功加载数据")
    print(f"   原始数据量: {len(daily):,} 条记录")
    print(f"   股票数量: {daily['ts_code'].nunique()}")

    # 数据预处理
    print(f"\n步骤2: 数据预处理和质量筛选")
    daily = daily.copy()
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], errors='coerce')
    if daily['trade_date'].isna().any():
        daily['trade_date'] = pd.to_datetime(daily['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
    daily = daily.dropna(subset=['trade_date', 'close'])

    # 数据质量检查
    if daily.empty:
        raise ValueError('数据预处理后无有效记录')
    
    initial_count = len(daily)
    print(f"   初始数据量: {initial_count:,} 条记录")
    
    # 【优化4】数据质量筛选
    print(f"\n   数据质量筛选:")
    
    # 筛选1: 基本数据完整性
    valid_basic = (
        daily['close'].notna() &
        (daily['close'] > 0) &
        daily['open'].notna() &
        (daily['open'] > 0) &
        daily['high'].notna() &
        (daily['high'] > 0) &
        daily['low'].notna() &
        (daily['low'] > 0)
    )
    filtered_basic = daily[~valid_basic]
    daily = daily[valid_basic]
    print(f"   - 过滤无效价格数据: {len(filtered_basic):,} 条 (价格<=0或缺失)")
    
    # 筛选2: 价格逻辑一致性
    valid_logic = (
        (daily['high'] >= daily['close']) &
        (daily['high'] >= daily['open']) &
        (daily['low'] <= daily['close']) &
        (daily['low'] <= daily['open']) &
        (daily['high'] >= daily['low'])
    )
    filtered_logic = daily[~valid_logic]
    daily = daily[valid_logic]
    print(f"   - 过滤价格逻辑错误: {len(filtered_logic):,} 条 (high<low等)")
    
    # 筛选3: 涨跌停股票
    # 涨停：涨幅 > 9.8% (考虑科创板/创业板20%涨停)
    # 跌停：跌幅 < -9.8%
    if 'pct_chg' in daily.columns:
        is_limit_up = daily['pct_chg'] > 9.8
        is_limit_down = daily['pct_chg'] < -9.8
        is_limit = is_limit_up | is_limit_down
        
        filtered_limit = daily[is_limit]
        daily = daily[~is_limit]
        
        limit_up_count = is_limit_up.sum()
        limit_down_count = is_limit_down.sum()
        print(f"   - 过滤涨停股票: {limit_up_count:,} 条")
        print(f"   - 过滤跌停股票: {limit_down_count:,} 条")
    else:
        print(f"   - 跳过涨跌停筛选 (缺少pct_chg字段)")
    
    # 筛选4: 异常波动（单日波动超过30%）
    if 'pct_chg' in daily.columns:
        extreme_volatility = daily['pct_chg'].abs() > 30
        filtered_extreme = daily[extreme_volatility]
        daily = daily[~extreme_volatility]
        print(f"   - 过滤异常波动(>30%): {len(filtered_extreme):,} 条")
    
    # 筛选5: 成交量异常
    if 'vol' in daily.columns:
        zero_volume = (daily['vol'] == 0) | daily['vol'].isna()
        filtered_volume = daily[zero_volume]
        daily = daily[~zero_volume]
        print(f"   - 过滤零成交量: {len(filtered_volume):,} 条")
        
        # 过滤成交量极端值（按股票分组，超过均值+5倍标准差）
        daily['vol_zscore'] = daily.groupby('ts_code')['vol'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        extreme_vol = daily['vol_zscore'].abs() > 5
        filtered_extreme_vol = daily[extreme_vol]
        daily = daily[~extreme_vol]
        daily = daily.drop(columns=['vol_zscore'])
        print(f"   - 过滤成交量极端值: {len(filtered_extreme_vol):,} 条")
    
    # 筛选6: ST股票（如果有股票名称信息）
    if 'name' in daily.columns:
        is_st = daily['name'].str.contains('ST|st|退|退市', regex=True, na=False)
        filtered_st = daily[is_st]
        daily = daily[~is_st]
        print(f"   - 过滤ST股票: {len(filtered_st):,} 条")
    
    # 统计过滤结果
    final_count = len(daily)
    filtered_total = initial_count - final_count
    filtered_pct = filtered_total / initial_count * 100
    
    print(f"\n   质量筛选总结:")
    print(f"   - 筛选前: {initial_count:,} 条")
    print(f"   - 筛选后: {final_count:,} 条")
    print(f"   - 过滤掉: {filtered_total:,} 条 ({filtered_pct:.2f}%)")
    print(f"   - 时间范围: {daily['trade_date'].min()} ~ {daily['trade_date'].max()}")

    # 按股票分组计算布林带指标
    print(f"\n步骤3: 计算布林带指标（window={window}, std={num_std}）")
    factor_dfs = []
    skipped_stocks = []
    error_stocks = []
    
    for code in daily['ts_code'].unique():
        stock_data = daily[daily['ts_code'] == code].sort_values('trade_date')
        
        # 检查数据充足性
        if len(stock_data) < window:
            skipped_stocks.append((code, len(stock_data), f"数据不足{window}天"))
            continue
        
        try:
            # 计算布林带指标
            middle = stock_data['close'].rolling(window=window).mean()
            std = stock_data['close'].rolling(window=window).std()
            upper = middle + num_std * std
            lower = middle - num_std * std
            
            # 【优化5】异常值检测和处理
            # 检测布林带是否有效
            invalid_bb = (
                middle.isna() | 
                std.isna() | 
                (std == 0) |  # 标准差为0（价格不变）
                (upper <= lower)  # 上下轨倒置
            )
            
            if invalid_bb.sum() > len(stock_data) * 0.5:
                # 如果超过50%的数据异常，跳过该股票
                skipped_stocks.append((code, len(stock_data), f"{invalid_bb.sum()}条异常布林带"))
                continue
            
            # 计算因子值
            bb_width = (upper - lower) / middle  # 归一化带宽
            percent_b = (stock_data['close'] - lower) / (upper - lower)
            
            # 【优化6】%B 异常值处理
            # %B 理论范围是 [0, 1]，但实际可能超出
            # 设置合理的上下限：[-1, 2]
            percent_b_clipped = percent_b.clip(lower=-1, upper=2)
            
            # 检测异常的 %B 值
            extreme_percent_b = (percent_b < -1) | (percent_b > 2)
            if extreme_percent_b.sum() > 0:
                # 记录但不跳过，只是做截断处理
                pass
            
            # 【优化7】带宽异常值处理
            # 带宽通常在 0.01-0.5 之间，极端情况可达 0-1
            # 过滤异常的带宽值
            valid_width = (bb_width > 0) & (bb_width < 1)
            
            # 构建因子DataFrame
            factor_df = pd.DataFrame({
                'bb_width': bb_width,
                'percent_b': percent_b_clipped,  # 使用截断后的值
                'above_upper': stock_data['close'] > upper,
                'below_lower': stock_data['close'] < lower
            })
            factor_df['ts_code'] = code
            factor_df['trade_date'] = stock_data['trade_date'].values
            
            # 只保留有效的布林带值
            factor_df = factor_df[~invalid_bb.values]
            
            if not factor_df.empty:
                factor_dfs.append(factor_df)
            else:
                skipped_stocks.append((code, len(stock_data), "所有布林带值无效"))
                
        except Exception as e:
            error_stocks.append((code, str(e)))
            continue

    # 统计计算结果
    print(f"   成功计算: {len(factor_dfs)} 只股票")
    
    if skipped_stocks:
        print(f"   跳过股票: {len(skipped_stocks)} 只")
        if len(skipped_stocks) <= 10:
            for code, count, reason in skipped_stocks[:5]:
                print(f"      - {code}: {reason}")
            if len(skipped_stocks) > 5:
                print(f"      ... 还有 {len(skipped_stocks)-5} 只")
        else:
            print(f"      (前5只): {', '.join([code for code, _, _ in skipped_stocks[:5]])}")
    
    if error_stocks:
        print(f"   计算错误: {len(error_stocks)} 只")
        for code, error in error_stocks[:3]:
            print(f"      - {code}: {error}")
        if len(error_stocks) > 3:
            print(f"      ... 还有 {len(error_stocks)-3} 只")
            
    if not factor_dfs:
        raise ValueError('无法计算任何股票的布林带因子')

    # 合并所有股票的因子值
    factor = pd.concat(factor_dfs, axis=0)
    factor = factor.set_index(['trade_date', 'ts_code'])
    
    # 【优化2】过滤到目标日期范围（去除缓冲期的数据）
    print(f"\n步骤4: 过滤到目标日期范围")
    print(f"   过滤前: {len(factor):,} 条记录")
    
    factor = factor.loc[factor.index.get_level_values('trade_date') >= pd.to_datetime(start_date)]
    factor = factor.loc[factor.index.get_level_values('trade_date') <= pd.to_datetime(end_date)]
    
    print(f"   过滤后: {len(factor):,} 条记录")
    print(f"   覆盖股票数: {factor.index.get_level_values('ts_code').nunique()}")
    print(f"   覆盖交易日数: {factor.index.get_level_values('trade_date').nunique()}")
    
    # 【优化3】因子值统计分析
    print(f"\n步骤5: 因子值统计分析")
    print(f"{'='*60}")
    
    # BB_Width (带宽) 统计
    if 'bb_width' in factor.columns:
        print(f"\n📊 BB_Width (带宽) 统计信息:")
        bb_stats = factor['bb_width'].describe()
        print(f"   样本数量: {bb_stats['count']:.0f}")
        print(f"   均值: {bb_stats['mean']:.4f}")
        print(f"   中位数: {bb_stats['50%']:.4f}")
        print(f"   标准差: {bb_stats['std']:.4f}")
        print(f"   最小值: {bb_stats['min']:.4f}")
        print(f"   25%分位: {bb_stats['25%']:.4f}")
        print(f"   75%分位: {bb_stats['75%']:.4f}")
        print(f"   最大值: {bb_stats['max']:.4f}")
        
        # 计算偏度和峰度
        try:
            from scipy import stats as scipy_stats
            skewness = scipy_stats.skew(factor['bb_width'].dropna())
            kurtosis = scipy_stats.kurtosis(factor['bb_width'].dropna())
            print(f"   偏度 (Skewness): {skewness:.4f}")
            print(f"   峰度 (Kurtosis): {kurtosis:.4f}")
        except:
            pass
        
        # 缺失值统计
        missing_count = factor['bb_width'].isna().sum()
        missing_pct = missing_count / len(factor) * 100
        print(f"   缺失值: {missing_count} ({missing_pct:.2f}%)")
        
        # 【新增】异常值统计
        print(f"\n   异常值检测 (带宽):")
        bb_valid = factor['bb_width'].dropna()
        q1 = bb_valid.quantile(0.25)
        q3 = bb_valid.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers_low = (bb_valid < lower_bound).sum()
        outliers_high = (bb_valid > upper_bound).sum()
        outliers_total = outliers_low + outliers_high
        outliers_pct = outliers_total / len(bb_valid) * 100
        
        print(f"   IQR范围: [{lower_bound:.4f}, {upper_bound:.4f}]")
        print(f"   低于下界: {outliers_low} ({outliers_low/len(bb_valid)*100:.2f}%)")
        print(f"   高于上界: {outliers_high} ({outliers_high/len(bb_valid)*100:.2f}%)")
        print(f"   异常值总数: {outliers_total} ({outliers_pct:.2f}%)")
        
        # 极端值
        extreme_low = (bb_valid < 0.01).sum()
        extreme_high = (bb_valid > 0.5).sum()
        print(f"   极端窄带宽(<0.01): {extreme_low} ({extreme_low/len(bb_valid)*100:.2f}%)")
        print(f"   极端宽带宽(>0.5): {extreme_high} ({extreme_high/len(bb_valid)*100:.2f}%)")
    
    # %B (价格位置) 统计
    if 'percent_b' in factor.columns:
        print(f"\n📊 %B (价格位置) 统计信息:")
        pb_stats = factor['percent_b'].describe()
        print(f"   样本数量: {pb_stats['count']:.0f}")
        print(f"   均值: {pb_stats['mean']:.4f}")
        print(f"   中位数: {pb_stats['50%']:.4f}")
        print(f"   标准差: {pb_stats['std']:.4f}")
        print(f"   最小值: {pb_stats['min']:.4f}")
        print(f"   25%分位: {pb_stats['25%']:.4f}")
        print(f"   75%分位: {pb_stats['75%']:.4f}")
        print(f"   最大值: {pb_stats['max']:.4f}")
        
        # 缺失值统计
        missing_count = factor['percent_b'].isna().sum()
        missing_pct = missing_count / len(factor) * 100
        print(f"   缺失值: {missing_count} ({missing_pct:.2f}%)")
        
        # %B 分布统计
        pb_valid = factor['percent_b'].dropna()
        print(f"\n   %B 分布:")
        print(f"   < 0 (下轨下方): {(pb_valid < 0).sum()} ({(pb_valid < 0).mean()*100:.1f}%)")
        print(f"   0-0.2 (超卖区): {((pb_valid >= 0) & (pb_valid < 0.2)).sum()} ({((pb_valid >= 0) & (pb_valid < 0.2)).mean()*100:.1f}%)")
        print(f"   0.2-0.5 (下半区): {((pb_valid >= 0.2) & (pb_valid < 0.5)).sum()} ({((pb_valid >= 0.2) & (pb_valid < 0.5)).mean()*100:.1f}%)")
        print(f"   0.5-0.8 (上半区): {((pb_valid >= 0.5) & (pb_valid < 0.8)).sum()} ({((pb_valid >= 0.5) & (pb_valid < 0.8)).mean()*100:.1f}%)")
        print(f"   0.8-1.0 (超买区): {((pb_valid >= 0.8) & (pb_valid <= 1.0)).sum()} ({((pb_valid >= 0.8) & (pb_valid <= 1.0)).mean()*100:.1f}%)")
        print(f"   > 1 (上轨上方): {(pb_valid > 1).sum()} ({(pb_valid > 1).mean()*100:.1f}%)")
        
        # 【新增】极端%B值统计
        print(f"\n   极端%B值:")
        extreme_low_pb = (pb_valid < -0.5).sum()
        extreme_high_pb = (pb_valid > 1.5).sum()
        print(f"   < -0.5 (远离下轨): {extreme_low_pb} ({extreme_low_pb/len(pb_valid)*100:.2f}%)")
        print(f"   > 1.5 (远离上轨): {extreme_high_pb} ({extreme_high_pb/len(pb_valid)*100:.2f}%)")
    
    # 布林带突破统计
    if 'above_upper' in factor.columns and 'below_lower' in factor.columns:
        print(f"\n📊 布林带突破统计:")
        above_count = factor['above_upper'].sum()
        below_count = factor['below_lower'].sum()
        total = len(factor)
        print(f"   突破上轨次数: {above_count} ({above_count/total*100:.2f}%)")
        print(f"   跌破下轨次数: {below_count} ({below_count/total*100:.2f}%)")
        print(f"   突破频率: {(above_count + below_count)/total*100:.2f}%")
    
    # 按日期统计
    print(f"\n📊 时间维度统计:")
    daily_counts = factor.groupby(factor.index.get_level_values('trade_date')).size()
    print(f"   交易日数量: {len(daily_counts)}")
    print(f"   平均每日股票数: {daily_counts.mean():.1f}")
    print(f"   最多每日股票数: {daily_counts.max()}")
    print(f"   最少每日股票数: {daily_counts.min()}")
    
    # 按股票统计
    print(f"\n📊 股票维度统计:")
    stock_counts = factor.groupby(factor.index.get_level_values('ts_code')).size()
    print(f"   股票数量: {len(stock_counts)}")
    print(f"   平均每只股票天数: {stock_counts.mean():.1f}")
    print(f"   最多每只股票天数: {stock_counts.max()}")
    print(f"   最少每只股票天数: {stock_counts.min()}")
    
    # 数据完整性
    total_possible = len(daily_counts) * len(stock_counts)
    data_completeness = len(factor) / total_possible * 100 if total_possible > 0 else 0
    print(f"\n📊 数据完整性:")
    print(f"   理论最大记录数: {total_possible:,} (交易日 × 股票数)")
    print(f"   实际记录数: {len(factor):,}")
    print(f"   完整度: {data_completeness:.1f}%")
    
    print(f"\n{'='*60}")
    print(f"✅ 布林带因子计算完成！")
    print(f"{'='*60}\n")
    
    # 【优化6】统一因子格式输出
    if return_all_columns:
        # 返回所有列用于分析
        print(f"📊 返回所有布林带列: {list(factor.columns)}")
        return factor
    else:
        # 返回单列 'factor' 用于回测
        if factor_type not in factor.columns:
            available_cols = list(factor.columns)
            raise ValueError(
                f"指定的 factor_type '{factor_type}' 不在可用列中。\n"
                f"可用列: {available_cols}\n"
                f"请选择: 'percent_b', 'bb_width', 'above_upper', 'below_lower'"
            )
        
        factor_standardized = factor[[factor_type]].copy()
        factor_standardized.columns = ['factor']
        
        print(f"📊 返回标准因子格式: 使用 '{factor_type}' 作为 'factor' 列")
        return factor_standardized


def generate_bollinger_signals(
    data_manager: DataManager,
    start_date: str,
    end_date: str,
    window: int = 20,
    num_std: float = 2,
    stock_codes: Optional[List[str]] = None,
    signal_type: str = 'oversold_bounce',
    bb_width_percentile: float = 0.2,
    use_index_components: bool = True,
    index_code: str = '000852.SH',
) -> pd.DataFrame:
    """
    基于布林带构建选股信号
    
    **选股策略说明**：
    1. **oversold_bounce（超卖反弹）**：
       - 价格触及下轨（%B < 0.1）后反弹
       - 适用于震荡市，捕捉超卖反弹机会
       
    2. **breakout_upper（突破上轨）**：
       - 价格突破上轨（%B > 0.9）
       - 适用于趋势市，追踪强势股
       
    3. **squeeze_expansion（缩口扩张）**：
       - 带宽从历史低位扩张（前期bb_width < 20%分位，当前扩张>10%）
       - 捕捉波动率扩张初期，可能出现大行情
       
    4. **middle_support（中轨支撑）**：
       - 价格回踩中轨获得支撑（0.4 < %B < 0.6，前日%B在此区间外）
       - 适用于上升趋势中的回调买点
    
    Parameters
    ----------
    data_manager : DataManager
        数据管理器
    start_date, end_date : str
        回测周期
    window : int
        布林带周期，默认20天
    num_std : float
        标准差倍数，默认2
    stock_codes : Optional[List[str]]
        股票池，如为 None 则根据 use_index_components 参数决定
    signal_type : str
        信号类型，可选：
        - 'oversold_bounce': 超卖反弹
        - 'breakout_upper': 突破上轨
        - 'squeeze_expansion': 缩口扩张
        - 'middle_support': 中轨支撑
    bb_width_percentile : float
        用于squeeze_expansion策略的带宽分位数阈值（默认0.2，即20%分位）
    use_index_components : bool
        是否使用指数成分股作为默认股票池，默认 True
    index_code : str
        指数代码，默认中证1000 (000852.SH)
    
    Returns
    -------
    pd.DataFrame
        MultiIndex (trade_date, ts_code) with column 'factor'.
        factor值为1表示产生信号，NaN表示无信号
    """
    print(f"\n{'='*60}")
    print(f"布林带选股信号生成 - {signal_type}")
    print(f"{'='*60}")
    
    # 【优化7】详细的步骤日志
    print(f"\n步骤1: 计算布林带因子")
    print(f"   时间范围: {start_date} ~ {end_date}")
    print(f"   布林带周期: {window} 天")
    print(f"   标准差倍数: {num_std}")
    if use_index_components:
        print(f"   股票池: {index_code} 成分股")
    elif stock_codes:
        print(f"   股票池: 指定股票 ({len(stock_codes)} 只)")
    else:
        print(f"   股票池: 全市场")
    
    # 【优化3】计算布林带因子时已经处理了缓冲期和股票池，这里直接调用即可
    bb_factor = calculate_bollinger_bands_factor(
        data_manager=data_manager,
        start_date=start_date,
        end_date=end_date,
        window=window,
        num_std=num_std,
        stock_codes=stock_codes,
        use_index_components=use_index_components,
        index_code=index_code,
        return_all_columns=True,  # 需要所有列来生成信号
    )
    
    if bb_factor.empty:
        print("⚠️ 布林带因子数据为空")
        return pd.DataFrame(columns=['factor']).rename_axis(['trade_date', 'ts_code'])
    
    print(f"✅ 布林带因子计算完成")
    print(f"   数据记录数: {len(bb_factor):,}")
    print(f"   覆盖股票数: {bb_factor.index.get_level_values('ts_code').nunique()}")
    print(f"   覆盖交易日数: {bb_factor.index.get_level_values('trade_date').nunique()}")
    
    # 重置索引以便处理
    df = bb_factor.reset_index()
    
    # 【优化7】步骤2: 按股票生成信号
    print(f"\n步骤2: 生成选股信号")
    print(f"   信号类型: {signal_type}")
    
    # 信号策略描述
    strategy_desc = {
        'oversold_bounce': '超卖反弹 - 价格触及下轨后反弹',
        'breakout_upper': '突破上轨 - 价格突破上轨追踪强势',
        'squeeze_expansion': f'缩口扩张 - 带宽从低位(<{bb_width_percentile*100:.0f}%分位)扩张',
        'middle_support': '中轨支撑 - 价格回踩中轨获得支撑',
    }
    print(f"   策略说明: {strategy_desc.get(signal_type, '未知策略')}")
    
    # 按股票分组生成信号
    signal_dfs = []
    total_stocks = df['ts_code'].nunique()
    processed_count = 0
    skipped_count = 0
    
    print(f"   处理进度:")
    
    for code in df['ts_code'].unique():
        stock_df = df[df['ts_code'] == code].sort_values('trade_date').copy()
        
        if len(stock_df) < window + 5:
            skipped_count += 1
            continue
        
        # 计算辅助指标
        stock_df['percent_b_prev'] = stock_df['percent_b'].shift(1)
        stock_df['bb_width_prev'] = stock_df['bb_width'].shift(1)
        stock_df['above_upper_prev'] = stock_df['above_upper'].shift(1)
        stock_df['below_lower_prev'] = stock_df['below_lower'].shift(1)
        
        # 计算带宽历史分位数（用于squeeze_expansion）
        stock_df['bb_width_percentile'] = stock_df['bb_width'].rolling(
            window=window*3, min_periods=window
        ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        
        # 带宽变化率
        stock_df['bb_width_change'] = (
            stock_df['bb_width'] / stock_df['bb_width_prev'] - 1
        )
        
        # 根据不同策略生成信号
        if signal_type == 'oversold_bounce':
            # 策略1：超卖反弹
            # 条件：前日%B < 0.1（触及下轨），当日%B > 0.1（反弹）
            signal = (
                (stock_df['percent_b_prev'] < 0.1) &
                (stock_df['percent_b'] > 0.1) &
                (stock_df['percent_b'] < 0.5)  # 确保还在下半部分
            )
            
        elif signal_type == 'breakout_upper':
            # 策略2：突破上轨
            # 条件：前日%B < 0.9，当日%B > 0.9（突破上轨）
            signal = (
                (stock_df['percent_b_prev'] < 0.9) &
                (stock_df['percent_b'] > 0.9)
            )
            
        elif signal_type == 'squeeze_expansion':
            # 策略3：缩口扩张
            # 条件：前期带宽处于低位（< 20%分位），当前带宽开始扩张（>10%）
            signal = (
                (stock_df['bb_width_percentile'].shift(1) < bb_width_percentile) &
                (stock_df['bb_width_change'] > 0.1) &
                (stock_df['percent_b'] > 0.3) &  # 价格不能太低
                (stock_df['percent_b'] < 0.7)    # 价格不能太高
            )
            
        elif signal_type == 'middle_support':
            # 策略4：中轨支撑
            # 条件：价格回踩到中轨附近（0.4 < %B < 0.6），前日不在此区间
            in_middle_zone = (stock_df['percent_b'] > 0.4) & (stock_df['percent_b'] < 0.6)
            was_outside = (stock_df['percent_b_prev'] <= 0.4) | (stock_df['percent_b_prev'] >= 0.6)
            signal = in_middle_zone & was_outside
            
        else:
            raise ValueError(f"不支持的信号类型: {signal_type}")
        
        # 构建信号DataFrame
        signal_df = stock_df[['trade_date', 'ts_code']].copy()
        signal_df['factor'] = np.where(signal, 1.0, np.nan)
        signal_dfs.append(signal_df)
        
        processed_count += 1
        
        # 【优化7】进度提示（每处理100只股票输出一次）
        if processed_count % 100 == 0:
            progress_pct = processed_count / total_stocks * 100
            print(f"      已处理 {processed_count}/{total_stocks} 只股票 ({progress_pct:.1f}%)")
    
    # 【优化7】最终处理统计
    print(f"   ✅ 完成处理:")
    print(f"      - 成功处理: {processed_count} 只股票")
    print(f"      - 跳过（数据不足）: {skipped_count} 只股票")
    
    if not signal_dfs:
        print("⚠️ 未生成任何有效信号")
        return pd.DataFrame(columns=['factor']).rename_axis(['trade_date', 'ts_code'])
    
    # 【优化7】步骤3: 合并和筛选信号
    print(f"\n步骤3: 合并和筛选信号")
    
    # 合并所有信号
    all_signals = pd.concat(signal_dfs, axis=0)
    print(f"   合并前: {len(all_signals):,} 条记录")
    
    all_signals = all_signals.dropna(subset=['factor'])
    print(f"   去除空值后: {len(all_signals):,} 条记录（有效信号）")
    
    if all_signals.empty:
        print("⚠️ 所有信号均被过滤")
        return pd.DataFrame(columns=['factor']).rename_axis(['trade_date', 'ts_code'])
    
    # 设置MultiIndex
    result = all_signals.set_index(['trade_date', 'ts_code'])
    
    # 【优化7】步骤4: 信号统计分析
    print(f"\n步骤4: 信号统计分析")
    total_signals = len(result)
    unique_stocks = result.index.get_level_values('ts_code').nunique()
    unique_dates = result.index.get_level_values('trade_date').nunique()
    
    print(f"   有效信号总数: {total_signals:,}")
    print(f"   涉及股票数: {unique_stocks}")
    print(f"   涉及交易日数: {unique_dates}")
    print(f"   平均每日信号数: {total_signals / unique_dates:.1f}")
    print(f"   平均每股信号数: {total_signals / unique_stocks:.1f}")
    
    # 【优化7】按日期统计
    daily_signals = result.groupby(result.index.get_level_values('trade_date')).size()
    print(f"\n   日期维度统计:")
    print(f"      - 信号最多的日期: {daily_signals.idxmax().strftime('%Y-%m-%d')} ({daily_signals.max()} 个)")
    print(f"      - 信号最少的日期: {daily_signals.idxmin().strftime('%Y-%m-%d')} ({daily_signals.min()} 个)")
    print(f"      - 每日信号数中位数: {daily_signals.median():.0f}")
    
    # 【优化7】按股票统计
    stock_signals = result.groupby(result.index.get_level_values('ts_code')).size()
    print(f"\n   股票维度统计:")
    print(f"      - 信号最多的股票: {stock_signals.idxmax()} ({stock_signals.max()} 次)")
    print(f"      - 每股信号数中位数: {stock_signals.median():.0f}")
    top5_stocks = stock_signals.nlargest(5)
    print(f"      - 信号最多的5只股票:")
    for stock, count in top5_stocks.items():
        print(f"         * {stock}: {count} 次")
    
    print(f"\n{'='*60}")
    print(f"✅ 选股信号生成完成")
    print(f"{'='*60}\n")
    
    return result
    print(f"   信号频率标准差: {stock_signals.std():.2f}")
    
    # 信号密度分析
    print(f"\n📊 信号密度分析:")
    total_possible = unique_dates * unique_stocks
    signal_density = total_signals / total_possible * 100 if total_possible > 0 else 0
    print(f"   理论最大信号数: {total_possible:,} (交易日 × 股票数)")
    print(f"   实际信号数: {total_signals:,}")
    print(f"   信号密度: {signal_density:.2f}%")
    
    # 信号稀疏度评估
    if signal_density < 1:
        print(f"   评估: 信号非常稀疏，高度选择性 ⭐⭐⭐")
    elif signal_density < 5:
        print(f"   评估: 信号较为稀疏，选择性较强 ⭐⭐")
    elif signal_density < 10:
        print(f"   评估: 信号适中，有一定选择性 ⭐")
    else:
        print(f"   评估: 信号较为频繁，选择性一般")
    
    print(f"{'='*60}\n")
    
    return result

def run_bb_factor_backtest(
    start_date: str = '2024-01-01',
    end_date: str = '2024-02-29',
    stock_codes: Optional[List[str]] = None,
    window: int = 20,
    num_std: float = 2,
    rebalance_freq: str = 'weekly',
    transaction_cost: float = 0.0003,
    factor_type: str = 'percent_b',  # 【优化6】选择因子类型
    use_index_components: bool = True,
    index_code: str = '000852.SH',
) -> dict:
    """
    使用布林带因子进行回测（使用 BacktestEngine 标准流程）
    
    Parameters
    ----------
    start_date, end_date : str
        回测周期
    stock_codes : Optional[List[str]]
        股票池，如为 None 则根据 use_index_components 参数决定
    window : int
        布林带周期，默认20天
    num_std : float
        标准差倍数，默认2
    rebalance_freq : str
        调仓频率，默认 'weekly'
    transaction_cost : float
        交易成本，默认 0.0003
    factor_type : str
        【优化6】选择作为因子的指标类型，默认 'percent_b'
        可选：'percent_b', 'bb_width', 'above_upper', 'below_lower'
    use_index_components : bool
        是否使用指数成分股作为默认股票池，默认 True
    index_code : str
        指数代码，默认中证1000 (000852.SH)
        
    Returns
    -------
    dict
        包含回测结果的字典
    """
    print("=" * 60)
    print(f"布林带因子回测（因子类型: {factor_type}）")
    print("=" * 60)
    
    # 【优化7】详细配置信息
    print(f"\n📋 回测配置:")
    print(f"   时间范围: {start_date} ~ {end_date}")
    print(f"   因子类型: {factor_type}")
    print(f"   布林带周期: {window} 天")
    print(f"   标准差倍数: {num_std}")
    print(f"   调仓频率: {rebalance_freq}")
    print(f"   交易成本: {transaction_cost:.4f}")
    if use_index_components:
        print(f"   股票池: {index_code} 成分股")
    elif stock_codes:
        print(f"   股票池: 指定股票 ({len(stock_codes)} 只)")
    else:
        print(f"   股票池: 全市场")
    
    data_manager = DataManager()

    try:
        from backtest_engine.engine import BacktestEngine
        
        # 【优化7】步骤1: 计算因子
        print(f"\n{'='*60}")
        print("步骤1: 计算布林带因子")
        print(f"{'='*60}")
        
        # 【优化6】使用 calculate_bollinger_bands_factor 计算因子
        factor_data = calculate_bollinger_bands_factor(
            data_manager=data_manager,
            start_date=start_date,
            end_date=end_date,
            window=window,
            num_std=num_std,
            stock_codes=stock_codes,
            use_index_components=use_index_components,
            index_code=index_code,
            factor_type=factor_type,
            return_all_columns=False,  # 返回标准格式
        )
        
        if factor_data.empty:
            print("⚠️ 未生成任何因子数据，无法回测")
            return {
                'factor_data': None,
                'portfolio_returns': None,
                'performance_metrics': {},
                'analysis_results': {}
            }
        
        # 【优化7】步骤2: 初始化回测引擎
        print(f"\n{'='*60}")
        print("步骤2: 初始化回测引擎")
        print(f"{'='*60}")
        
        # bb_width 和 percent_b 都是"高值更好"的因子
        long_direction = 'high' if factor_type in ['bb_width', 'percent_b'] else 'low'
        
        print(f"   回测引擎配置:")
        print(f"      - 因子方向: {long_direction} (做多{'高因子值' if long_direction == 'high' else '低因子值'})")
        print(f"      - 调仓频率: {rebalance_freq}")
        print(f"      - 交易费率: {transaction_cost:.4f}")
        
        engine = BacktestEngine(
            data_manager=data_manager,
            fee=transaction_cost,
            long_direction=long_direction,
            rebalance_freq=rebalance_freq,
            factor_name='factor',  # 【优化6】统一使用 'factor' 列名
        )
        
        # 设置因子数据
        engine.factor_data = factor_data
        print(f"   ✅ 因子数据已设置")
        print(f"      - 因子记录数: {len(factor_data):,}")
        print(f"      - 覆盖股票: {factor_data.index.get_level_values('ts_code').nunique()} 只")
        print(f"      - 覆盖交易日: {factor_data.index.get_level_values('trade_date').nunique()} 天")
        
        # 【优化7】步骤3: 运行回测
        print(f"\n{'='*60}")
        print("步骤3: 运行回测")
        print(f"{'='*60}")
        print(f"   正在计算组合收益...")
        
        # 运行回测
        portfolio_returns = engine.run()
        
        print(f"   ✅ 回测计算完成")
        print(f"      - 收益序列长度: {len(portfolio_returns)}")
        
        # 【优化7】步骤4: 计算性能指标
        print(f"\n{'='*60}")
        print("步骤4: 计算性能指标")
        print(f"{'='*60}")
        if not isinstance(portfolio_returns, pd.DataFrame) or 'Long_Only' not in portfolio_returns.columns:
            raise ValueError('回测结果缺少 Long_Only 列')
        
        series = portfolio_returns['Long_Only']
        cum = (1 + series).cumprod()
        perf_metrics = {
            'total_return': float(cum.iloc[-1] - 1) if len(cum) else 0.0,
            'annualized_return': float(cum.iloc[-1] ** (252 / len(series)) - 1) if len(series) > 0 else 0.0,
            'volatility': float(series.std() * np.sqrt(252)),
            'max_drawdown': float((cum / cum.cummax() - 1).min()) if len(cum) else 0.0,
            'rebalance_count': len(engine._get_rebalance_dates()),
        }
        perf_metrics['sharpe_ratio'] = (
            perf_metrics['annualized_return'] / perf_metrics['volatility']
            if perf_metrics['volatility'] > 0 else 0.0
        )
        
        print(f"   ✅ 性能指标计算完成")
        print(f"      - 调仓次数: {perf_metrics['rebalance_count']}")
        
        # 【优化7】步骤5: IC 分析
        print(f"\n{'='*60}")
        print("步骤5: 因子有效性分析（IC）")
        print(f"{'='*60}")
        
        # 获取因子分析结果
        analyzer = engine.get_performance_analysis()
        analysis_results = {
            'metrics': analyzer.calculate_metrics(),
            'ic_series': analyzer.ic_series
        }
        
        if analysis_results['ic_series'] is not None and not analysis_results['ic_series'].empty:
            ic_count = len(analysis_results['ic_series'])
            print(f"   ✅ IC 分析完成")
            print(f"      - IC 序列长度: {ic_count}")
        else:
            print(f"   ⚠️ IC 数据不可用")
        
        # 打印结果
        print("\n" + "=" * 60)
        print("回测结果")
        print("=" * 60)
        
        print(f"\n📊 业绩指标 (Long_Only):")
        print(f"  总收益率: {perf_metrics['total_return']:.2%}")
        print(f"  年化收益率: {perf_metrics['annualized_return']:.2%}")
        print(f"  年化波动率: {perf_metrics['volatility']:.2%}")
        print(f"  夏普比率: {perf_metrics['sharpe_ratio']:.2f}")
        print(f"  最大回撤: {perf_metrics['max_drawdown']:.2%}")
        print(f"  调仓次数: {perf_metrics['rebalance_count']}")
        
        if analysis_results['ic_series'] is not None and not analysis_results['ic_series'].empty:
            ic_mean = analysis_results['ic_series'].mean()
            ic_std = analysis_results['ic_series'].std()
            icir = ic_mean / ic_std if ic_std > 0 else 0
            ic_positive_ratio = (analysis_results['ic_series'] > 0).mean()
            
            print(f"\n📊 IC分析:")
            print(f"  IC均值: {ic_mean:.4f}")
            print(f"  IC标准差: {ic_std:.4f}")
            print(f"  ICIR: {icir:.4f}")
            print(f"  IC>0占比: {ic_positive_ratio:.2%}")
        
        return {
            'factor_data': factor_data,
            'portfolio_returns': portfolio_returns,
            'performance_metrics': perf_metrics,
            'analysis_results': analysis_results,
        }
        
    except Exception as e:
        print(f"回测过程发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def run_bb_signal_backtest(
    start_date: str = '2024-01-01',
    end_date: str = '2024-12-31',
    stock_codes: Optional[List[str]] = None,
    window: int = 20,
    num_std: float = 2,
    signal_type: str = 'oversold_bounce',
    rebalance_freq: str = 'daily',
    transaction_cost: float = 0.0003,
    use_index_components: bool = True,
    index_code: str = '000852.SH',
) -> dict:
    """
    使用布林带选股信号进行回测（使用 BacktestEngine 标准流程）
    
    Parameters
    ----------
    start_date, end_date : str
        回测周期
    stock_codes : Optional[List[str]]
        股票池，如为 None 则根据 use_index_components 参数决定
    window : int
        布林带周期
    num_std : float
        标准差倍数
    signal_type : str
        信号类型：'oversold_bounce', 'breakout_upper', 'squeeze_expansion', 'middle_support'
    rebalance_freq : str
        调仓频率（信号策略通常使用'daily'）
    transaction_cost : float
        交易成本
    use_index_components : bool
        是否使用指数成分股作为默认股票池，默认 True
    index_code : str
        指数代码，默认中证1000 (000852.SH)
        
    Returns
    -------
    dict
        包含回测结果的字典
    """
    print("=" * 60)
    print("布林带选股信号回测（使用 BacktestEngine）")
    print("=" * 60)
    
    # 【优化7】详细配置信息
    print(f"\n📋 回测配置:")
    print(f"   时间范围: {start_date} ~ {end_date}")
    print(f"   信号类型: {signal_type}")
    print(f"   布林带周期: {window} 天")
    print(f"   标准差倍数: {num_std}")
    print(f"   调仓频率: {rebalance_freq}")
    print(f"   交易成本: {transaction_cost:.4f}")
    if stock_codes is None and use_index_components:
        print(f"   股票池: {index_code} 成分股")
    elif stock_codes:
        print(f"   股票池: 指定股票 ({len(stock_codes)} 只)")
    else:
        print(f"   股票池: 全市场")
    
    # 初始化数据管理器
    data_manager = DataManager()
    
    # 【优化7】步骤1: 生成选股信号
    print(f"\n{'='*60}")
    print("步骤1: 生成选股信号")
    print(f"{'='*60}")
    
    # 生成选股信号
    signal_data = generate_bollinger_signals(
        data_manager=data_manager,
        start_date=start_date,
        end_date=end_date,
        window=window,
        num_std=num_std,
        stock_codes=stock_codes,
        signal_type=signal_type,
        use_index_components=use_index_components,
        index_code=index_code,
    )
    
    if signal_data.empty:
        print("⚠️ 未生成任何信号，无法回测")
        return {
            'factor_data': None,
            'portfolio_returns': None,
            'performance_metrics': {},
            'analysis_results': {}
        }
    
    # 【优化】使用 BacktestEngine 标准流程
    try:
        from backtest_engine.engine import BacktestEngine
        
        # 【优化7】步骤2: 初始化回测引擎
        print(f"\n{'='*60}")
        print("步骤2: 初始化回测引擎")
        print(f"{'='*60}")
        
        print(f"   回测引擎配置:")
        print(f"      - 信号方向: 做多（信号值=1表示买入）")
        print(f"      - 调仓频率: {rebalance_freq}")
        print(f"      - 交易费率: {transaction_cost:.4f}")
        
        # 创建回测引擎
        engine = BacktestEngine(
            data_manager=data_manager,
            fee=transaction_cost,
            long_direction='high',  # 信号值为1表示做多
            rebalance_freq=rebalance_freq,
            factor_name='factor',
        )
        
        # 直接设置因子数据
        engine.factor_data = signal_data
        
        print(f"   ✅ 信号数据已设置")
        print(f"      - 信号记录数: {len(signal_data):,}")
        
        # 【优化7】步骤3: 准备收益率数据
        print(f"\n{'='*60}")
        print("步骤3: 准备收益率数据")
        print(f"{'='*60}")
        
        # 准备收益率数据
        stock_list = signal_data.index.get_level_values('ts_code').unique().tolist()
        print(f"   正在加载 {len(stock_list)} 只股票的行情数据...")
        
        stock_data = data_manager.load_data(
            'daily',
            start_date=start_date,
            end_date=end_date,
            stock_codes=stock_list
        )
        
        if stock_data is None or stock_data.empty:
            raise ValueError("无法加载用于回测的股票数据")
        
        print(f"   ✅ 行情数据加载完成: {len(stock_data):,} 条记录")
        
        # 计算次日收益率
        print(f"   正在计算次日收益率...")
        stock_data = stock_data.sort_values(['ts_code', 'trade_date'])
        stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'])
        stock_data['next_return'] = stock_data.groupby('ts_code')['close'].pct_change().shift(-1)
        
        # 合并因子和收益率
        print(f"   正在合并信号与收益率数据...")
        factor_reset = signal_data.reset_index()
        stock_subset = stock_data[['ts_code', 'trade_date', 'next_return']].copy()
        
        engine.combined_data = pd.merge(
            factor_reset,
            stock_subset,
            on=['ts_code', 'trade_date'],
            how='inner'
        )
        engine.combined_data.dropna(subset=['factor', 'next_return'], inplace=True)
        
        if engine.combined_data.empty:
            print("⚠️ 合并后无有效数据")
            return {
                'factor_data': signal_data,
                'portfolio_returns': None,
                'performance_metrics': {},
                'analysis_results': {}
            }
        
        print(f"   ✅ 数据合并完成: {len(engine.combined_data):,} 条有效记录")
        
        # 【优化7】步骤4: 运行回测
        print(f"\n{'='*60}")
        print("步骤4: 运行回测")
        print(f"{'='*60}")
        print(f"   正在计算组合收益...")
        
        # 运行回测
        portfolio_returns = engine.run()
        
        print(f"   ✅ 回测计算完成")
        print(f"      - 收益序列长度: {len(portfolio_returns)}")
        
        # 【优化7】步骤5: 计算性能指标
        print(f"\n{'='*60}")
        print("步骤5: 计算性能指标")
        print(f"{'='*60}")
        
        # 计算基本业绩指标（基于 Long_Only）
        if not isinstance(portfolio_returns, pd.DataFrame) or 'Long_Only' not in portfolio_returns.columns:
            raise ValueError('回测结果缺少 Long_Only 列')
        
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
        
        print(f"   ✅ 性能指标计算完成")
        print(f"      - 调仓次数: {len(engine._get_rebalance_dates())}")
        
        # 【优化7】步骤6: IC 分析
        print(f"\n{'='*60}")
        print("步骤6: 因子有效性分析（IC）")
        print(f"{'='*60}")
        
        # 集成 PerformanceAnalyzer（含 IC 分析）
        analyzer = engine.get_performance_analysis()
        metrics_df = analyzer.calculate_metrics()
        ic_series = analyzer.ic_series
        
        if ic_series is not None and not ic_series.empty:
            ic_count = len(ic_series)
            print(f"   ✅ IC 分析完成")
            print(f"      - IC 序列长度: {ic_count}")
        else:
            print(f"   ⚠️ IC 数据不可用")
        
        # 打印结果
        print("\n" + "=" * 60)
        print("回测结果汇总")
        print("=" * 60)
        
        print(f"\n📊 业绩指标 (Long_Only):")
        print(f"  总收益率: {total_return:.2%}")
        print(f"  年化收益率: {annualized_return:.2%}")
        print(f"  年化波动率: {volatility:.2%}")
        print(f"  夏普比率: {sharpe_ratio:.2f}")
        print(f"  最大回撤: {max_drawdown:.2%}")
        print(f"  调仓次数: {len(engine._get_rebalance_dates())}")
        
        if ic_series is not None and not ic_series.empty:
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            icir = ic_mean / ic_std if ic_std > 0 else 0
            ic_positive_ratio = (ic_series > 0).mean()
            
            print(f"\n📊 IC分析:")
            print(f"  IC均值: {ic_mean:.4f}")
            print(f"  IC标准差: {ic_std:.4f}")
            print(f"  ICIR: {icir:.4f}")
            print(f"  IC>0占比: {ic_positive_ratio:.2%}")
        else:
            print(f"\n⚠️ IC分析不可用")
        
        print(f"\n📈 信号覆盖:")
        print(f"  有效信号数: {len(signal_data)}")
        print(f"  涉及股票数: {signal_data.index.get_level_values('ts_code').nunique()}")
        print(f"  涉及交易日数: {signal_data.index.get_level_values('trade_date').nunique()}")
        
        return {
            'factor_data': signal_data,
            'portfolio_returns': portfolio_returns,
            'performance_metrics': {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'rebalance_count': len(engine._get_rebalance_dates()),
            },
            'analysis_results': {
                'metrics': metrics_df,
                'ic_series': ic_series,
                'ic_mean': ic_series.mean() if ic_series is not None and not ic_series.empty else None,
                'ic_std': ic_series.std() if ic_series is not None and not ic_series.empty else None,
                'icir': (ic_series.mean() / ic_series.std()) if ic_series is not None and not ic_series.empty and ic_series.std() > 0 else None,
                'ic_positive_ratio': (ic_series > 0).mean() if ic_series is not None and not ic_series.empty else None,
            }
        }
        
    except ImportError:
        print("\n⚠️ 无法导入 BacktestEngine，使用简化回测流程")
        
        # 回退到简化版本（保持向后兼容）
        stock_list = signal_data.index.get_level_values('ts_code').unique().tolist()
        
        stock_data = data_manager.load_data(
            'daily',
            start_date=start_date,
            end_date=end_date,
            stock_codes=stock_list
        )
        
        if stock_data is None or stock_data.empty:
            raise ValueError("无法加载用于回测的股票数据")
        
        stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'])
        stock_data = stock_data.sort_values(['ts_code', 'trade_date'])
        
        # 计算下一日收益率
        stock_data['next_close'] = stock_data.groupby('ts_code')['close'].shift(-1)
        stock_data['next_return'] = (stock_data['next_close'] / stock_data['close']) - 1
        
        # 合并信号和收益数据
        combined = pd.merge(
            signal_data.reset_index(),
            stock_data[['trade_date', 'ts_code', 'next_return']],
            on=['trade_date', 'ts_code'],
            how='inner'
        )
        
        combined = combined.dropna(subset=['next_return'])
        
        if combined.empty:
            print("⚠️ 合并后无有效数据")
            return {
                'factor_data': signal_data,
                'portfolio_returns': None,
                'performance_metrics': {},
                'analysis_results': {}
            }
        
        # Long-Only策略：等权持有所有有信号的股票
        portfolio_returns = combined.groupby('trade_date')['next_return'].mean()
        
        # 模拟交易成本
        if rebalance_freq == 'daily':
            rebalance_dates = portfolio_returns.index
        else:
            freq_map = {'weekly': 'W-MON', 'monthly': 'MS'}
            rebalance_dates = pd.date_range(
                start=start_date, 
                end=end_date, 
                freq=freq_map.get(rebalance_freq)
            )
        
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
        
        # 简化版IC分析
        ic_series = None
        
        print("\n" + "=" * 60)
        print("回测结果（简化版）")
        print("=" * 60)
        
        print(f"\n📊 业绩指标:")
        print(f"  总收益率: {total_return:.2%}")
        print(f"  年化收益率: {annualized_return:.2%}")
        print(f"  年化波动率: {volatility:.2%}")
        print(f"  夏普比率: {sharpe_ratio:.2f}")
        print(f"  最大回撤: {max_drawdown:.2%}")
        
        return {
            'factor_data': signal_data,
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
                'ic_mean': None,
                'ic_std': None,
                'icir': None,
                'ic_positive_ratio': None,
            }
        }

def main():
    """主函数：演示布林带因子策略和选股信号"""
    print("=" * 60)
    print("布林带因子策略演示")
    print("默认股票池: 中证1000成分股")
    print("=" * 60)

    try:
        # 演示1：传统布林带因子回测
        print("\n【演示1：传统布林带因子回测（中证1000）】")
        config1 = {
            'start_date': '2020-01-01',
            'end_date': '2023-12-31',
            'window': 20,
            'num_std': 2,  # 【优化6】添加 num_std 参数
            'rebalance_freq': 'weekly',
            'transaction_cost': 0.0003,
            'factor_type': 'bb_width',  # 使用带宽因子
            'use_index_components': True,  # 【优化6】添加索引成分股参数
            'index_code': '000852.SH',      # 中证1000
        }

        print("\n回测配置:")
        for key, value in config1.items():
            print(f"  {key}: {value}")

        results1 = run_bb_factor_backtest(**config1)

        print("\n回测结果摘要:")
        metrics1 = results1['performance_metrics']
        print(f"  夏普比率: {metrics1['sharpe_ratio']:.3f}")
        print(f"  总收益率: {metrics1['total_return']:.2%}")
        print(f"  年化收益: {metrics1['annualized_return']:.2%}")
        print(f"  年化波动: {metrics1['volatility']:.2%}")
        print(f"  最大回撤: {metrics1['max_drawdown']:.2%}")
        print(f"  调仓次数: {metrics1['rebalance_count']}")

        print("\nIC分析:")
        if results1['analysis_results']['ic_series'] is not None:
            ic_mean = results1['analysis_results']['ic_series'].mean()
            ic_std = results1['analysis_results']['ic_series'].std()
            print(f"  IC均值: {ic_mean:.3f}")
            print(f"  IC标准差: {ic_std:.3f}")
            print(f"  IR比率: {(ic_mean/ic_std if ic_std > 0 else 0):.3f}")

        # 演示2：布林带选股信号回测（测试所有信号类型，使用中证1000）
        print("\n" + "=" * 60)
        print("【演示2：布林带选股信号回测（中证1000）】")
        print("=" * 60)
        
        signal_types = [
            'oversold_bounce',      # 超卖反弹
            'breakout_upper',       # 突破上轨
            'squeeze_expansion',    # 缩口扩张
            'middle_support'        # 中轨支撑
        ]
        
        signal_results = {}
        
        for sig_type in signal_types:
            print(f"\n{'='*60}")
            print(f"测试信号类型: {sig_type}")
            print(f"{'='*60}")
            
            config2 = {
                'start_date': '2020-01-01',
                'end_date': '2023-12-31',
                'window': 20,
                'num_std': 2,
                'signal_type': sig_type,
                'rebalance_freq': 'daily',
                'transaction_cost': 0.0003,
                'use_index_components': True,  # 使用指数成分股
                'index_code': '000852.SH',     # 中证1000
            }
            
            try:
                results2 = run_bb_signal_backtest(**config2)
                signal_results[sig_type] = results2
                
                if results2['portfolio_returns'] is not None:
                    metrics2 = results2['performance_metrics']
                    print(f"\n✅ {sig_type} 策略业绩:")
                    print(f"  夏普比率: {metrics2['sharpe_ratio']:.3f}")
                    print(f"  总收益率: {metrics2['total_return']:.2%}")
                    print(f"  年化收益: {metrics2['annualized_return']:.2%}")
                    print(f"  最大回撤: {metrics2['max_drawdown']:.2%}")
                else:
                    print(f"⚠️ {sig_type} 策略无有效回测结果")
                    
            except Exception as e:
                print(f"❌ {sig_type} 策略测试失败: {e}")
                continue
        
        # 对比各策略表现
        print("\n" + "=" * 60)
        print("各信号策略业绩对比 (中证1000)")
        print("=" * 60)
        print(f"{'策略':<20} {'夏普比率':>10} {'年化收益':>10} {'最大回撤':>10}")
        print("-" * 60)
        
        for sig_type, results in signal_results.items():
            if results['portfolio_returns'] is not None:
                m = results['performance_metrics']
                print(f"{sig_type:<20} {m['sharpe_ratio']:>10.3f} {m['annualized_return']:>9.2%} {m['max_drawdown']:>9.2%}")
        
        print("\n✅ 布林带因子策略演示完成!")
        print("\n💡 提示：可以通过修改 use_index_components 和 index_code 参数来使用不同的股票池")
        print("   - 000300.SH: 沪深300")
        print("   - 000905.SH: 中证500")
        print("   - 000852.SH: 中证1000 (默认)")

    except Exception as e:
        print(f"\n❌ 演示运行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
