import pandas as pd
import numpy as np
import sys
import time
import multiprocessing
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from numba import jit
from tqdm import tqdm

# 获取CPU核心数
CPU_CORES = multiprocessing.cpu_count()

class SWMAFactorError(Exception):
    """SWMA因子计算相关的异常基类"""
    pass

class ParameterError(SWMAFactorError):
    """参数验证错误"""
    pass

class DataError(SWMAFactorError):
    """数据相关错误"""
    pass

class CalculationError(SWMAFactorError):
    """计算过程中的错误"""
    pass

def validate_parameters(
    period: int,
    start_date: str,
    end_date: str,
    batch_size: int,
    n_jobs: int,
    stock_codes: Optional[List[str]] = None,
) -> None:
    """
    验证SWMA因子计算的输入参数
    
    Parameters
    ----------
    period : int
        SWMA计算周期
    start_date : str
        开始日期
    end_date : str
        结束日期
    batch_size : int
        批处理大小
    n_jobs : int
        并行进程数
    stock_codes : Optional[List[str]]
        股票代码列表
        
    Raises
    ------
    ParameterError
        当参数验证失败时抛出
    """
    # 验证period
    if not isinstance(period, int):
        raise ParameterError("period必须是整数类型")
    if period < 2:
        raise ParameterError("period必须大于等于2")
    if period > 100:
        raise ParameterError("period不应过大（建议小于100）")
        
    # 验证日期
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        if end <= start:
            raise ParameterError("end_date必须晚于start_date")
        if end > pd.Timestamp.now():
            raise ParameterError("end_date不能晚于当前日期")
        if start.year < 1990:
            raise ParameterError("start_date不能早于1990年")
    except ValueError as e:
        raise ParameterError(f"日期格式无效: {str(e)}")
        
    # 验证batch_size
    if not isinstance(batch_size, int):
        raise ParameterError("batch_size必须是整数类型")
    if batch_size < 1:
        raise ParameterError("batch_size必须大于0")
    
    # 验证n_jobs
    if not isinstance(n_jobs, int):
        raise ParameterError("n_jobs必须是整数类型")
    if n_jobs < 1:
        raise ParameterError("n_jobs必须大于0")
    if n_jobs > CPU_CORES * 2:
        raise ParameterError(f"n_jobs不建议超过CPU核心数的2倍（当前CPU核心数: {CPU_CORES}）")
    
    # 验证stock_codes（如果提供）
    if stock_codes is not None:
        if not isinstance(stock_codes, (list, tuple, np.ndarray)):
            raise ParameterError("stock_codes必须是列表、元组或numpy数组类型")
        if not all(isinstance(code, str) for code in stock_codes):
            raise ParameterError("所有股票代码必须是字符串类型")
        if not all(len(code) >= 6 for code in stock_codes):
            raise ParameterError("股票代码格式无效")

def validate_data(data: pd.DataFrame) -> None:
    """
    验证输入数据的有效性
    
    Parameters
    ----------
    data : pd.DataFrame
        需要验证的数据
        
    Raises
    ------
    DataError
        当数据验证失败时抛出
    """
    # 验证必需的列
    required_columns = ['ts_code', 'trade_date', 'close']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise DataError(f"数据缺少必需的列: {', '.join(missing_columns)}")
    
    # 验证数据不为空
    if data.empty:
        raise DataError("输入数据为空")
    
    # 验证数据类型
    if not pd.api.types.is_numeric_dtype(data['close']):
        raise DataError("close列必须是数值类型")
    
    # 验证关键字段不存在空值
    null_counts = data[required_columns].isnull().sum()
    if null_counts.any():
        raise DataError(f"关键字段存在空值:\n{null_counts[null_counts > 0].to_string()}")
    
    # 验证价格的有效性
    if (data['close'] <= 0).any():
        raise DataError("存在无效的价格数据（小于等于0）")

# 路径：把项目根目录加入 sys.path，便于使用绝对包导入
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_manager.data import DataManager

@jit(nopython=True)
def _calculate_swma_for_array(prices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    使用numba加速的SWMA计算函数
    
    Parameters
    ----------
    prices : np.ndarray
        价格数组
    weights : np.ndarray
        权重数组
    
    Returns
    -------
    np.ndarray
        SWMA值数组
    """
    n = len(prices)
    w_len = len(weights)
    result = np.empty(n - w_len + 1)
    
    for i in range(len(result)):
        result[i] = np.sum(prices[i:i+w_len] * weights)
    
    return result

def calculate_swma_factor(
    data_manager: DataManager,
    start_date: str,
    end_date: str,
    stock_codes: Optional[List[str]] = None,
    period: int = 4,
    batch_size: int = 100,     # 批处理大小
    n_jobs: int = CPU_CORES,   # 默认使用所有CPU核心
    use_numba: bool = True     # 是否使用numba加速
) -> pd.DataFrame:
    """
    计算SWMA因子，使用对称加权移动平均。增加异常值处理和详细日志。
    
    Parameters
    ----------
    data_manager : DataManager
        数据管理器实例
    start_date : str
        开始日期，格式 'YYYY-MM-DD'
    end_date : str
        结束日期，格式 'YYYY-MM-DD'
    stock_codes : Optional[List[str]]
        股票代码列表，如为None则使用全市场股票
    period : int
        SWMA计算周期，默认4
    batch_size : int
        批处理大小，用于并行计算，默认100
    n_jobs : int
        并行进程数，默认使用所有CPU核心
    use_numba : bool
        是否使用numba加速，默认True
        
    Returns
    -------
    pd.DataFrame
        因子值DataFrame，MultiIndex (trade_date, ts_code)
        
    Raises
    ------
    ParameterError
        参数验证失败时抛出
    DataError
        数据验证失败时抛出
    CalculationError
        计算过程出错时抛出
    """
    print(f"\n{'='*60}")
    print("对称加权移动平均线 (SWMA) 因子计算")
    print(f"{'='*60}")
    
    # 步骤1: 确定股票池
    print("\n步骤1: 确定股票池...")
    if stock_codes is None:
        print("未指定股票池，使用全市场股票...")
        all_daily = data_manager.load_data('daily', start_date=start_date, end_date=end_date)
        if all_daily is None or all_daily.empty:
            raise ValueError("无法获取日线数据，请检查数据源和日期范围")
        stock_codes = all_daily['ts_code'].unique().tolist()
        print(f"✅ 股票池: {len(stock_codes)} 只股票")
    else:
        print(f"✅ 使用指定股票池: {len(stock_codes)} 只股票")

    # 步骤2: 加载数据
    print("\n步骤2: 加载行情数据...")
    # 向前扩展日期以确保有足够的数据计算SWMA
    buffer_days = period * 3  # 预留充足的缓冲期
    start_date_extended = (pd.to_datetime(start_date) - pd.Timedelta(days=buffer_days)).strftime('%Y-%m-%d')
    
    daily = data_manager.load_data('daily', start_date=start_date_extended, end_date=end_date, stock_codes=stock_codes)
    if daily is None or daily.empty:
        raise ValueError("无法获取日线数据，请检查数据源和日期范围")
    
    # 统一日期为 datetime 并排序
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], errors='coerce')
    if daily['trade_date'].isna().any():
        print("⚠️ 警告：存在无效日期，这些记录将被过滤")
    daily = daily.dropna(subset=['trade_date'])
    daily = daily.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

    print(f"✅ 成功加载数据:")
    print(f"   数据时间范围: {daily['trade_date'].min()} ~ {daily['trade_date'].max()}")
    print(f"   原始数据量: {len(daily):,} 条记录")
    print(f"   覆盖股票数: {daily['ts_code'].nunique():,}")
    print(f"   覆盖交易日数: {daily['trade_date'].nunique():,}")

    # 步骤3: 数据质量检查和预处理
    print("\n步骤3: 数据质量检查...")
    
    # 检查价格异常值
    price_mask = (daily['close'] > 0) & (daily['close'].notna())
    invalid_price_count = (~price_mask).sum()
    if invalid_price_count > 0:
        print(f"⚠️ 发现 {invalid_price_count:,} 条无效价格记录，将被过滤")
    daily = daily[price_mask].copy()

    # 检查涨跌停
    limit_up_mask = daily['pct_chg'] >= 9.8
    limit_down_mask = daily['pct_chg'] <= -9.8
    limit_count = limit_up_mask.sum() + limit_down_mask.sum()
    if limit_count > 0:
        print(f"📊 涨跌停统计:")
        print(f"   涨停数量: {limit_up_mask.sum():,}")
        print(f"   跌停数量: {limit_down_mask.sum():,}")

    # 步骤4: 计算SWMA
    print("\n步骤4: 计算SWMA...")
    try:
        start_time = time.time()
        
        # 对称加权系数
        weights = np.array([i+1 for i in range(period)])
        weights = np.concatenate([weights, weights[::-1]])
        weights = weights / weights.sum()
        
        def process_stock_batch(stock_codes_batch: List[str]) -> pd.DataFrame:
            batch_results = []
            
            for code in stock_codes_batch:
                # 获取单个股票数据
                stock_data = daily[daily['ts_code'] == code].copy()
                if len(stock_data) < len(weights):
                    continue
                
                # 计算SWMA
                prices = stock_data['close'].values
                if use_numba:
                    swma_values = _calculate_swma_for_array(prices, weights)
                else:
                    swma_values = np.convolve(prices, weights, mode='valid')
                
                # 补充NaN使长度对齐
                padded_swma = np.full(len(stock_data), np.nan)
                start_idx = (len(weights) - 1) // 2
                padded_swma[start_idx:start_idx + len(swma_values)] = swma_values
                
                stock_data['swma'] = padded_swma
                batch_results.append(stock_data)
            
            return pd.concat(batch_results) if batch_results else pd.DataFrame()
        
        # 并行处理
        unique_stocks = daily['ts_code'].unique()
        stock_batches = [
            unique_stocks[i:i + batch_size] 
            for i in range(0, len(unique_stocks), batch_size)
        ]
        
        results = []
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            # 使用tqdm显示进度
            futures = {
                executor.submit(process_stock_batch, batch): i 
                for i, batch in enumerate(stock_batches)
            }
            
            with tqdm(total=len(stock_batches), desc="计算SWMA") as pbar:
                for future in as_completed(futures):
                    batch_result = future.result()
                    if not batch_result.empty:
                        results.append(batch_result)
                    pbar.update(1)
        
        # 合并结果
        daily = pd.concat(results, ignore_index=True)
        
        # 计算因子值：价格相对SWMA的偏离度
        daily['factor'] = (daily['close'] - daily['swma']) / daily['swma']
        
        # 使用numba加速的异常值处理
        @jit(nopython=True)
        def calculate_bounds(values, n_std=3):
            valid_values = values[~np.isnan(values)]
            med = np.median(valid_values)
            std = np.std(valid_values)
            return med - n_std * std, med + n_std * std
        
        # 去除极端值
        factor_values = daily['factor'].values
        lower_bound, upper_bound = calculate_bounds(factor_values)
        
        valid_factor_mask = (
            (daily['factor'] >= lower_bound) & 
            (daily['factor'] <= upper_bound) &
            (daily['factor'].notna())
        )
        
        outlier_count = (~valid_factor_mask).sum()
        if outlier_count > 0:
            print(f"⚠️ 发现 {outlier_count:,} 条因子极端值，将被过滤")
        
        daily = daily[valid_factor_mask]
        
        end_time = time.time()
        print(f"\n✨ SWMA计算完成，耗时: {end_time - start_time:.2f}秒")
        
    except Exception as e:
        raise ValueError(f"SWMA计算过程中出现错误: {str(e)}")

    # 步骤5: 构建最终因子
    print("\n步骤5: 构建最终因子...")
    result = daily[['trade_date', 'ts_code', 'factor']].copy()
    result = result.set_index(['trade_date', 'ts_code'])
    
    # 只保留在指定日期范围内的数据
    result = result.loc[result.index.get_level_values('trade_date') >= pd.to_datetime(start_date)]
    result = result.loc[result.index.get_level_values('trade_date') <= pd.to_datetime(end_date)]
    
    # 因子统计信息
    factor_stats = result['factor'].describe()
    print(f"\nSWMA因子统计信息:")
    print(f"  均值: {factor_stats['mean']:.4f}")
    print(f"  中位数: {factor_stats['50%']:.4f}")
    print(f"  标准差: {factor_stats['std']:.4f}")
    print(f"  最小值: {factor_stats['min']:.4f}")
    print(f"  25%分位: {factor_stats['25%']:.4f}")
    print(f"  75%分位: {factor_stats['75%']:.4f}")
    print(f"  最大值: {factor_stats['max']:.4f}")
    
    print(f"\n✅ SWMA因子计算完成！")
    print(f"   最终有效记录数: {len(result):,}")
    print(f"   覆盖股票数: {result.index.get_level_values('ts_code').nunique():,}")
    print(f"   覆盖交易日数: {result.index.get_level_values('trade_date').nunique():,}")
    print(f"{'='*60}\n")
    
    return result

def run_swma_factor_backtest(
    start_date: str = '2024-01-01',
    end_date: str = '2024-02-29',
    stock_codes: Optional[List[str]] = None,
    period: int = 4,
    use_volume_weighted: bool = True,
    rebalance_freq: str = 'weekly',
    transaction_cost: float = 0.0003,
    long_direction: str = 'high',
    # 新增基本面参数
    pe_min: float = 0.0,
    pe_max: float = 150.0,
    use_marketcap_filter: bool = True,
) -> dict:
    """
    运行SWMA因子策略回测
    
    **策略说明**：
    - 基于SWMA（对称加权移动平均线）的动量策略
    - 可选择做多价格高于SWMA的股票（上升趋势）或低于SWMA的股票（反转机会）
    - 支持换手率加权和基本面筛选
    - 定期调仓
    
    Parameters
    ----------
    start_date, end_date : str
        回测周期
    stock_codes : Optional[List[str]]
        股票池，None则使用全市场
    period : int
        SWMA计算周期
    use_volume_weighted : bool
        是否使用换手率加权
    rebalance_freq : str
        调仓频率: 'daily', 'weekly', 'monthly'
    transaction_cost : float
        单边交易成本
    long_direction : str
        'high': 做多价格高于SWMA的股票（趋势策略）
        'low': 做多价格低于SWMA的股票（反转策略）
    pe_min, pe_max : float
        PE-TTM筛选范围
    use_marketcap_filter : bool
        是否使用市值筛选
        
    Returns
    -------
    dict
        包含回测结果的字典:
        - factor_data: 因子数据
        - portfolio_returns: 组合收益
        - performance_metrics: 业绩指标
        - analysis_results: 分析结果（含IC）
    """
    print("=" * 60)
    print("SWMA因子策略回测")
    print("=" * 60)
    print(f"\n回测配置:")
    print(f"  时间范围: {start_date} ~ {end_date}")
    print(f"  SWMA周期: {period}")
    print(f"  使用换手率加权: {use_volume_weighted}")
    print(f"  调仓频率: {rebalance_freq}")
    print(f"  交易成本: {transaction_cost:.4f}")
    print(f"  做多方向: {'高于SWMA' if long_direction == 'high' else '低于SWMA'}")
    print(f"  PE筛选范围: [{pe_min}, {pe_max}]")
    print(f"  使用市值筛选: {use_marketcap_filter}")
    
    # 初始化数据管理器
    data_manager = DataManager()
    
    # 计算因子
    factor_data = calculate_swma_factor(
        data_manager=data_manager,
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_codes,
        period=period,
    )
    
    if factor_data.empty:
        raise ValueError("因子计算结果为空")
    
    # 准备收益率数据
    print("\n准备股票收益率数据...")
    stock_list = factor_data.index.get_level_values('ts_code').unique().tolist()
    
    stock_data = data_manager.load_data(
        'daily',
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_list
    )
    
    if stock_data is None or stock_data.empty:
        raise ValueError("无法获取股票收益率数据")
    
    # 计算下一日收益率
    stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'])
    stock_data = stock_data.sort_values(['ts_code', 'trade_date'])
    stock_data['next_close'] = stock_data.groupby('ts_code')['close'].shift(-1)
    stock_data['next_return'] = (stock_data['next_close'] / stock_data['close']) - 1
    
    # 合并因子和收益数据
    combined = pd.merge(
        factor_data.reset_index(),
        stock_data[['trade_date', 'ts_code', 'next_return']],
        on=['trade_date', 'ts_code'],
        how='inner'
    )
    
    combined = combined.dropna(subset=['next_return'])
    
    # 确定调仓日期
    if rebalance_freq == 'daily':
        rebalance_dates = combined['trade_date'].unique()
    else:
        # 按周或月调仓
        date_groups = pd.Grouper(freq='W' if rebalance_freq == 'weekly' else 'M')
        rebalance_dates = (
            pd.DataFrame({'trade_date': combined['trade_date'].unique()})
            .set_index('trade_date')
            .groupby(date_groups)
            .first()
            .index
        )
    
    # 计算每期持仓
    portfolio_returns = []
    positions_history = []
    
    print("\n开始回测...")
    for date in rebalance_dates:
        # 获取当期因子值
        current_factors = combined[combined['trade_date'] == date]
        
        if current_factors.empty:
            continue
            
        # 根据因子值排序选股
        if long_direction == 'high':
            selected_stocks = current_factors.nlargest(10, 'factor')
        else:
            selected_stocks = current_factors.nsmallest(10, 'factor')
            
        # 等权配置
        position_size = 1.0 / len(selected_stocks)
        selected_stocks['weight'] = position_size
        
        # 记录持仓
        positions_history.append(selected_stocks[['trade_date', 'ts_code', 'weight']])
        
        # 计算组合收益
        portfolio_return = (selected_stocks['next_return'] * position_size).sum()
        
        # 考虑交易成本
        if len(portfolio_returns) > 0:  # 非首次调仓
            portfolio_return -= transaction_cost * 2  # 双边成本
            
        portfolio_returns.append({
            'trade_date': date,
            'return': portfolio_return
        })
    
    # 构建收益率序列
    portfolio_returns = pd.DataFrame(portfolio_returns)
    portfolio_returns.set_index('trade_date', inplace=True)
    portfolio_returns.columns = ['Long_Only']
    
    # 计算绩效指标
    cum_returns = (1 + portfolio_returns).cumprod()
    total_return = float(cum_returns.iloc[-1] - 1) if len(cum_returns) else 0
    
    days = len(portfolio_returns)
    annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
    
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    running_max = cum_returns.cummax()
    drawdown = cum_returns / running_max - 1
    max_drawdown = drawdown.min() if not drawdown.empty else 0
    
    # IC分析
    ic_series = combined.groupby('trade_date').apply(
        lambda x: x['factor'].corr(x['next_return'])
    )
    
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    icir = ic_mean / ic_std if ic_std > 0 else 0
    ic_positive_ratio = (ic_series > 0).mean()
    
    # 打印回测结果
    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    
    print(f"\n📊 业绩指标:")
    print(f"  总收益率: {total_return:.2%}")
    print(f"  年化收益率: {annualized_return:.2%}")
    print(f"  年化波动率: {volatility:.2%}")
    print(f"  夏普比率: {sharpe_ratio:.2f}")
    print(f"  最大回撤: {max_drawdown:.2%}")
    
    print(f"\n📈 IC分析:")
    print(f"  IC均值: {ic_mean:.4f}")
    print(f"  ICIR: {icir:.4f}")
    print(f"  IC>0占比: {ic_positive_ratio:.2%}")
    
    print(f"\n🔄 换手情况:")
    print(f"  调仓次数: {len(rebalance_dates)}")
    print(f"  平均持股数: {len(selected_stocks)}")
    
    # 返回结果
    return {
        'factor_data': factor_data,
        'portfolio_returns': portfolio_returns['Long_Only'],
        'positions': pd.concat(positions_history) if positions_history else pd.DataFrame(),
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
    """主函数：演示SWMA因子计算和回测，包含性能测试"""
    print("=" * 60)
    print("SWMA因子策略演示（性能优化版本）")
    print("=" * 60)

    try:
        # 配置参数
        config = {
            'start_date': '2022-01-01',  # 使用更近的起始日期
            'end_date': '2023-12-31',
            'period': 4,  # SWMA周期
            'use_volume_weighted': True,  # 使用换手率加权
            'rebalance_freq': 'weekly',
            'transaction_cost': 0.0003,
            'long_direction': 'high',  # SWMA策略通常做多高SWMA值
            # 基本面筛选参数
            'pe_min': 0.0,
            'pe_max': 150.0,
            'use_marketcap_filter': True,
            # 性能优化参数
            'batch_size': 100,
            'n_jobs': 4,
            'use_numba': True
        }
        
        # 性能测试
        print("\n🔍 性能测试")
        print("=" * 60)
        print(f"检测到CPU核心数: {CPU_CORES}")
        
        # 测试不同配置
        test_configs = [
            {'batch_size': 50, 'n_jobs': 1, 'use_numba': False, 'name': '基础版本（单核）'},
            {'batch_size': 100, 'n_jobs': CPU_CORES, 'use_numba': True, 'name': f'优化版本（{CPU_CORES}核）'},
        ]
        
        for test_config in test_configs:
            print(f"\n测试配置: {test_config['name']}")
            test_start = time.time()
            
            # 更新配置
            current_config = config.copy()
            current_config.update({
                'batch_size': test_config['batch_size'],
                'n_jobs': test_config['n_jobs'],
                'use_numba': test_config['use_numba']
            })

            print(f"\n配置详情:")
            print(f"  批处理大小: {test_config['batch_size']}")
            print(f"  并行进程数: {test_config['n_jobs']}")
            print(f"  使用Numba加速: {test_config['use_numba']}")
            
            # 运行测试
            results = run_swma_factor_backtest(**current_config)
            
            test_end = time.time()
            test_duration = test_end - test_start
            
            print(f"\n性能指标:")
            print(f"  总耗时: {test_duration:.2f}秒")
            if 'factor_data' in results:
                print(f"  处理记录数: {len(results['factor_data']):,}")
                print(f"  每秒处理记录数: {len(results['factor_data'])/test_duration:,.0f}")
            
            # 记录性能指标
            test_config['duration'] = test_duration
            test_config['records'] = len(results['factor_data']) if 'factor_data' in results else 0
        
        # 比较性能提升
        if len(test_configs) > 1:
            base_duration = test_configs[0]['duration']
            optimized_duration = test_configs[1]['duration']
            speedup = base_duration / optimized_duration if optimized_duration > 0 else 0
            
            print("\n📊 性能优化效果:")
            print(f"  基础版本耗时: {base_duration:.2f}秒")
            print(f"  优化版本耗时: {optimized_duration:.2f}秒")
            print(f"  性能提升: {speedup:.1f}倍")
        
        # 使用最优配置运行完整回测
        print("\n🚀 使用优化配置运行完整回测...")
        results = run_swma_factor_backtest(**config)

        if results:
            # 提取关键指标
            metrics = results['performance_metrics']
            analysis = results['analysis_results']
            
            print("\n" + "=" * 60)
            print("回测结果汇总")
            print("=" * 60)
            
            print(f"\n📊 收益指标:")
            print(f"  总收益率: {metrics['total_return']:.2%}")
            print(f"  年化收益率: {metrics['annualized_return']:.2%}")
            print(f"  年化波动率: {metrics['volatility']:.2%}")
            print(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
            print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
            
            print(f"\n📈 因子效果分析:")
            print(f"  IC均值: {analysis['ic_mean']:.4f}")
            print(f"  IC标准差: {analysis['ic_std']:.4f}")
            print(f"  ICIR: {analysis['icir']:.4f}")
            print(f"  IC>0占比: {analysis['ic_positive_ratio']:.2%}")
            
            print("\n✨ 策略特点:")
            if config['long_direction'] == 'high':
                print("  - 趋势跟随策略：做多突破SWMA的股票")
            else:
                print("  - 反转策略：做多回落至SWMA下方的股票")
            
            if config['use_volume_weighted']:
                print("  - 使用换手率加权提高信号质量")
            
            if config['use_marketcap_filter']:
                print("  - 考虑市值因素，优先选择较大市值股票")
            
            print(f"\n📅 回测区间: {config['start_date']} 至 {config['end_date']}")

        print("\n✅ SWMA因子策略演示完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 演示运行失败: {str(e)}")
        raise

class TestSWMAFactor:
    """SWMA因子计算的测试类"""
    
    def __init__(self):
        self.data_manager = DataManager()
        
        # 创建测试数据
        self.test_data = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 10 + ['000002.SZ'] * 10,
            'trade_date': pd.date_range('2023-01-01', '2023-01-10').tolist() * 2,
            'close': [10, 11, 12, 11, 10, 9, 8, 9, 10, 11] * 2,
            'turnover_rate': [1.0] * 20,
            'pe_ttm': [15.0] * 20,
            'total_mv': [1000000] * 20
        })
    
    def test_basic_calculation(self):
        """测试基本的SWMA计算功能"""
        print("\n测试1: 基本SWMA计算")
        print("=" * 60)
        try:
            result = calculate_swma_factor(
                data_manager=self.data_manager,
                start_date='2023-01-01',
                end_date='2023-01-10',
                period=4
            )
            assert not result.empty, "因子计算结果不应为空"
            assert 'factor' in result.columns, "结果中应包含factor列"
            print("✅ 基本SWMA计算测试通过")
        except Exception as e:
            print(f"❌ 基本SWMA计算测试失败: {e}")
    
    def test_parameter_validation(self):
        """测试参数验证功能"""
        print("\n测试2: 参数验证")
        print("=" * 60)
        
        test_cases = [
            {'period': -1, 'expected_error': ValueError},
            {'period': 0, 'expected_error': ValueError},
            {'start_date': '2023-13-01', 'expected_error': ValueError},
            {'end_date': '2023-01-32', 'expected_error': ValueError},
        ]
        
        for case in test_cases:
            params = {
                'data_manager': self.data_manager,
                'start_date': '2023-01-01',
                'end_date': '2023-01-10',
                'period': 4
            }
            params.update({k: v for k, v in case.items() if k != 'expected_error'})
            
            try:
                calculate_swma_factor(**params)
                print(f"❌ 测试失败: 参数 {case} 应该抛出异常")
            except case['expected_error']:
                print(f"✅ 参数验证测试通过: {case}")
    
    def test_extreme_values(self):
        """测试极端值处理"""
        print("\n测试3: 极端值处理")
        print("=" * 60)
        
        # 创建包含极端值的测试数据
        extreme_data = self.test_data.copy()
        extreme_data.loc[5, 'close'] = 1000  # 添加价格极端值
        
        try:
            result = calculate_swma_factor(
                data_manager=self.data_manager,
                start_date='2023-01-01',
                end_date='2023-01-10',
                period=4
            )
            
            # 验证极端值是否被正确处理
            assert result['factor'].max() < 10, "极端值应该被过滤"
            assert result['factor'].min() > -10, "极端值应该被过滤"
            print("✅ 极端值处理测试通过")
        except Exception as e:
            print(f"❌ 极端值处理测试失败: {e}")
    
    def test_missing_data(self):
        """测试缺失数据处理"""
        print("\n测试4: 缺失数据处理")
        print("=" * 60)
        
        # 创建包含缺失值的测试数据
        missing_data = self.test_data.copy()
        missing_data.loc[3:5, 'close'] = np.nan
        
        try:
            result = calculate_swma_factor(
                data_manager=self.data_manager,
                start_date='2023-01-01',
                end_date='2023-01-10',
                period=4
            )
            
            # 验证缺失值处理
            assert not result.isnull().all().all(), "结果不应全为空"
            print("✅ 缺失数据处理测试通过")
        except Exception as e:
            print(f"❌ 缺失数据处理测试失败: {e}")
    
    def test_performance(self):
        """测试性能优化"""
        print("\n测试5: 性能测试")
        print("=" * 60)
        
        try:
            # 测试不同的并行设置
            start_time = time.time()
            result_single = calculate_swma_factor(
                data_manager=self.data_manager,
                start_date='2023-01-01',
                end_date='2023-01-10',
                period=4,
                n_jobs=1
            )
            single_time = time.time() - start_time
            
            start_time = time.time()
            result_parallel = calculate_swma_factor(
                data_manager=self.data_manager,
                start_date='2023-01-01',
                end_date='2023-01-10',
                period=4,
                n_jobs=CPU_CORES
            )
            parallel_time = time.time() - start_time
            
            # 验证结果一致性
            assert result_single.equals(result_parallel), "并行结果应与单线程结果一致"
            
            # 输出性能比较
            print(f"单线程耗时: {single_time:.2f}秒")
            print(f"多线程耗时: {parallel_time:.2f}秒")
            print(f"性能提升: {single_time/parallel_time:.1f}倍")
            print("✅ 性能测试通过")
        except Exception as e:
            print(f"❌ 性能测试失败: {e}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("\n开始运行SWMA因子测试套件...")
        print("=" * 60)
        
        test_methods = [
            self.test_basic_calculation,
            self.test_parameter_validation,
            self.test_extreme_values,
            self.test_missing_data,
            self.test_performance
        ]
        
        passed = 0
        failed = 0
        
        for test in test_methods:
            try:
                test()
                passed += 1
            except Exception as e:
                print(f"测试 {test.__name__} 失败: {e}")
                failed += 1
        
        print("\n测试结果汇总")
        print("=" * 60)
        print(f"通过: {passed}")
        print(f"失败: {failed}")
        print(f"总计: {passed + failed}")
        print("=" * 60)

def run_tests():
    """运行测试套件"""
    tester = TestSWMAFactor()
    tester.run_all_tests()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        run_tests()
    else:
        main()
