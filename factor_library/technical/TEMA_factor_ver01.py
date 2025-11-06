"""
TEMA (Triple Exponential Moving Average) 技术因子

本模块实现了基于三重指数移动平均线的综合技术因子，
包括因子计算、回测分析、选股信号生成等完整功能。

**主要功能**：
1. calculate_tema_factor(): 计算TEMA综合技术因子
2. run_tema_factor_backtest(): 运行因子策略回测
3. generate_tema_signals(): 生成多级选股信号
4. get_top_stocks(): 获取Top N推荐股票

**TEMA简介**：
三重指数移动平均线(TEMA)是一种高级技术指标，由Patrick Mulloy于1994年开发。
通过对价格进行三重指数平滑，TEMA能够有效减少滞后性，同时保持平滑的趋势线。

核心组成部分：
- EMA1：第一次指数移动平均
- EMA2：EMA1的指数移动平均
- EMA3：EMA2的指数移动平均
- TEMA = 3*EMA1 - 3*EMA2 + EMA3

**因子构建逻辑**：
本因子整合了四个维度：
1. TEMA乖离率（40%）：价格相对TEMA的偏离程度
2. TEMA斜率（30%）：TEMA的变化趋势和方向
3. TEMA动量（20%）：TEMA的变化速度
4. TEMA交叉信号（10%）：短期TEMA与长期TEMA的交叉关系

综合评分后进行截面标准化，生成z-score形式的因子值。

**使用示例**：

基础用法：
>>> from pathlib import Path
>>> import sys
>>> PROJECT_ROOT = Path(__file__).resolve().parents[2]
>>> sys.path.append(str(PROJECT_ROOT))
>>> from data_manager.data import DataManager
>>> 
>>> data_manager = DataManager()
>>> 
>>> # 计算因子
>>> factor = calculate_tema_factor(
...     data_manager,
...     start_date='2023-01-01',
...     end_date='2023-12-31'
... )
>>> 
>>> # 生成信号
>>> signals = generate_tema_signals(factor)
>>> 
>>> # 获取推荐股票
>>> top_stocks = get_top_stocks(signals, date='2023-12-31', top_n=10)

完整回测：
>>> results = run_tema_factor_backtest(
...     start_date='2023-01-01',
...     end_date='2023-12-31',
...     rebalance_freq='weekly'
... )
>>> print(results['performance_metrics'])

**因子特点**：
- 优势：
  * 多维度综合评分，减少单一指标的假信号
  * 响应速度快，滞后性小
  * 适合短中期趋势跟踪策略
  * 参数可调，适应不同市场环境

- 局限：
  * 在震荡市中可能频繁调整
  * 需要足够的历史数据
  * 对噪音较为敏感

**数据要求**：
- 必需字段：trade_date, ts_code, close
- 最小数据量：建议至少60个交易日
- 数据质量：需要清洗异常值（收盘价≤0等）

**版本历史**：
- v1.0: 基础TEMA因子实现（面向对象）
- v2.0: 重构为函数式设计，统一接口
- v2.1: 添加DataManager集成
- v2.2: 添加详细文档和日志系统

作者：量化投资团队
日期：2025-11-06
参考：Mulloy, Patrick (1994). "Smoothing Data with Faster Moving Averages"
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Union, Optional, List, Dict
import logging
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 配置日志
logger = logging.getLogger(__name__)


def setup_logger(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
    file_mode: str = 'a'
) -> None:
    """
    配置日志系统
    
    Parameters
    ----------
    level : int, default=logging.INFO
        日志级别
    log_file : str, optional
        日志文件路径
    console : bool, default=True
        是否输出到控制台
    file_mode : str, default='a'
        文件写入模式
    """
    logger.handlers.clear()
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file, mode=file_mode, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


# 默认初始化日志
setup_logger()

# 路径配置
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_manager.data import DataManager


def get_index_components(
    data_manager: DataManager, 
    index_code: str = '000852.SH', 
    trade_date: Optional[str] = None
) -> List[str]:
    """
    获取指定指数的成分股列表
    
    Parameters
    ----------
    data_manager : DataManager
        数据管理器实例
    index_code : str, default='000852.SH'
        指数代码
        - '000852.SH': 中证1000
        - '000300.SH': 沪深300
        - '000905.SH': 中证500
        - '000016.SH': 上证50
    trade_date : Optional[str], default=None
        指定日期，格式 'YYYY-MM-DD' 或 'YYYYMMDD'
        如果为None，使用最新一期数据
    
    Returns
    -------
    List[str]
        成分股代码列表
    """
    from pathlib import Path
    import warnings
    
    # 直接从raw_data加载指数权重数据
    raw_data_path = Path(__file__).resolve().parent.parent.parent / 'data_manager' / 'raw_data' / 'index_weight_data.parquet'
    
    try:
        index_weights = pd.read_parquet(raw_data_path)
    except Exception as e:
        logger.warning(f"无法加载 index_weight 数据: {e}")
        warnings.warn(f"无法加载 index_weight 数据，请先运行 data_manager/data_loader/index_weight_data_loader.py")
        return []
    
    if index_weights is None or index_weights.empty:
        logger.warning("index_weight 数据为空")
        warnings.warn("index_weight 数据为空，请先运行 data_manager/data_loader/index_weight_data_loader.py")
        return []
    
    # 筛选指定指数
    index_data = index_weights[index_weights['index_code'] == index_code].copy()
    
    if index_data.empty:
        logger.warning(f"未找到指数 {index_code} 的权重数据")
        warnings.warn(f"未找到指数 {index_code} 的权重数据")
        return []
    
    # 如果指定了日期，筛选该日期的数据
    if trade_date is not None:
        if '-' in trade_date:
            trade_date = trade_date.replace('-', '')
        index_data = index_data[index_data['trade_date'] == trade_date]
        
        if index_data.empty:
            logger.warning(f"日期 {trade_date} 无数据，使用最新一期")
            index_data = index_weights[index_weights['index_code'] == index_code].copy()
            latest_date = index_data['trade_date'].max()
            index_data = index_data[index_data['trade_date'] == latest_date]
    else:
        # 使用最新一期数据
        latest_date = index_data['trade_date'].max()
        index_data = index_data[index_data['trade_date'] == latest_date]
    
    # 提取成分股代码
    components = index_data['con_code'].unique().tolist()
    
    logger.info(f"获取指数 {index_code} 成分股: {len(components)} 只")
    print(f"✅ 获取指数 {index_code} 成分股:")
    print(f"   日期: {index_data['trade_date'].iloc[0] if not index_data.empty else 'N/A'}")
    print(f"   成分股数量: {len(components)}")
    
    return components


def calculate_tema(series: pd.Series, n: int) -> pd.Series:
    """
    计算三重指数移动平均线(TEMA)
    
    TEMA = 3*EMA1 - 3*EMA2 + EMA3
    其中：
    - EMA1 = EMA(price, n)
    - EMA2 = EMA(EMA1, n)
    - EMA3 = EMA(EMA2, n)
    
    Parameters
    ----------
    series : pd.Series
        价格序列
    n : int
        周期参数
        
    Returns
    -------
    pd.Series
        TEMA值序列
    """
    try:
        e1 = series.ewm(span=n, adjust=False).mean()
        e2 = e1.ewm(span=n, adjust=False).mean()
        e3 = e2.ewm(span=n, adjust=False).mean()
        tema = 3 * e1 - 3 * e2 + e3
        return tema
    except Exception as e:
        logger.error(f"TEMA计算失败: {str(e)}")
        raise


def calculate_tema_factor(
    data_manager: DataManager,
    start_date: str,
    end_date: str,
    stock_codes: Optional[List[str]] = None,
    tema_period: int = 20,
    slope_period: int = 5,
    momentum_period: int = 10,
    short_n: int = 10,
    long_n: int = 20,
) -> pd.DataFrame:
    """
    计算基于TEMA的综合技术因子
    
    **因子逻辑**：
    TEMA因子整合了价格相对TEMA的乖离率、TEMA的斜率、TEMA的动量，
    以及短期/长期TEMA的交叉信号等多个维度，生成一个综合技术评分。
    
    **因子计算公式**：
    综合因子 = 乖离率标准化 × 0.4 + 斜率标准化 × 0.3 + 
               动量标准化 × 0.2 + 交叉信号 × 0.1
    
    其中：
    - 乖离率 = (价格 - TEMA) / TEMA × 100
    - 斜率 = (TEMA_t - TEMA_{t-n}) / n
    - 动量 = (TEMA_t - TEMA_{t-n}) / TEMA_{t-n} × 100
    - 交叉信号 = 1 if 短期TEMA > 长期TEMA else -1
    
    **因子方向**：
    - 高因子值 → 技术面强势，适合做多
    - 低因子值 → 技术面弱势，避免或做空
    
    **数据要求**：
    - 至少需要max(tema_period, momentum_period, long_n)个交易日的历史数据
    - 函数自动扩展数据缓冲期以确保数据充足
    
    Parameters
    ----------
    data_manager : DataManager
        数据管理器实例
    start_date : str
        因子计算开始日期，格式 'YYYY-MM-DD'
    end_date : str
        因子计算结束日期，格式 'YYYY-MM-DD'
    stock_codes : Optional[List[str]], default=None
        股票代码列表，如为None则使用所有可用股票
    tema_period : int, default=20
        TEMA计算周期
    slope_period : int, default=5
        斜率计算周期
    momentum_period : int, default=10
        动量计算周期
    short_n : int, default=10
        短期TEMA周期
    long_n : int, default=20
        长期TEMA周期
        
    Returns
    -------
    pd.DataFrame
        MultiIndex (trade_date, ts_code) with single column 'factor'
        - trade_date: 交易日期（datetime类型）
        - ts_code: 股票代码
        - factor: 标准化后的因子值（z-score）
        
    Raises
    ------
    ValueError
        - 无法获取日行情数据
        - 数据缺少必要列
        
    Examples
    --------
    >>> from data_manager.data import DataManager
    >>> data_manager = DataManager()
    >>> 
    >>> # 计算指定股票的TEMA因子
    >>> factor = calculate_tema_factor(
    ...     data_manager=data_manager,
    ...     start_date='2023-01-01',
    ...     end_date='2023-12-31',
    ...     stock_codes=['000001.SZ', '600000.SH']
    ... )
    """
    # 股票池处理
    if stock_codes is None:
        logger.warning("未指定股票池，将使用所有可用股票（可能导致计算缓慢）")
        print("⚠️ 未指定股票池，将使用所有可用股票...")
        all_daily = data_manager.load_data('daily', start_date=start_date, end_date=end_date)
        if all_daily is None or all_daily.empty:
            raise ValueError("无法获取日行情数据以确定股票池")
        stock_codes = all_daily['ts_code'].unique().tolist()
        logger.info(f"自动确定股票池: {len(stock_codes)} 只股票")
    else:
        logger.info(f"使用指定股票池: {len(stock_codes)} 只股票")
    
    # 添加数据缓冲期处理
    buffer_days = max(tema_period, momentum_period, long_n) * 3
    
    try:
        start_date_extended = (pd.to_datetime(start_date) - pd.Timedelta(days=buffer_days)).strftime('%Y-%m-%d')
    except Exception as e:
        logger.error(f"日期解析失败: {e}")
        raise
    
    logger.info(f"{'='*60}")
    logger.info(f"计算 TEMA 综合技术因子")
    logger.info(f"{'='*60}")
    logger.info(f"目标日期范围: {start_date} ~ {end_date}")
    logger.info(f"数据加载范围: {start_date_extended} ~ {end_date} (含缓冲期)")
    logger.info(f"股票池: {len(stock_codes)} 只股票")
    
    print(f"\n{'='*60}")
    print(f"计算 TEMA 综合技术因子")
    print(f"{'='*60}")
    print(f"目标日期范围: {start_date} ~ {end_date}")
    print(f"数据加载范围: {start_date_extended} ~ {end_date} (含缓冲期)")
    print(f"股票池: {len(stock_codes)} 只股票")
    
    # 加载日线数据
    logger.info("开始加载日线数据...")
    print(f"\n加载日线数据...")
    
    try:
        daily = data_manager.load_data(
            'daily',
            start_date=start_date_extended,
            end_date=end_date,
            stock_codes=stock_codes
        )
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        raise
    
    if daily is None or daily.empty:
        logger.error("无法获取日行情数据")
        raise ValueError("无法获取日行情数据")
    
    # 数据清洗和预处理
    logger.info("开始数据清洗和预处理...")
    daily = daily.copy()
    
    # 转换日期格式
    logger.debug("转换日期格式...")
    print(f"转换日期格式...")
    try:
        daily['trade_date'] = pd.to_datetime(daily['trade_date'])
        daily = daily.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    except Exception as e:
        logger.error(f"日期转换失败: {e}")
        raise
    
    # 检查必要字段
    logger.debug("检查必要字段...")
    required_cols = ['close', 'ts_code', 'trade_date']
    missing_cols = [col for col in required_cols if col not in daily.columns]
    if missing_cols:
        logger.error(f"数据缺少必要列: {missing_cols}")
        raise ValueError(f"数据缺少必要列: {missing_cols}")
    
    # 删除缺失值
    daily = daily.dropna(subset=['trade_date', 'close'])
    logger.debug(f"删除缺失值后记录数: {len(daily)}")
    
    # 数据质量检查和异常值处理
    logger.info("开始数据质量检查...")
    print(f"数据质量检查...")
    original_count = len(daily)
    
    # 过滤异常值
    daily = daily[daily['close'] > 0]
    
    filtered_count = len(daily)
    if filtered_count < original_count:
        logger.warning(f"过滤异常值: {original_count - filtered_count} 条")
        print(f"⚠️ 过滤异常值: {original_count - filtered_count} 条")
    else:
        logger.info("未发现异常值")
        print(f"✅ 未发现异常值")
    
    if daily.empty:
        logger.error("过滤后数据为空")
        raise ValueError("过滤后数据为空")
    
    n_stocks = daily['ts_code'].nunique()
    avg_records = len(daily) / n_stocks if n_stocks > 0 else 0
    logger.info(f"数据加载完成: 时间范围 {daily['trade_date'].min()} ~ {daily['trade_date'].max()}, "
                f"{n_stocks} 只股票, {len(daily):,} 条记录, 平均每只 {avg_records:.0f} 条")
    
    print(f"\n数据加载完成:")
    print(f"  时间范围: {daily['trade_date'].min()} ~ {daily['trade_date'].max()}")
    print(f"  股票数量: {n_stocks}")
    print(f"  数据记录: {len(daily):,} 条")
    print(f"  平均每只股票: {avg_records:.0f} 条")
    
    # 按股票分组计算TEMA因子
    logger.info("开始计算 TEMA 指标...")
    print(f"\n开始计算 TEMA 指标...")
    factor_results = []
    failed_stocks = []
    insufficient_data_stocks = []
    
    total_stocks = daily['ts_code'].nunique()
    logger.info(f"待处理股票总数: {total_stocks}")
    
    for idx, code in enumerate(daily['ts_code'].unique(), 1):
        try:
            stock_data = daily[daily['ts_code'] == code].copy()
            
            # 检查数据量是否充足
            min_required = max(tema_period, momentum_period, long_n) + 10
            if len(stock_data) < min_required:
                insufficient_data_stocks.append(code)
                logger.debug(f"[{idx}/{total_stocks}] {code} - 数据不足 ({len(stock_data)} < {min_required})")
                continue
            
            # 计算TEMA乖离率因子
            tema_values = calculate_tema(stock_data['close'], tema_period)
            deviation = (stock_data['close'] - tema_values) / tema_values * 100
            
            # 计算TEMA斜率因子
            slope = (tema_values - tema_values.shift(slope_period)) / slope_period
            
            # 计算TEMA动量因子
            momentum = (tema_values - tema_values.shift(momentum_period)) / tema_values.shift(momentum_period) * 100
            
            # 计算TEMA交叉信号因子
            short_tema = calculate_tema(stock_data['close'], short_n)
            long_tema = calculate_tema(stock_data['close'], long_n)
            cross_signal = np.where(short_tema > long_tema, 1, -1)
            
            # 构建临时DataFrame用于标准化
            temp_df = pd.DataFrame({
                'deviation': deviation.values,
                'slope': slope.values,
                'momentum': momentum.values,
                'cross_signal': cross_signal
            }, index=stock_data.index)
            
            # 对每个子因子进行标准化（仅针对该股票）
            for col in ['deviation', 'slope', 'momentum']:
                col_mean = temp_df[col].mean()
                col_std = temp_df[col].std()
                if col_std > 0:
                    temp_df[col] = (temp_df[col] - col_mean) / col_std
                else:
                    temp_df[col] = 0
            
            # 综合因子 = 加权平均
            factor = (
                temp_df['deviation'] * 0.4 +
                temp_df['slope'] * 0.3 +
                temp_df['momentum'] * 0.2 +
                temp_df['cross_signal'] * 0.1
            )
            
            # 保存结果
            result_df = pd.DataFrame({
                'trade_date': stock_data['trade_date'].values,
                'ts_code': code,
                'factor': factor.values
            })
            
            factor_results.append(result_df)
            
            if idx % 50 == 0 or idx == total_stocks:
                logger.info(f"进度: {idx}/{total_stocks} ({idx/total_stocks*100:.1f}%)")
                print(f"进度: {idx}/{total_stocks} ({idx/total_stocks*100:.1f}%)")
                
        except Exception as e:
            failed_stocks.append(code)
            logger.error(f"[{idx}/{total_stocks}] {code} - 计算失败: {str(e)}")
            continue
    
    # 统计计算结果
    logger.info(f"计算完成 - 成功: {len(factor_results)}, 数据不足: {len(insufficient_data_stocks)}, 失败: {len(failed_stocks)}")
    print(f"\n计算统计:")
    print(f"  成功: {len(factor_results)} 只")
    print(f"  数据不足: {len(insufficient_data_stocks)} 只")
    print(f"  计算失败: {len(failed_stocks)} 只")
    
    if insufficient_data_stocks and len(insufficient_data_stocks) <= 10:
        logger.debug(f"数据不足的股票: {insufficient_data_stocks}")
        print(f"  数据不足股票: {insufficient_data_stocks[:5]}..." if len(insufficient_data_stocks) > 5 else insufficient_data_stocks)
    
    if failed_stocks:
        logger.warning(f"计算失败的股票: {failed_stocks}")
        print(f"  ⚠️ 计算失败股票: {failed_stocks[:5]}..." if len(failed_stocks) > 5 else failed_stocks)
    
    if not factor_results:
        logger.error("没有成功计算任何股票的因子值")
        raise ValueError("没有成功计算任何股票的因子值，请检查数据质量或参数设置")
    
    # 合并结果
    print(f"\n合并因子数据...")
    try:
        factor_df = pd.concat(factor_results, ignore_index=True)
    except Exception as e:
        logger.error(f"合并数据失败: {e}")
        raise
    
    factor_df = factor_df.dropna(subset=['factor'])
    
    if factor_df.empty:
        logger.error("合并后因子数据为空")
        raise ValueError("合并后因子数据为空")
    
    print(f"\n因子计算完成:")
    print(f"  有效股票数: {factor_df['ts_code'].nunique()}")
    print(f"  有效记录数: {len(factor_df):,} 条")
    print(f"  缺失值数量: {factor_df['factor'].isna().sum()}")
    
    # 截面标准化处理
    print(f"进行截面标准化处理...")
    logger.info("进行截面标准化处理...")
    
    try:
        def standardize_factor(group):
            """截面标准化：每日因子值转为z-score"""
            mean = group.mean()
            std = group.std()
            if std > 0:
                return (group - mean) / std
            else:
                return group - mean
        
        factor_df['factor'] = factor_df.groupby('trade_date')['factor'].transform(standardize_factor)
        logger.info("截面标准化完成")
        print(f"✅ 截面标准化完成")
            
    except Exception as e:
        logger.error(f"标准化失败: {e}")
        raise
    
    # 过滤到目标日期范围
    try:
        factor_df = factor_df[
            (factor_df['trade_date'] >= pd.to_datetime(start_date)) &
            (factor_df['trade_date'] <= pd.to_datetime(end_date))
        ]
    except Exception as e:
        logger.error(f"日期过滤失败: {e}")
        raise
    
    if factor_df.empty:
        logger.error("过滤到目标日期范围后数据为空")
        raise ValueError("过滤到目标日期范围后数据为空")
    
    # 设置多重索引
    try:
        result = factor_df.set_index(['trade_date', 'ts_code'])
    except Exception as e:
        logger.error(f"设置索引失败: {e}")
        raise
    
    print(f"\n✅ TEMA 因子计算完成！")
    print(f"  最终记录数: {len(result):,} 条")
    print(f"  覆盖股票数: {result.index.get_level_values('ts_code').nunique()}")
    print(f"  覆盖交易日: {result.index.get_level_values('trade_date').nunique()}")
    
    # 因子统计
    factor_values = result['factor']
    print(f"\n因子统计:")
    print(f"  均值: {factor_values.mean():.4f} (应接近0)")
    print(f"  标准差: {factor_values.std():.4f} (应接近1)")
    print(f"  最小值: {factor_values.min():.4f}")
    print(f"  最大值: {factor_values.max():.4f}")
    print(f"  中位数: {factor_values.median():.4f}")
    
    # 异常值检测
    extreme_values = ((factor_values < -5) | (factor_values > 5)).sum()
    if extreme_values > 0:
        logger.warning(f"发现 {extreme_values} 个极端值 (|z-score| > 5)")
        print(f"  ⚠️ 极端值: {extreme_values} 个 (|z-score| > 5)")
    
    print(f"{'='*60}\n")
    
    return result


def generate_tema_signals(
    factor_data: pd.DataFrame,
    strong_buy_threshold: float = 1.0,
    buy_threshold: float = 0.5,
    sell_threshold: float = -0.5,
    strong_sell_threshold: float = -1.0
) -> pd.DataFrame:
    """
    基于TEMA因子生成多级选股信号
    
    **信号逻辑**：
    根据标准化后的因子值（z-score），将股票划分为5个等级：
    - 强烈买入（2）：因子值 ≥ 1.0σ（前16%）
    - 买入（1）：0.5σ ≤ 因子值 < 1.0σ（16%-31%）
    - 中性（0）：-0.5σ < 因子值 < 0.5σ（中间38%）
    - 卖出（-1）：-1.0σ < 因子值 ≤ -0.5σ（31%-16%）
    - 强烈卖出（-2）：因子值 ≤ -1.0σ（后16%）
    
    Parameters
    ----------
    factor_data : pd.DataFrame
        TEMA因子数据，MultiIndex (trade_date, ts_code)
    strong_buy_threshold : float, default=1.0
        强烈买入信号阈值
    buy_threshold : float, default=0.5
        买入信号阈值
    sell_threshold : float, default=-0.5
        卖出信号阈值
    strong_sell_threshold : float, default=-1.0
        强烈卖出信号阈值
        
    Returns
    -------
    pd.DataFrame
        包含信号的DataFrame，列包括：
        - factor: 原始因子值
        - signal: 数值信号（-2 到 2）
        - signal_label: 中文信号标签
    """
    logger.info(f"开始生成 TEMA 信号 - 阈值: 强买={strong_buy_threshold}, 买={buy_threshold}, 卖={sell_threshold}, 强卖={strong_sell_threshold}")
    
    if not isinstance(factor_data, pd.DataFrame):
        raise TypeError("factor_data 必须是 DataFrame 类型")
    
    if factor_data.empty:
        logger.warning("输入因子数据为空")
        return pd.DataFrame()
    
    # 复制数据
    signals = factor_data.copy()
    
    if 'factor' not in signals.columns:
        raise ValueError("factor_data 必须包含 'factor' 列")
    
    # 生成数值信号
    def categorize_signal(factor_value):
        if pd.isna(factor_value):
            return 0
        elif factor_value >= strong_buy_threshold:
            return 2
        elif factor_value >= buy_threshold:
            return 1
        elif factor_value <= strong_sell_threshold:
            return -2
        elif factor_value <= sell_threshold:
            return -1
        else:
            return 0
    
    try:
        signals['signal'] = signals['factor'].apply(categorize_signal)
    except Exception as e:
        logger.error(f"信号分类失败: {e}")
        raise
    
    # 生成信号标签
    signal_labels = {
        2: '强烈买入',
        1: '买入',
        0: '中性',
        -1: '卖出',
        -2: '强烈卖出'
    }
    signals['signal_label'] = signals['signal'].map(signal_labels)
    
    # 统计信号分布
    signal_counts = signals['signal'].value_counts().sort_index()
    logger.info(f"信号分布: {signal_counts.to_dict()}")
    print(f"\n信号分布:")
    for sig, label in signal_labels.items():
        count = signal_counts.get(sig, 0)
        pct = count / len(signals) * 100 if len(signals) > 0 else 0
        print(f"  {label}({sig}): {count} ({pct:.1f}%)")
    
    return signals


def get_top_stocks(
    signals: pd.DataFrame,
    date: str,
    top_n: int = 10,
    signal_filter: int = 1
) -> pd.DataFrame:
    """
    获取指定日期的Top N推荐股票
    
    Parameters
    ----------
    signals : pd.DataFrame
        信号数据，MultiIndex (trade_date, ts_code)
    date : str
        查询日期，格式 'YYYY-MM-DD'
    top_n : int, default=10
        返回的股票数量
    signal_filter : int, default=1
        信号筛选阈值，只返回信号 >= signal_filter 的股票
        
    Returns
    -------
    pd.DataFrame
        Top N股票列表，按因子值降序排列
    """
    try:
        date_dt = pd.to_datetime(date)
        
        # 提取指定日期的数据
        if isinstance(signals.index, pd.MultiIndex):
            date_signals = signals.xs(date_dt, level='trade_date')
        else:
            date_signals = signals[signals.index == date_dt]
        
        if date_signals.empty:
            logger.warning(f"日期 {date} 无数据")
            print(f"⚠️ 日期 {date} 无数据")
            return pd.DataFrame()
        
        # 筛选信号
        filtered = date_signals[date_signals['signal'] >= signal_filter]
        
        # 按因子值排序
        top_stocks = filtered.nlargest(top_n, 'factor')
        
        logger.info(f"日期 {date} 推荐 {len(top_stocks)} 只股票")
        print(f"\n📊 {date} Top {len(top_stocks)} 推荐股票:")
        print(top_stocks[['factor', 'signal', 'signal_label']])
        
        return top_stocks
        
    except Exception as e:
        logger.error(f"获取Top股票失败: {e}")
        raise


def run_tema_factor_backtest(
    start_date: str = '2023-01-01',
    end_date: str = '2023-12-31',
    stock_codes: Optional[List[str]] = None,
    tema_period: int = 20,
    slope_period: int = 5,
    momentum_period: int = 10,
    short_n: int = 10,
    long_n: int = 20,
    rebalance_freq: str = 'weekly',
    transaction_cost: float = 0.0003,
) -> dict:
    """
    运行TEMA因子回测
    
    **策略说明**：
    - 采用Long-Only策略
    - 每日等权持有所有因子值为正的股票
    - 定期调仓
    
    Parameters
    ----------
    start_date, end_date : str
        回测周期
    stock_codes : Optional[List[str]]
        股票池
    tema_period : int, default=20
        TEMA计算周期
    slope_period : int, default=5
        斜率计算周期
    momentum_period : int, default=10
        动量计算周期
    short_n : int, default=10
        短期TEMA周期
    long_n : int, default=20
        长期TEMA周期
    rebalance_freq : str, default='weekly'
        调仓频率: 'daily', 'weekly', 'monthly'
    transaction_cost : float, default=0.0003
        单边交易成本
        
    Returns
    -------
    dict
        包含回测结果的字典:
        - factor_data: 因子数据
        - portfolio_returns: 组合收益
        - performance_metrics: 业绩指标
        - analysis_results: IC分析结果
    """
    print("=" * 60)
    print("TEMA 技术因子回测")
    print("=" * 60)
    print(f"\n回测配置:")
    print(f"  时间范围: {start_date} ~ {end_date}")
    print(f"  TEMA周期: {tema_period}")
    print(f"  调仓频率: {rebalance_freq}")
    print(f"  交易成本: {transaction_cost:.4f}")
    
    # 初始化数据管理器
    data_manager = DataManager()
    
    # 计算因子
    factor_data = calculate_tema_factor(
        data_manager=data_manager,
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_codes,
        tema_period=tema_period,
        slope_period=slope_period,
        momentum_period=momentum_period,
        short_n=short_n,
        long_n=long_n,
    )
    
    if factor_data.empty:
        print("⚠️ 因子数据为空，无法回测")
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
        raise ValueError("无法加载用于回测的股票数据")
    
    stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'])
    stock_data = stock_data.sort_values(['ts_code', 'trade_date'])
    
    # 计算下一日收益率
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
    
    # Long-Only策略：等权持有所有有因子值的股票
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
    
    # IC分析
    ic_series = None
    ic_mean = None
    ic_std = None
    icir = None
    ic_positive_ratio = None
    
    try:
        ic_list = []
        for date in combined['trade_date'].unique():
            date_data = combined[combined['trade_date'] == date]
            if len(date_data) >= 10:
                correlation = date_data[['factor', 'next_return']].corr(method='spearman').iloc[0, 1]
                if not np.isnan(correlation):
                    ic_list.append({'trade_date': date, 'ic': correlation})
        
        if ic_list:
            ic_series = pd.DataFrame(ic_list).set_index('trade_date')['ic']
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            icir = ic_mean / ic_std if ic_std > 0 else 0
            ic_positive_ratio = (ic_series > 0).mean()
    except Exception as e:
        logger.error(f"IC计算失败: {e}")
        print(f"⚠️ IC计算失败: {e}")
    
    # 打印结果
    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    
    print(f"\n📊 业绩指标:")
    print(f"  总收益率: {total_return:.2%}")
    print(f"  年化收益率: {annualized_return:.2%}")
    print(f"  年化波动率: {volatility:.2%}")
    print(f"  夏普比率: {sharpe_ratio:.2f}")
    print(f"  最大回撤: {max_drawdown:.2%}")
    
    if ic_series is not None:
        print(f"\n📊 IC分析:")
        print(f"  IC均值: {ic_mean:.4f}")
        print(f"  IC标准差: {ic_std:.4f}")
        print(f"  ICIR: {icir:.4f}")
        print(f"  IC>0占比: {ic_positive_ratio:.2%}")
    
    print(f"\n📈 因子覆盖:")
    print(f"  有效因子记录数: {len(factor_data)}")
    print(f"  覆盖股票数: {factor_data.index.get_level_values('ts_code').nunique()}")
    print(f"  覆盖交易日数: {factor_data.index.get_level_values('trade_date').nunique()}")
    
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
    """主函数：演示TEMA因子计算和回测"""
    print("=" * 60)
    print("TEMA 技术因子演示")
    print("=" * 60)
    
    try:
        # 配置参数
        config = {
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'tema_period': 20,
            'slope_period': 5,
            'momentum_period': 10,
            'short_n': 10,
            'long_n': 20,
            'rebalance_freq': 'weekly',
            'transaction_cost': 0.0003,
        }
        
        print("\n回测配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # 运行回测
        results = run_tema_factor_backtest(**config)
        
        if results['factor_data'] is not None:
            # 生成信号
            print("\n" + "=" * 60)
            print("生成选股信号")
            print("=" * 60)
            signals = generate_tema_signals(results['factor_data'])
            
            # 获取最新一天的推荐股票
            latest_date = results['factor_data'].index.get_level_values('trade_date').max()
            top_stocks = get_top_stocks(signals, date=latest_date.strftime('%Y-%m-%d'), top_n=10)
        
        print("\n✅ 回测完成！")
        
    except Exception as e:
        print(f"\n❌ 运行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
