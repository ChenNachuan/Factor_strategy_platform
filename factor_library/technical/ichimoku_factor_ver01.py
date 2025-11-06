"""
Ichimoku云图（一目均衡表）技术因子

本模块实现了基于日本传统技术分析工具Ichimoku云图的综合技术因子，
包括因子计算、回测分析、选股信号生成等完整功能。

日志配置：
- 默认级别：INFO
- 可通过 setup_logger() 修改日志级别和输出格式
- 支持文件输出和控制台输出

**主要功能**：
1. calculate_ichimoku_factor(): 计算Ichimoku综合技术因子
2. run_ichimoku_factor_backtest(): 运行因子策略回测
3. generate_ichimoku_signals(): 生成多级选股信号
4. get_top_stocks(): 获取Top N推荐股票
5. generate_trading_signals(): 生成每日信号统计

**Ichimoku云图简介**：
一目均衡表（Ichimoku Kinko Hyo）由日本分析师细田悟一于1930年代开发，
是一种综合性技术分析工具，能够同时反映价格趋势、支撑阻力、动量等多个维度。

核心组成部分：
- 转换线（Tenkan-sen）：9日最高最低价中值，短期趋势
- 基准线（Kijun-sen）：26日最高最低价中值，中期趋势
- 先行带A（Senkou Span A）：转换线和基准线的均值，前移26日
- 先行带B（Senkou Span B）：52日最高最低价中值，前移26日
- 云图（Kumo）：先行带A和B之间的区域，表示支撑/阻力区
- 迟行线（Chikou Span）：收盘价后移26日（本实现中未使用）

**因子构建逻辑**：
本因子整合了四个维度：
1. 价格位置（40%）：价格相对于云图的位置
2. 云趋势（30%）：云的方向（上升云vs下降云）
3. TK交叉（20%）：转换线和基准线的交叉关系
4. 云宽度动能（10%）：云厚度的变化趋势

综合评分后进行截面标准化，生成z-score形式的因子值。

**使用示例**：

基础用法：
>>> from data_manager.data import DataManager
>>> data_manager = DataManager()
>>> 
>>> # 计算因子
>>> factor = calculate_ichimoku_factor(
...     data_manager,
...     start_date='2023-01-01',
...     end_date='2023-12-31'
... )
>>> 
>>> # 生成信号
>>> signals = generate_ichimoku_signals(factor)
>>> 
>>> # 获取推荐股票
>>> top_stocks = get_top_stocks(signals, date='2023-12-31', top_n=10)

完整回测：
>>> results = run_ichimoku_factor_backtest(
...     start_date='2023-01-01',
...     end_date='2023-12-31',
...     rebalance_freq='weekly'
... )
>>> print(results['performance_metrics'])

**因子特点**：
- 优势：
  * 多维度综合评分，减少单一指标的假信号
  * 同时考虑趋势、动量、支撑阻力
  * 适合中长期趋势跟踪策略
  * 视觉化强，易于理解和验证

- 局限：
  * 在震荡市中可能频繁调整
  * 需要较长的历史数据（至少78个交易日）
  * 滞后性：先行带前移导致一定滞后
  * 参数固定（9、26、52），不适应所有市场环境

**数据要求**：
- 必需字段：trade_date, ts_code, high, low, close
- 最小数据量：52个交易日（建议扩展至78日以上）
- 数据质量：需要清洗异常值（收盘价≤0等）

**性能考虑**：
- 因子计算使用循环处理每只股票（可能较慢）
- 建议限制股票池大小（如沪深300、中证500）
- 大规模计算建议使用并行处理

**版本历史**：
- v1.0: 基础Ichimoku因子实现
- v1.1: 添加数据缓冲期处理
- v1.2: 完善回测函数逻辑
- v1.3: 添加详细文档和选股信号功能
- v1.4: 添加详细日志输出功能

作者：量化投资团队
日期：2024-11-06
参考：《Technical Analysis of the Financial Markets》by John J. Murphy
"""

import logging
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Optional, List

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
        日志级别，可选值：
        - logging.DEBUG: 详细调试信息
        - logging.INFO: 一般信息
        - logging.WARNING: 警告信息
        - logging.ERROR: 错误信息
        - logging.CRITICAL: 严重错误
    
    log_file : str, optional
        日志文件路径，如果为None则不输出到文件
    
    console : bool, default=True
        是否输出到控制台
    
    file_mode : str, default='a'
        文件写入模式，'a'追加，'w'覆盖
    
    Examples
    --------
    >>> # 基础配置（仅控制台INFO级别）
    >>> setup_logger()
    >>> 
    >>> # 详细调试模式
    >>> setup_logger(level=logging.DEBUG)
    >>> 
    >>> # 同时输出到文件
    >>> setup_logger(
    ...     level=logging.INFO,
    ...     log_file='ichimoku_factor.log',
    ...     console=True
    ... )
    """
    # 清除已有的handlers
    logger.handlers.clear()
    logger.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 添加控制台handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 添加文件handler
    if log_file:
        file_handler = logging.FileHandler(log_file, mode=file_mode, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"日志将输出到文件: {log_file}")


# 默认初始化日志（INFO级别，仅控制台）
setup_logger()

# 路径配置
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_manager.data import DataManager

def calculate_ichimoku_factor(
    data_manager: DataManager,
    start_date: str,
    end_date: str,
    stock_codes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    计算基于Ichimoku云图（一目均衡表）的综合技术因子。
    
    **因子逻辑**：
    Ichimoku云图是一种综合性技术分析工具，由日本分析师细田悟一于1930年代开发。
    该因子整合了价格相对云图的位置、云的趋势方向、短期/中期均线交叉、
    以及云宽度动能等多个维度，生成一个综合技术评分。
    
    **核心组成部分**：
    1. 转换线（Tenkan-sen）：9日最高最低价中值
    2. 基准线（Kijun-sen）：26日最高最低价中值
    3. 先行带A（Senkou Span A）：(转换线+基准线)/2，前移26日
    4. 先行带B（Senkou Span B）：52日最高最低价中值，前移26日
    5. 云图（Kumo）：先行带A和B之间的区域
    
    **因子计算公式**：
    综合因子 = 价格位置 × 0.4 + 云趋势 × 0.3 + TK交叉 × 0.2 + 云宽度动能 × 0.1
    
    其中：
    - 价格位置：价格在云上方=2，价格在云中=1，价格在云下方=0
    - 云趋势：先行带A > 先行带B（上升云）=1，否则=0
    - TK交叉：转换线 > 基准线（金叉）=1，否则=0
    - 云宽度动能：当前云宽度 > 20日均值=1，否则=0
    
    **因子方向**：
    - 高因子值 → 技术面强势，适合做多
    - 低因子值 → 技术面弱势，避免或做空
    
    **数据要求**：
    - 至少需要52个交易日的历史数据（计算先行带B）
    - 再加上26日前移，实际需要78个交易日
    - 函数自动扩展156天缓冲期以确保数据充足
    
    **因子特点**：
    - 多维度综合评分，减少单一指标的假信号
    - 适合趋势性市场，在震荡市中可能频繁调整
    - 因子值经过截面标准化，均值为0，标准差为1
    
    Parameters
    ----------
    data_manager : DataManager
        数据管理器实例，用于加载行情数据
    start_date : str
        因子计算开始日期，格式 'YYYY-MM-DD'
        注：函数会自动向前扩展156天以确保数据充足
    end_date : str
        因子计算结束日期，格式 'YYYY-MM-DD'
    stock_codes : Optional[List[str]], default=None
        股票代码列表，如 ['000001.SZ', '600000.SH']
        如果为 None，则使用所有可用股票（可能导致计算缓慢）

    Returns
    -------
    pd.DataFrame
        MultiIndex (trade_date, ts_code) with single column 'factor'
        - trade_date: 交易日期（datetime类型）
        - ts_code: 股票代码
        - factor: 标准化后的因子值（z-score），范围通常在[-3, 3]
        
    Raises
    ------
    ValueError
        - 无法获取日行情数据
        - 数据缺少必要列（high, low, close）
        - 没有计算出有效的因子值
        
    Notes
    -----
    - 因子值已进行截面标准化：每个交易日内，均值=0，标准差=1
    - 数据量不足52日的股票会被自动过滤
    - 异常值（收盘价≤0，最高价<最低价）会被过滤
    - 返回的数据仅包含目标日期范围内的记录
    
    Examples
    --------
    >>> from data_manager.data import DataManager
    >>> data_manager = DataManager()
    >>> 
    >>> # 计算指定股票的Ichimoku因子
    >>> factor = calculate_ichimoku_factor(
    ...     data_manager=data_manager,
    ...     start_date='2023-01-01',
    ...     end_date='2023-12-31',
    ...     stock_codes=['000001.SZ', '000002.SZ', '600000.SH']
    ... )
    >>> 
    >>> # 查看因子统计
    >>> print(factor['factor'].describe())
    >>> 
    >>> # 查看某只股票的因子时间序列
    >>> stock_factor = factor.xs('000001.SZ', level='ts_code')
    >>> print(stock_factor.head())
    
    References
    ----------
    - Hosoda, Goichi (1996). "Ichimoku Kinko Studies"
    - 一目均衡表：日本传统技术分析的精髓
    - 适用于趋势跟踪和动量策略
    """
    # 股票池处理
    if stock_codes is None:
        print("未指定股票池，尝试获取全市场股票...")
        try:
            all_daily = data_manager.load_data('daily', start_date=start_date, end_date=end_date, cleaned=True)
            if all_daily is None or all_daily.empty:
                print("⚠️ 无法获取全市场数据，使用默认股票池")
                stock_codes = ['000001.SZ', '000002.SZ', '000858.SZ', '600000.SH', '600036.SH', '600519.SH']
            else:
                stock_codes = all_daily['ts_code'].unique().tolist()
                print(f"✅ 成功获取全市场股票池: {len(stock_codes)} 只")
        except Exception as e:
            print(f"⚠️ 加载全市场数据失败: {str(e)}")
            print("使用默认股票池")
            stock_codes = ['000001.SZ', '000002.SZ', '000858.SZ', '600000.SH', '600036.SH', '600519.SH']
    else:
        if not isinstance(stock_codes, list):
            raise TypeError(f"stock_codes 必须是列表类型，当前类型: {type(stock_codes)}")
        if len(stock_codes) == 0:
            raise ValueError("stock_codes 不能为空列表")
        print(f"使用指定股票池: {len(stock_codes)} 只股票")

    # 添加数据缓冲期处理
    # Ichimoku需要52日数据，先行带还要前移26日，所以需要至少78个交易日的历史数据
    # 为保险起见，向前扩展52*3=156天（约6个月）
    buffer_days = 52 * 3
    
    try:
        start_date_extended = (pd.to_datetime(start_date) - pd.Timedelta(days=buffer_days)).strftime('%Y-%m-%d')
        logger.debug(f"计算缓冲期: 向前扩展 {buffer_days} 天")
    except Exception as e:
        logger.error(f"日期格式转换失败: {str(e)}")
        raise ValueError(f"日期格式错误，start_date 应为 'YYYY-MM-DD' 格式: {str(e)}")
    
    logger.info(f"{'='*60}")
    logger.info(f"计算 Ichimoku 云图因子")
    logger.info(f"{'='*60}")
    logger.info(f"目标日期范围: {start_date} ~ {end_date}")
    logger.info(f"数据加载范围: {start_date_extended} ~ {end_date} (含缓冲期)")
    logger.info(f"股票池: {len(stock_codes)} 只股票")
    
    print(f"\n{'='*60}")
    print(f"计算 Ichimoku 云图因子")
    print(f"{'='*60}")
    print(f"目标日期范围: {start_date} ~ {end_date}")
    print(f"数据加载范围: {start_date_extended} ~ {end_date} (含缓冲期)")
    print(f"股票池: {len(stock_codes)} 只股票")

    # 加载日线数据（使用扩展后的开始日期）
    logger.info("开始加载日线数据...")
    print(f"\n加载日线数据...")
    try:
        daily = data_manager.load_data('daily', start_date=start_date_extended, end_date=end_date, stock_codes=stock_codes)
        logger.info(f"日线数据加载成功: {len(daily)} 条记录")
    except Exception as e:
        logger.error(f"加载日线数据失败: {str(e)}", exc_info=True)
        raise RuntimeError(f"加载日线数据失败: {str(e)}")
    
    if daily is None or daily.empty:
        logger.error("获取的日行情数据为空")
        raise ValueError(f'无法获取日行情数据，请检查：\n'
                        f'  1. 日期范围是否正确: {start_date_extended} ~ {end_date}\n'
                        f'  2. 股票代码是否有效: {stock_codes[:5]}...\n'
                        f'  3. 数据源是否可用')

    # 数据清洗和预处理
    logger.info("开始数据清洗和预处理...")
    daily = daily.copy()
    
    # 转换日期格式
    logger.debug("转换日期格式...")
    print(f"转换日期格式...")
    try:
        daily['trade_date'] = pd.to_datetime(daily['trade_date'], errors='coerce')
        if daily['trade_date'].isna().any():
            logger.warning("部分日期格式不正确，尝试使用 %Y%m%d 格式")
            # 尝试使用不同的日期格式
            daily['trade_date'] = pd.to_datetime(daily['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
        
        # 检查是否还有无效日期
        invalid_dates = daily['trade_date'].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"发现 {invalid_dates} 条无效日期记录，将被过滤")
            print(f"⚠️ 警告: 发现 {invalid_dates} 条无效日期记录，将被过滤")
            daily = daily.dropna(subset=['trade_date'])
            
        if daily.empty:
            logger.error("日期转换后数据为空")
            raise ValueError("日期转换后数据为空，请检查日期格式")
        
        logger.info(f"日期转换完成，有效记录: {len(daily)} 条")
    except Exception as e:
        logger.error(f"日期格式转换失败: {str(e)}", exc_info=True)
        raise ValueError(f"日期格式转换失败: {str(e)}")
    
    # 检查必要字段
    logger.debug("检查必要字段...")
    required_cols = ['high', 'low', 'close', 'ts_code']
    missing_cols = [col for col in required_cols if col not in daily.columns]
    if missing_cols:
        logger.error(f"数据缺少必要列: {missing_cols}")
        raise ValueError(f'数据缺少必要列: {missing_cols}\n'
                        f'当前列: {list(daily.columns)}\n'
                        f'请确保数据源包含: {required_cols}')
    
    # 删除缺失值
    daily = daily.dropna(subset=['trade_date', 'high', 'low', 'close'])
    logger.debug(f"删除缺失值后记录数: {len(daily)}")
    
    # 数据质量检查和异常值处理
    logger.info("开始数据质量检查...")
    print(f"数据质量检查...")
    original_count = len(daily)
    
    # 过滤异常值
    daily = daily[daily['close'] > 0]
    daily = daily[daily['high'] >= daily['low']]
    daily = daily[daily['high'] > 0]
    daily = daily[daily['low'] > 0]
    
    filtered_count = len(daily)
    if filtered_count < original_count:
        filtered_ratio = (original_count - filtered_count) / original_count * 100
        logger.warning(f"过滤异常值: {original_count - filtered_count} 条 ({filtered_ratio:.1f}%)")
        print(f"⚠️ 过滤异常值: {original_count - filtered_count} 条 "
              f"({filtered_ratio:.1f}%)")
    else:
        logger.info("数据质量良好，无异常值")
    
    if daily.empty:
        logger.error("数据质量检查后无有效数据")
        raise ValueError("数据质量检查后无有效数据，请检查数据源质量")
    
    n_stocks = daily['ts_code'].nunique()
    avg_records = len(daily) / n_stocks
    logger.info(f"数据加载完成: 时间范围 {daily['trade_date'].min()} ~ {daily['trade_date'].max()}, "
                f"{n_stocks} 只股票, {len(daily):,} 条记录, 平均每只 {avg_records:.0f} 条")
    
    print(f"\n数据加载完成:")
    print(f"  时间范围: {daily['trade_date'].min()} ~ {daily['trade_date'].max()}")
    print(f"  股票数量: {n_stocks}")
    print(f"  数据记录: {len(daily):,} 条")
    print(f"  平均每只股票: {avg_records:.0f} 条")

    # 按股票分组计算Ichimoku因子
    print(f"\n开始计算 Ichimoku 指标...")
    factor_results = []
    failed_stocks = []
    insufficient_data_stocks = []
    # 按股票分组计算Ichimoku因子
    logger.info("开始计算 Ichimoku 指标...")
    print(f"\n开始计算 Ichimoku 指标...")
    factor_results = []
    failed_stocks = []
    insufficient_data_stocks = []
    
    total_stocks = daily['ts_code'].nunique()
    logger.info(f"待处理股票总数: {total_stocks}")
    
    for idx, code in enumerate(daily['ts_code'].unique(), 1):
        try:
            stock_data = daily[daily['ts_code'] == code].sort_values('trade_date')
            
            # 检查数据量是否充足
            if len(stock_data) < 52:
                insufficient_data_stocks.append((code, len(stock_data)))
                logger.debug(f"{code}: 数据不足 ({len(stock_data)} < 52)")
                if idx % 100 == 0 or idx == total_stocks:
                    print(f"  进度: {idx}/{total_stocks} ({idx/total_stocks*100:.1f}%) - 当前: {code} (数据不足)")
                continue
            
            logger.debug(f"{code}: 开始计算 Ichimoku 指标 (数据量: {len(stock_data)})")
            
            # 计算Ichimoku指标
            df = pd.DataFrame({
                'High': stock_data['high'].values,
                'Low': stock_data['low'].values,
                'Close': stock_data['close'].values
            })
            
            # 基础指标计算
            tenkan_sen = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
            kijun_sen = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            senkou_span_b = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
            
            # 检查是否有有效值
            if senkou_span_a.isna().all() or senkou_span_b.isna().all():
                insufficient_data_stocks.append((code, len(stock_data)))
                logger.debug(f"{code}: 指标计算后无有效值")
                continue
            
            logger.debug(f"{code}: Ichimoku 基础指标计算完成")
            
            # 因子计算
            price_position = np.where(
                df['Close'] > senkou_span_a,
                np.where(df['Close'] > senkou_span_b, 2, 1),
                0
            )
            cloud_trend = (senkou_span_a > senkou_span_b).astype(int)
            tk_cross = np.where(tenkan_sen > kijun_sen, 1, -1)
            cloud_width = abs(senkou_span_a - senkou_span_b)
            cloud_width_momentum = cloud_width / cloud_width.rolling(window=20).mean()
            
            # 综合因子
            combined_factor = (
                price_position * 0.4 +
                cloud_trend * 0.3 +
                (tk_cross == 1).astype(int) * 0.2 +
                (cloud_width_momentum > 1).astype(int) * 0.1
            )
            
            logger.debug(f"{code}: 综合因子计算完成 (有效值: {(~np.isnan(combined_factor)).sum()})")
            
            # 保存结果
            factor_data = pd.DataFrame({
                'trade_date': stock_data['trade_date'].values,
                'ts_code': code,
                'factor': combined_factor
            })
            factor_results.append(factor_data)
            
            # 进度提示
            if idx % 100 == 0 or idx == total_stocks:
                print(f"  进度: {idx}/{total_stocks} ({idx/total_stocks*100:.1f}%) - 当前: {code}")
                
        except Exception as e:
            failed_stocks.append((code, str(e)))
            logger.error(f"{code}: 计算失败 - {str(e)}", exc_info=False)
            if idx % 100 == 0 or idx == total_stocks:
                print(f"  进度: {idx}/{total_stocks} ({idx/total_stocks*100:.1f}%) - ⚠️ {code} 计算失败")
            continue
    
    # 统计计算结果
    logger.info(f"计算完成 - 成功: {len(factor_results)}, 数据不足: {len(insufficient_data_stocks)}, 失败: {len(failed_stocks)}")
    print(f"\n计算统计:")
    print(f"  成功: {len(factor_results)} 只")
    print(f"  数据不足: {len(insufficient_data_stocks)} 只")
    print(f"  计算失败: {len(failed_stocks)} 只")
    
    if insufficient_data_stocks and len(insufficient_data_stocks) <= 10:
        print(f"\n数据不足的股票 (前10只):")
        for code, count in insufficient_data_stocks[:10]:
            print(f"    {code}: {count} 条记录 (需要至少52条)")
    
    if failed_stocks:
        print(f"\n⚠️ 计算失败的股票:")
        for code, error in failed_stocks[:5]:  # 只显示前5个错误
            print(f"    {code}: {error}")
        if len(failed_stocks) > 5:
            print(f"    ... 还有 {len(failed_stocks) - 5} 只股票计算失败")
    
    if not factor_results:
        raise ValueError(f'没有计算出有效的因子值\n'
                        f'  总股票数: {total_stocks}\n'
                        f'  数据不足: {len(insufficient_data_stocks)}\n'
                        f'  计算失败: {len(failed_stocks)}\n'
                        f'建议：\n'
                        f'  1. 检查股票代码是否正确\n'
                        f'  2. 确保数据时间跨度足够（至少78个交易日）\n'
                        f'  3. 检查数据源完整性')
    
    # 合并结果
    print(f"\n合并因子数据...")
    try:
        factor_df = pd.concat(factor_results, ignore_index=True)
    except Exception as e:
        raise RuntimeError(f"合并因子数据失败: {str(e)}")
    
    factor_df = factor_df.dropna(subset=['factor'])
    
    if factor_df.empty:
        raise ValueError("因子数据在删除缺失值后为空")
    
    print(f"\n因子计算完成:")
    print(f"  有效股票数: {factor_df['ts_code'].nunique()}")
    print(f"  有效记录数: {len(factor_df):,} 条")
    print(f"  缺失值数量: {factor_df['factor'].isna().sum()}")
    
    # 标准化处理
    print(f"进行截面标准化处理...")
    try:
        def safe_standardize(x):
            """安全的标准化函数，处理标准差为0的情况"""
            if len(x) == 0:
                return x
            mean_val = x.mean()
            std_val = x.std()
            if std_val > 0:
                return (x - mean_val) / std_val
            else:
                # 标准差为0，所有值相同，返回0
                return pd.Series(0, index=x.index)
        
        factor_df['factor'] = factor_df.groupby('trade_date')['factor'].transform(safe_standardize)
        
        # 检查标准化结果
        inf_count = np.isinf(factor_df['factor']).sum()
        if inf_count > 0:
            print(f"⚠️ 警告: 发现 {inf_count} 个无穷值，将被替换为NaN")
            factor_df['factor'].replace([np.inf, -np.inf], np.nan, inplace=True)
            factor_df = factor_df.dropna(subset=['factor'])
            
    except Exception as e:
        raise RuntimeError(f"标准化处理失败: {str(e)}")
    
    # 过滤到目标日期范围
    try:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        factor_df = factor_df[factor_df['trade_date'] >= start_dt]
        factor_df = factor_df[factor_df['trade_date'] <= end_dt]
    except Exception as e:
        raise ValueError(f"日期过滤失败: {str(e)}")
    
    if factor_df.empty:
        raise ValueError(f"过滤到目标日期范围后数据为空\n"
                        f"  目标范围: {start_date} ~ {end_date}\n"
                        f"  实际范围: {factor_df['trade_date'].min() if not factor_df.empty else 'N/A'} ~ "
                        f"{factor_df['trade_date'].max() if not factor_df.empty else 'N/A'}")
    
    # 设置多重索引
    try:
        factor_df = factor_df.set_index(['trade_date', 'ts_code'])['factor']
        result = factor_df.to_frame()
    except Exception as e:
        raise RuntimeError(f"设置索引失败: {str(e)}")
    
    print(f"\n✅ Ichimoku 因子计算完成！")
    print(f"  最终记录数: {len(result):,} 条")
    print(f"  覆盖股票数: {result.index.get_level_values('ts_code').nunique()}")
    print(f"  覆盖交易日: {result.index.get_level_values('trade_date').nunique()}")
    
    # 因子统计（添加异常检测）
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
        print(f"  ⚠️ 极端值 (|z|>5): {extreme_values} 个 ({extreme_values/len(factor_values)*100:.2f}%)")
    
    print(f"{'='*60}\n")
    
    return result

def generate_ichimoku_signals(
    factor_data: pd.DataFrame,
    strong_buy_threshold: float = 1.0,
    buy_threshold: float = 0.5,
    sell_threshold: float = -0.5,
    strong_sell_threshold: float = -1.0
) -> pd.DataFrame:
    """
    基于Ichimoku因子生成多级选股信号。
    
    **信号逻辑**：
    根据标准化后的因子值（z-score），将股票划分为5个等级：
    - 强烈买入（2）：因子值 ≥ 1.0σ（前16%）
    - 买入（1）：0.5σ ≤ 因子值 < 1.0σ（16%-31%）
    - 中性（0）：-0.5σ < 因子值 < 0.5σ（中间38%）
    - 卖出（-1）：-1.0σ < 因子值 ≤ -0.5σ（31%-16%）
    - 强烈卖出（-2）：因子值 ≤ -1.0σ（后16%）
    
    **使用场景**：
    - 强烈买入：技术面极度强势，价格显著高于云图，云趋势明确向上
    - 买入：技术面偏强，适合建仓或加仓
    - 中性：技术面不明朗，观望为主
    - 卖出：技术面转弱，考虑减仓
    - 强烈卖出：技术面极度看空，建议离场
    
    Parameters
    ----------
    factor_data : pd.DataFrame
        Ichimoku因子数据，必须包含 'factor' 列
        通常是 calculate_ichimoku_factor() 的返回值
        Index: MultiIndex (trade_date, ts_code)
    strong_buy_threshold : float, default=1.0
        强烈买入信号阈值（标准化后的因子值）
        默认1.0表示1个标准差以上
    buy_threshold : float, default=0.5
        买入信号阈值
        默认0.5表示0.5个标准差以上
    sell_threshold : float, default=-0.5
        卖出信号阈值
        默认-0.5表示-0.5个标准差以下
    strong_sell_threshold : float, default=-1.0
        强烈卖出信号阈值
        默认-1.0表示-1个标准差以下
    
    Returns
    -------
    pd.DataFrame
        包含信号的DataFrame，Index 与输入相同，列包括：
        - factor: 原始因子值（保留）
        - signal: 数值信号（-2 到 2 的整数）
        - signal_label: 中文信号标签
        
    Raises
    ------
    ValueError
        - factor_data 不是 DataFrame 类型
        - factor_data 缺少 'factor' 列
        
    Notes
    -----
    - 阈值可根据策略风格调整：
      * 保守策略：提高买入阈值（如1.5），降低假阳性
      * 积极策略：降低买入阈值（如0.3），增加候选股票
    - 因子值为 NaN 时，信号为中性（0）
    
    Examples
    --------
    >>> # 使用默认阈值生成信号
    >>> signals = generate_ichimoku_signals(factor_data)
    >>> 
    >>> # 查看信号分布
    >>> print(signals['signal'].value_counts())
    >>> 
    >>> # 筛选买入及以上信号的股票
    >>> buy_signals = signals[signals['signal'] >= 1]
    >>> 
    >>> # 使用更严格的阈值
    >>> strict_signals = generate_ichimoku_signals(
    ...     factor_data,
    ...     strong_buy_threshold=1.5,
    ...     buy_threshold=1.0
    ... )
    
    See Also
    --------
    get_top_stocks : 获取指定日期的Top N选股列表
    generate_trading_signals : 生成每日信号统计报告
    """
    logger.info(f"开始生成 Ichimoku 信号 - 阈值: 强买={strong_buy_threshold}, 买={buy_threshold}, 卖={sell_threshold}, 强卖={strong_sell_threshold}")
    
    if not isinstance(factor_data, pd.DataFrame):
        logger.error(f"factor_data 类型错误: {type(factor_data)}")
        raise TypeError(f"factor_data必须是DataFrame类型，当前类型: {type(factor_data)}")
    
    if factor_data.empty:
        logger.error("factor_data 为空")
        raise ValueError("factor_data不能为空")
    
    logger.debug(f"输入因子数据: {len(factor_data)} 条记录")
    
    # 复制数据避免修改原始数据
    signals = factor_data.copy()
    
    # 确保有factor列
    if 'factor' not in signals.columns:
        logger.warning("factor_data 缺少 'factor' 列，尝试自动识别")
        if len(signals.columns) == 1:
            signals.columns = ['factor']
            logger.info("已将唯一列重命名为 'factor'")
        else:
            logger.error(f"无法识别 factor 列: {list(signals.columns)}")
            raise ValueError(f"factor_data必须包含'factor'列，当前列: {list(signals.columns)}")
    
    # 检查阈值的合理性
    logger.debug("检查阈值合理性...")
    if strong_buy_threshold <= buy_threshold:
        logger.error(f"阈值设置不合理: strong_buy({strong_buy_threshold}) <= buy({buy_threshold})")
        raise ValueError(f"strong_buy_threshold ({strong_buy_threshold}) 必须大于 buy_threshold ({buy_threshold})")
    if sell_threshold <= strong_sell_threshold:
        logger.error(f"阈值设置不合理: sell({sell_threshold}) <= strong_sell({strong_sell_threshold})")
        raise ValueError(f"sell_threshold ({sell_threshold}) 必须大于 strong_sell_threshold ({strong_sell_threshold})")
    if buy_threshold <= sell_threshold:
        logger.error(f"阈值设置不合理: buy({buy_threshold}) <= sell({sell_threshold})")
        raise ValueError(f"buy_threshold ({buy_threshold}) 必须大于 sell_threshold ({sell_threshold})")
    
    # 生成数值信号
    def categorize_signal(factor_value):
        try:
            if pd.isna(factor_value):
                return 0
            if not np.isfinite(factor_value):
                return 0  # 无穷值视为中性
            if factor_value >= strong_buy_threshold:
                return 2
            elif factor_value >= buy_threshold:
                return 1
            elif factor_value <= strong_sell_threshold:
                return -2
            elif factor_value <= sell_threshold:
                return -1
            else:
                return 0
        except Exception:
            return 0  # 任何异常都返回中性
    
    try:
        signals['signal'] = signals['factor'].apply(categorize_signal)
    except Exception as e:
        raise RuntimeError(f"生成信号失败: {str(e)}")
    
    # 生成信号标签
    signal_labels = {
        2: '强烈买入',
        1: '买入',
        0: '中性',
        -1: '卖出',
        -2: '强烈卖出'
    }
    signals['signal_label'] = signals['signal'].map(signal_labels)
    
    # 检查是否有未映射的信号值
    unmapped = signals['signal_label'].isna().sum()
    if unmapped > 0:
        print(f"⚠️ 警告: {unmapped} 个信号值未能映射到标签")
        signals['signal_label'].fillna('未知', inplace=True)
    
    return signals

def get_top_stocks(
    signals: pd.DataFrame,
    date: str,
    top_n: int = 10,
    signal_filter: int = 1
) -> pd.DataFrame:
    """
    获取指定日期的Top N推荐股票列表。
    
    **功能说明**：
    从指定交易日的信号数据中，筛选出满足信号等级要求的股票，
    并按因子值降序排列，返回前N只股票。
    
    **典型应用**：
    - 每日选股：选出当天技术面最强的股票
    - 组合构建：选出Top N只股票等权配置
    - 信号验证：检查某个日期的推荐股票质量
    
    Parameters
    ----------
    signals : pd.DataFrame
        包含信号的DataFrame，必须是 generate_ichimoku_signals() 的返回值
        Index: MultiIndex (trade_date, ts_code)
        Columns: ['factor', 'signal', 'signal_label']
    date : str
        目标日期，格式 'YYYY-MM-DD' 或 'YYYYMMDD'
        会自动转换为 datetime 类型
    top_n : int, default=10
        选取的股票数量
        如果满足条件的股票少于top_n，则返回所有满足条件的股票
    signal_filter : int, default=1
        最低信号等级筛选条件：
        - 1: 买入及以上（signal >= 1），包括买入和强烈买入
        - 2: 仅强烈买入（signal >= 2），最严格筛选
        - 0: 中性及以上（signal >= 0），包括所有非卖出信号
        - -1: 卖出及以上（signal >= -1），几乎所有股票
    
    Returns
    -------
    pd.DataFrame
        Top N股票列表，按因子值降序排列
        Index: MultiIndex (trade_date, ts_code)
        Columns: ['factor', 'signal', 'signal_label']
        
    Raises
    ------
    ValueError
        - signals 不包含指定日期的数据
        - date 格式错误
        
    Notes
    -----
    - 返回的股票数量可能少于top_n，如果满足条件的股票不足
    - 因子值相同时，排序可能不稳定
    - 建议 signal_filter=1（买入及以上）用于日常选股
    - signal_filter=2（仅强烈买入）用于保守策略
    
    Examples
    --------
    >>> # 获取最新日期的Top 10推荐股票
    >>> latest_date = signals.index.get_level_values('trade_date').max()
    >>> top_stocks = get_top_stocks(signals, latest_date, top_n=10)
    >>> print(top_stocks)
    >>> 
    >>> # 只选择强烈买入信号的股票
    >>> strong_buy_stocks = get_top_stocks(
    ...     signals,
    ...     date='2024-01-15',
    ...     top_n=20,
    ...     signal_filter=2
    ... )
    >>> 
    >>> # 遍历显示推荐股票
    >>> for idx, (index, row) in enumerate(top_stocks.iterrows(), 1):
    ...     ts_code = index[1]
    ...     print(f"{idx}. {ts_code}: {row['signal_label']}, 因子={row['factor']:.3f}")
    
    See Also
    --------
    generate_ichimoku_signals : 生成选股信号
    generate_trading_signals : 生成每日信号统计
    """
    if not isinstance(signals, pd.DataFrame):
        raise TypeError(f"signals必须是DataFrame类型，当前类型: {type(signals)}")
    
    if signals.empty:
        raise ValueError("signals数据为空")
    
    if 'signal' not in signals.columns or 'factor' not in signals.columns:
        raise ValueError(f"signals必须包含'signal'和'factor'列，当前列: {list(signals.columns)}")
    
    # 转换日期
    try:
        date = pd.to_datetime(date)
    except Exception as e:
        raise ValueError(f"日期格式错误: {date}，应为'YYYY-MM-DD'格式: {str(e)}")
    
    # 检查参数合理性
    if top_n <= 0:
        raise ValueError(f"top_n必须大于0，当前值: {top_n}")
    
    if signal_filter not in [-2, -1, 0, 1, 2]:
        raise ValueError(f"signal_filter必须在[-2, -1, 0, 1, 2]范围内，当前值: {signal_filter}")
    
    # 筛选指定日期的数据
    try:
        if isinstance(signals.index, pd.MultiIndex):
            # 检查日期是否存在
            available_dates = signals.index.get_level_values('trade_date').unique()
            if date not in available_dates:
                closest_date = min(available_dates, key=lambda x: abs(x - date))
                raise ValueError(f"指定日期 {date.strftime('%Y-%m-%d')} 不存在\n"
                               f"  可用日期范围: {available_dates.min().strftime('%Y-%m-%d')} ~ "
                               f"{available_dates.max().strftime('%Y-%m-%d')}\n"
                               f"  最近的日期: {closest_date.strftime('%Y-%m-%d')}")
            
            date_data = signals.xs(date, level='trade_date', drop_level=False)
        else:
            date_data = signals[signals.index.get_level_values('trade_date') == date]
    except KeyError as e:
        raise ValueError(f"无法获取日期 {date.strftime('%Y-%m-%d')} 的数据: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"筛选日期数据失败: {str(e)}")
    
    if date_data.empty:
        raise ValueError(f"日期 {date.strftime('%Y-%m-%d')} 没有数据")
    
    # 筛选满足信号条件的股票
    filtered_data = date_data[date_data['signal'] >= signal_filter].copy()
    
    if filtered_data.empty:
        print(f"⚠️ 警告: 日期 {date.strftime('%Y-%m-%d')} 没有满足条件(signal>={signal_filter})的股票")
        return pd.DataFrame()  # 返回空DataFrame
    
    # 按因子值降序排列,取Top N
    try:
        top_stocks = filtered_data.sort_values('factor', ascending=False).head(top_n)
    except Exception as e:
        raise RuntimeError(f"排序和筛选失败: {str(e)}")
    
    return top_stocks

def generate_trading_signals(
    factor_data: pd.DataFrame,
    strong_buy_threshold: float = 1.0,
    buy_threshold: float = 0.5
) -> pd.DataFrame:
    """
    生成交易信号每日统计汇总报告。
    
    **功能说明**：
    对因子数据生成信号后，按交易日汇总各类信号的数量，
    以及因子的统计特征，用于追踪市场整体技术面的变化趋势。
    
    **应用场景**：
    - 市场情绪监控：观察买入/卖出信号数量的变化
    - 信号质量评估：跟踪因子分布的稳定性
    - 择时参考：信号数量剧烈变化可能预示市场转折
    - 策略调整：根据信号分布调整仓位或阈值
    
    **输出指标**：
    1. 信号数量统计：
       - strong_buy_count: 强烈买入信号数量
       - buy_count: 买入信号数量
       - neutral_count: 中性信号数量
       - sell_count: 卖出信号数量
       - strong_sell_count: 强烈卖出信号数量
    
    2. 因子统计特征：
       - mean: 因子均值（应接近0）
       - std: 因子标准差（应接近1）
       - min: 因子最小值
       - max: 因子最大值
    
    Parameters
    ----------
    factor_data : pd.DataFrame
        Ichimoku因子数据，calculate_ichimoku_factor() 的返回值
        Index: MultiIndex (trade_date, ts_code)
        Columns: ['factor']
    strong_buy_threshold : float, default=1.0
        强烈买入信号阈值
        与 generate_ichimoku_signals() 保持一致
    buy_threshold : float, default=0.5
        买入信号阈值
        与 generate_ichimoku_signals() 保持一致
    
    Returns
    -------
    pd.DataFrame
        每日信号统计，Index 为交易日期（datetime）
        Columns:
        - signal_strong_buy_count: 强烈买入数量
        - signal_buy_count: 买入数量
        - signal_neutral_count: 中性数量
        - signal_sell_count: 卖出数量
        - signal_strong_sell_count: 强烈卖出数量
        - factor_mean: 因子均值
        - factor_std: 因子标准差
        - factor_min: 因子最小值
        - factor_max: 因子最大值
        
    Notes
    -----
    - 信号数量之和应等于当天有因子值的股票总数
    - 因子均值应接近0，标准差应接近1（因为做了标准化）
    - 如果某天没有数据，该日期不会出现在结果中
    
    Examples
    --------
    >>> # 生成每日信号统计
    >>> daily_stats = generate_trading_signals(factor_data)
    >>> 
    >>> # 查看最近10天的统计
    >>> print(daily_stats.tail(10))
    >>> 
    >>> # 计算总买入信号数量（强烈买入+买入）
    >>> daily_stats['total_buy'] = (
    ...     daily_stats['signal_strong_buy_count'] + 
    ...     daily_stats['signal_buy_count']
    ... )
    >>> 
    >>> # 绘制买入信号数量趋势
    >>> import matplotlib.pyplot as plt
    >>> daily_stats['total_buy'].plot(title='每日买入信号数量')
    >>> plt.show()
    >>> 
    >>> # 检测信号异常（买入信号数量突增）
    >>> ma_20 = daily_stats['total_buy'].rolling(20).mean()
    >>> anomaly = daily_stats[daily_stats['total_buy'] > ma_20 * 1.5]
    >>> print(f"发现 {len(anomaly)} 个异常交易日")
    
    See Also
    --------
    generate_ichimoku_signals : 生成个股选股信号
    get_top_stocks : 获取Top N推荐股票
    """
    signals = generate_ichimoku_signals(
        factor_data,
        strong_buy_threshold=strong_buy_threshold,
        buy_threshold=buy_threshold
    )
    
    # 按日期汇总统计
    if isinstance(signals.index, pd.MultiIndex):
        daily_stats = signals.groupby(level='trade_date').agg({
            'signal': [
                ('strong_buy_count', lambda x: (x == 2).sum()),
                ('buy_count', lambda x: (x == 1).sum()),
                ('neutral_count', lambda x: (x == 0).sum()),
                ('sell_count', lambda x: (x == -1).sum()),
                ('strong_sell_count', lambda x: (x == -2).sum())
            ],
            'factor': ['mean', 'std', 'min', 'max']
        })
        daily_stats.columns = ['_'.join(col).strip() for col in daily_stats.columns.values]
    else:
        daily_stats = pd.DataFrame()
    
    return daily_stats

def run_ichimoku_factor_backtest(
    start_date: str = '2024-01-01',
    end_date: str = '2024-02-29',
    stock_codes: Optional[List[str]] = None,
    rebalance_freq: str = 'weekly',
    transaction_cost: float = 0.0003,
    long_direction: str = 'high'
) -> dict:
    """
    运行Ichimoku因子策略回测
    
    Parameters
    ----------
    start_date : str
        回测开始日期
    end_date : str
        回测结束日期
    stock_codes : Optional[List[str]]
        股票代码列表
    rebalance_freq : str
        调仓频率：'daily', 'weekly', 'monthly'
    transaction_cost : float
        单边交易费用，默认 0.03%
    long_direction : str
        多头方向：'high' 做多高因子值（推荐），'low' 做多低因子值
        
    Returns
    -------
    dict
        包含因子数据、组合收益、业绩指标和IC分析结果
    """
    print("\n" + "=" * 60)
    print("Ichimoku 云图因子策略回测")
    print("=" * 60)
    
    # 参数验证
    try:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        if start_dt >= end_dt:
            raise ValueError(f"开始日期必须早于结束日期: {start_date} >= {end_date}")
    except Exception as e:
        raise ValueError(f"日期格式错误: {str(e)}")
    
    if rebalance_freq not in ['daily', 'weekly', 'monthly']:
        raise ValueError(f"rebalance_freq必须是'daily', 'weekly'或'monthly'，当前值: {rebalance_freq}")
    
    if transaction_cost < 0 or transaction_cost > 0.01:
        print(f"⚠️ 警告: 交易成本 {transaction_cost:.4f} 异常(通常在0-0.01之间)")
    
    if long_direction not in ['high', 'low']:
        raise ValueError(f"long_direction必须是'high'或'low'，当前值: {long_direction}")
    
    # 初始化数据管理器
    try:
        data_manager = DataManager()
    except Exception as e:
        raise RuntimeError(f"初始化数据管理器失败: {str(e)}")
    
    # 配置信息
    print(f"\n回测配置:")
    print(f"  时间范围: {start_date} ~ {end_date}")
    print(f"  调仓频率: {rebalance_freq}")
    print(f"  交易成本: {transaction_cost:.4f}")
    print(f"  多头方向: {long_direction}")
    
    # 步骤1: 计算因子
    print(f"\n{'='*60}")
    print("步骤 1: 计算 Ichimoku 因子")
    print(f"{'='*60}")
    
    try:
        factor_data = calculate_ichimoku_factor(
            data_manager=data_manager,
            start_date=start_date,
            end_date=end_date,
            stock_codes=stock_codes
        )
    except Exception as e:
        raise RuntimeError(f"计算因子失败: {str(e)}")
    
    if factor_data is None or factor_data.empty:
        raise ValueError(f'因子数据为空，无法回测\n'
                        f'可能原因：\n'
                        f'  1. 日期范围内无有效数据\n'
                        f'  2. 股票代码不正确\n'
                        f'  3. 数据源问题')
    
    print(f"✅ 因子计算完成: {len(factor_data):,} 条记录")
    
    # 添加详细的因子统计分析
    print(f"\n因子数据统计:")
    print(f"{'─'*60}")
    
    factor_values = factor_data['factor']
    
    # 基础统计
    print(f"数据量统计:")
    print(f"  总记录数: {len(factor_data):,}")
    print(f"  覆盖股票: {factor_data.index.get_level_values('ts_code').nunique()} 只")
    print(f"  覆盖交易日: {factor_data.index.get_level_values('trade_date').nunique()} 天")
    print(f"  平均每日股票数: {len(factor_data) / factor_data.index.get_level_values('trade_date').nunique():.0f} 只")
    
    # 因子分布统计
    print(f"\n因子分布:")
    print(f"  均值: {factor_values.mean():>8.4f} (标准化后应接近0)")
    print(f"  标准差: {factor_values.std():>8.4f} (标准化后应接近1)")
    print(f"  最小值: {factor_values.min():>8.4f}")
    print(f"  25%分位: {factor_values.quantile(0.25):>8.4f}")
    print(f"  中位数: {factor_values.median():>8.4f}")
    print(f"  75%分位: {factor_values.quantile(0.75):>8.4f}")
    print(f"  最大值: {factor_values.max():>8.4f}")
    print(f"  偏度: {factor_values.skew():>8.4f} (接近0表示对称)")
    print(f"  峰度: {factor_values.kurtosis():>8.4f} (接近0表示正态)")
    
    # 因子分组统计
    print(f"\n因子分组 (按标准差):")
    very_low = (factor_values < -2).sum()
    low = ((factor_values >= -2) & (factor_values < -1)).sum()
    below_avg = ((factor_values >= -1) & (factor_values < 0)).sum()
    above_avg = ((factor_values >= 0) & (factor_values < 1)).sum()
    high = ((factor_values >= 1) & (factor_values < 2)).sum()
    very_high = (factor_values >= 2).sum()
    
    total = len(factor_values)
    print(f"  极低 (< -2σ):  {very_low:>6} ({very_low/total*100:>5.1f}%)")
    print(f"  低 (-2σ~-1σ):  {low:>6} ({low/total*100:>5.1f}%)")
    print(f"  偏低 (-1σ~0):  {below_avg:>6} ({below_avg/total*100:>5.1f}%)")
    print(f"  偏高 (0~1σ):   {above_avg:>6} ({above_avg/total*100:>5.1f}%)")
    print(f"  高 (1σ~2σ):    {high:>6} ({high/total*100:>5.1f}%)")
    print(f"  极高 (> 2σ):   {very_high:>6} ({very_high/total*100:>5.1f}%)")
    
    # 异常值检测
    extreme_low = (factor_values < -5).sum()
    extreme_high = (factor_values > 5).sum()
    if extreme_low > 0 or extreme_high > 0:
        print(f"\n⚠️  极端值检测:")
        if extreme_low > 0:
            print(f"  极端低值 (< -5σ): {extreme_low} ({extreme_low/total*100:.2f}%)")
        if extreme_high > 0:
            print(f"  极端高值 (> 5σ): {extreme_high} ({extreme_high/total*100:.2f}%)")
    
    # 时间序列统计
    print(f"\n时间序列特征:")
    daily_mean = factor_data.groupby(level='trade_date')['factor'].mean()
    daily_std = factor_data.groupby(level='trade_date')['factor'].std()
    print(f"  日均值波动: {daily_mean.std():.4f}")
    print(f"  日标准差均值: {daily_std.mean():.4f}")
    print(f"  日标准差波动: {daily_std.std():.4f}")
    
    print(f"{'─'*60}\n")
    
    # 步骤2: 准备收益率数据
    print(f"\n{'='*60}")
    print("步骤 2: 准备收益率数据")
    print(f"{'='*60}")
    
    try:
        stock_list = factor_data.index.get_level_values('ts_code').unique().tolist()
    except Exception as e:
        raise RuntimeError(f"提取股票列表失败: {str(e)}")
    
    print(f"加载 {len(stock_list)} 只股票的价格数据...")
    
    try:
        stock_data = data_manager.load_data(
            'daily',
            start_date=start_date,
            end_date=end_date,
            stock_codes=stock_list
        )
    except Exception as e:
        raise RuntimeError(f"加载股票数据失败: {str(e)}")
    
    if stock_data is None or stock_data.empty:
        raise ValueError(f"无法加载用于回测的股票数据\n"
                        f"  股票数量: {len(stock_list)}\n"
                        f"  日期范围: {start_date} ~ {end_date}")
    
    try:
        stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'])
        stock_data = stock_data.sort_values(['ts_code', 'trade_date'])
        
        # 计算下一日收益率
        stock_data['next_close'] = stock_data.groupby('ts_code')['close'].shift(-1)
        stock_data['next_return'] = (stock_data['next_close'] / stock_data['close']) - 1
    except Exception as e:
        raise RuntimeError(f"计算收益率失败: {str(e)}")
    
    print(f"✅ 收益率数据准备完成")
    print(f"  数据记录: {len(stock_data):,} 条")
    print(f"  有效收益率: {stock_data['next_return'].notna().sum():,} 条 "
          f"({stock_data['next_return'].notna().sum()/len(stock_data)*100:.1f}%)")
    
    # 步骤3: 合并因子和收益率数据
    print(f"\n{'='*60}")
    print("步骤 3: 合并因子和收益率数据")
    print(f"{'='*60}")
    
    try:
        combined = pd.merge(
            factor_data.reset_index(),
            stock_data[['trade_date', 'ts_code', 'next_return']],
            on=['trade_date', 'ts_code'],
            how='inner'
        )
    except Exception as e:
        raise RuntimeError(f"合并数据失败: {str(e)}")
    
    # 删除缺失值
    original_len = len(combined)
    combined = combined.dropna(subset=['factor', 'next_return'])
    dropped = original_len - len(combined)
    
    if combined.empty:
        raise ValueError(f"合并后数据为空\n"
                        f"  因子记录: {len(factor_data)}\n"
                        f"  收益率记录: {stock_data['next_return'].notna().sum()}\n"
                        f"  可能原因: 日期不匹配")
    
    print(f"✅ 数据合并完成")
    print(f"  合并后记录: {len(combined):,} 条")
    print(f"  覆盖交易日: {combined['trade_date'].nunique()}")
    if dropped > 0:
        print(f"  过滤缺失值: {dropped} 条 ({dropped/original_len*100:.1f}%)")
    
    # 步骤4: 使用 BacktestEngine 运行回测
    print(f"\n{'='*60}")
    print("步骤 4: 运行回测")
    print(f"{'='*60}")
    
    try:
        from backtest_engine.engine import BacktestEngine
        
        engine = BacktestEngine(
            data_manager=data_manager,
            fee=transaction_cost,
            long_direction=long_direction,
            rebalance_freq=rebalance_freq,
            factor_name='factor'
        )
        
        # 直接设置因子数据和组合数据
        engine.factor_data = factor_data
        engine.combined_data = combined
        
        # 运行回测
        portfolio_returns = engine.run()
        print("✅ 回测完成")
        
    except ImportError:
        print("⚠️ BacktestEngine 不可用，使用简化回测方法")
        
        # 简化回测：Long-Only策略，等权持有所有有因子值的股票
        portfolio_returns = combined.groupby('trade_date')['next_return'].mean()
        
        # 模拟交易成本
        if rebalance_freq == 'daily':
            rebalance_dates = portfolio_returns.index
        else:
            freq_map = {'weekly': 'W-MON', 'monthly': 'MS'}
            rebalance_dates = pd.date_range(
                start=start_date,
                end=end_date,
                freq=freq_map.get(rebalance_freq, 'W-MON')
            )
        
        if len(portfolio_returns) > 0:
            cost_impact = len(rebalance_dates) * transaction_cost / len(portfolio_returns)
            portfolio_returns -= cost_impact
        
        portfolio_returns = pd.DataFrame({'Long_Only': portfolio_returns})
    
    # 步骤5: 计算业绩指标
    print(f"\n{'='*60}")
    print("步骤 5: 计算业绩指标")
    print(f"{'='*60}")
    
    if not isinstance(portfolio_returns, pd.DataFrame) or 'Long_Only' not in portfolio_returns.columns:
        raise ValueError('回测结果缺少 Long_Only 列')
    
    series = portfolio_returns['Long_Only']
    cum_returns = (1 + series).cumprod()
    
    # 基础指标计算
    total_return = float(cum_returns.iloc[-1] - 1) if len(cum_returns) > 0 else 0
    trading_days = len(series)
    annualized_return = float(cum_returns.iloc[-1] ** (252 / trading_days) - 1) if trading_days > 0 else 0
    volatility = float(series.std() * np.sqrt(252))
    sharpe_ratio = float(annualized_return / volatility) if volatility > 0 else 0
    
    running_max = cum_returns.cummax()
    drawdown = cum_returns / running_max - 1
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0
    
    win_rate = float((series > 0).mean())
    
    # 调仓次数
    if rebalance_freq == 'daily':
        rebalance_count = trading_days
    else:
        freq_map = {'weekly': 'W-MON', 'monthly': 'MS'}
        rebalance_dates = pd.date_range(
            start=start_date,
            end=end_date,
            freq=freq_map.get(rebalance_freq, 'W-MON')
        )
        rebalance_count = len(rebalance_dates)
    
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'rebalance_count': rebalance_count
    }
    
    # 步骤6: IC分析
    print(f"\n{'='*60}")
    print("步骤 6: IC 分析")
    print(f"{'='*60}")
    
    ic_series = None
    ic_mean = None
    ic_std = None
    icir = None
    ic_positive_ratio = None
    
    try:
        # 尝试使用 PerformanceAnalyzer
        from backtest_engine.engine import BacktestEngine
        analyzer = engine.get_performance_analysis()
        ic_series = analyzer.ic_series
        
        if ic_series is not None and not ic_series.empty:
            ic_mean = float(ic_series.mean())
            ic_std = float(ic_series.std())
            icir = float(ic_mean / ic_std) if ic_std > 0 else 0
            ic_positive_ratio = float((ic_series > 0).mean())
            print(f"✅ 使用 PerformanceAnalyzer 计算 IC")
    except:
        # 手动计算IC
        print(f"手动计算 IC...")
        ic_list = []
        
        for date in combined['trade_date'].unique():
            date_data = combined[combined['trade_date'] == date]
            if len(date_data) >= 10:  # 至少需要10个样本
                # 计算Spearman相关系数
                correlation = date_data[['factor', 'next_return']].corr(method='spearman').iloc[0, 1]
                if not np.isnan(correlation):
                    ic_list.append({'trade_date': date, 'ic': correlation})
        
        if ic_list:
            ic_series = pd.DataFrame(ic_list).set_index('trade_date')['ic']
            ic_mean = float(ic_series.mean())
            ic_std = float(ic_series.std())
            icir = float(ic_mean / ic_std) if ic_std > 0 else 0
            ic_positive_ratio = float((ic_series > 0).mean())
            print(f"✅ 手动计算完成")
    
    analysis_results = {
        'ic_series': ic_series,
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        'ic_positive_ratio': ic_positive_ratio
    }
    
    # 打印结果摘要
    print(f"\n{'='*60}")
    print("回测结果摘要")
    print(f"{'='*60}")
    print(f"\n📊 业绩指标:")
    print(f"  总收益率: {metrics['total_return']:.2%}")
    print(f"  年化收益率: {metrics['annualized_return']:.2%}")
    print(f"  年化波动率: {metrics['volatility']:.2%}")
    print(f"  夏普比率: {metrics['sharpe_ratio']:.3f}")
    print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
    print(f"  胜率: {metrics['win_rate']:.2%}")
    print(f"  调仓次数: {metrics['rebalance_count']}")
    
    if ic_series is not None:
        print(f"\n📊 IC 分析:")
        print(f"{'─'*60}")
        print(f"基础统计:")
        print(f"  IC 均值: {ic_mean:>8.4f}")
        print(f"  IC 标准差: {ic_std:>8.4f}")
        print(f"  ICIR: {icir:>8.4f} (IC均值/标准差)")
        print(f"  IC>0 占比: {ic_positive_ratio:>8.2%}")
        
        # 额外的IC统计
        print(f"\nIC 分布:")
        print(f"  最小值: {ic_series.min():>8.4f}")
        print(f"  25%分位: {ic_series.quantile(0.25):>8.4f}")
        print(f"  中位数: {ic_series.median():>8.4f}")
        print(f"  75%分位: {ic_series.quantile(0.75):>8.4f}")
        print(f"  最大值: {ic_series.max():>8.4f}")
        
        # IC稳定性分析
        ic_abs_mean = ic_series.abs().mean()
        print(f"\nIC 稳定性:")
        print(f"  |IC|均值: {ic_abs_mean:>8.4f}")
        print(f"  IC偏度: {ic_series.skew():>8.4f}")
        print(f"  IC峰度: {ic_series.kurtosis():>8.4f}")
        
        # IC显著性检测
        significant_positive = (ic_series > 0.02).sum()
        significant_negative = (ic_series < -0.02).sum()
        print(f"\nIC 显著性 (|IC|>0.02):")
        print(f"  显著正IC: {significant_positive} ({significant_positive/len(ic_series)*100:.1f}%)")
        print(f"  显著负IC: {significant_negative} ({significant_negative/len(ic_series)*100:.1f}%)")
        
        # IC趋势分析
        ic_first_half = ic_series[:len(ic_series)//2].mean()
        ic_second_half = ic_series[len(ic_series)//2:].mean()
        print(f"\nIC 时间趋势:")
        print(f"  前半段IC均值: {ic_first_half:>8.4f}")
        print(f"  后半段IC均值: {ic_second_half:>8.4f}")
        print(f"  趋势变化: {ic_second_half - ic_first_half:>8.4f}")
        
        print(f"{'─'*60}")
    
    print(f"\n{'='*60}\n")
    
    return {
        'factor_data': factor_data,
        'portfolio_returns': portfolio_returns,
        'performance_metrics': metrics,
        'analysis_results': analysis_results
    }

def main():
    """
    主函数：演示Ichimoku因子计算、回测和选股信号功能。
    
    **演示内容**：
    1. 因子计算：展示完整的Ichimoku因子计算流程
    2. 回测分析：运行策略回测并输出业绩指标
    3. IC分析：评估因子预测能力
    4. 选股信号：生成买卖信号并推荐Top股票
    5. 信号趋势：展示每日信号数量变化
    
    **默认配置**：
    - 时间范围：2023-01-01 至 2024-02-29（约14个月）
    - 调仓频率：每周（weekly）
    - 交易成本：0.03%（双边0.06%）
    - 策略方向：做多高因子值股票
    
    **输出信息**：
    - 回测业绩：收益率、夏普比率、最大回撤等
    - IC指标：IC均值、ICIR、IC胜率
    - 信号统计：各类信号的数量分布
    - 推荐股票：最新日期的Top 10股票列表
    - 趋势分析：最近10个交易日的信号变化
    
    Notes
    -----
    - 可通过修改 config 字典调整回测参数
    - 如果数据不足，可能会抛出异常
    - 运行时间取决于股票数量和时间跨度
    
    Examples
    --------
    在命令行中运行：
    >>> python ichimoku_factor.py
    
    或在Python脚本中调用：
    >>> if __name__ == "__main__":
    ...     main()
    """
    print("Ichimoku云图因子策略演示")
    print("=" * 50)
    
    try:
        config = {
            'start_date': '2023-01-01',
            'end_date': '2024-02-29',
            'rebalance_freq': 'weekly',
            'transaction_cost': 0.0003,
            'long_direction': 'high'
        }
        
        print("\n回测配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")
            
        results = run_ichimoku_factor_backtest(**config)
        
        print("\n回测结果总结:")
        metrics = results['performance_metrics']
        print(f"  总收益率: {metrics['total_return']:.2%}")
        print(f"  年化收益: {metrics['annualized_return']:.2%}")
        print(f"  夏普比率: {metrics['sharpe_ratio']:.3f}")
        print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"  胜率: {metrics['win_rate']:.2%}")
        print(f"  调仓次数: {metrics['rebalance_count']}")
        
        # IC分析结果
        if results['analysis_results']['ic_series'] is not None:
            ic = results['analysis_results']['ic_series']
            print(f"\nIC分析:")
            print(f"  IC均值: {ic.mean():.3f}")
            print(f"  IC标准差: {ic.std():.3f}")
            print(f"  IC_IR: {(ic.mean() / ic.std()):.3f}")
        
        # 生成选股信号
        print("\n" + "=" * 50)
        print("选股信号分析")
        print("=" * 50)
        
        factor_data = results['factor_data']
        signals = generate_ichimoku_signals(factor_data)
        
        # 信号统计 - 详细版
        print(f"\n信号分布统计 (全部数据):")
        print(f"{'─'*60}")
        signal_counts = signals['signal'].value_counts().sort_index(ascending=False)
        total_signals = len(signals)
        
        for sig in [2, 1, 0, -1, -2]:
            if sig in signal_counts.index:
                count = signal_counts[sig]
                label = {2: '强烈买入', 1: '买入', 0: '中性', -1: '卖出', -2: '强烈卖出'}[sig]
                pct = count / total_signals * 100
                bar_length = int(pct / 2)  # 每2%一个字符
                bar = '█' * bar_length
                print(f"  {label:>6} ({sig:>2}): {count:>6} ({pct:>5.1f}%) {bar}")
            else:
                label = {2: '强烈买入', 1: '买入', 0: '中性', -1: '卖出', -2: '强烈卖出'}[sig]
                print(f"  {label:>6} ({sig:>2}): {0:>6} (  0.0%)")
        
        # 买卖信号汇总
        buy_signals = signal_counts.get(2, 0) + signal_counts.get(1, 0)
        sell_signals = signal_counts.get(-1, 0) + signal_counts.get(-2, 0)
        neutral_signals = signal_counts.get(0, 0)
        
        print(f"\n信号汇总:")
        print(f"  总买入信号 (1+2): {buy_signals:>6} ({buy_signals/total_signals*100:>5.1f}%)")
        print(f"  中性信号 (0):     {neutral_signals:>6} ({neutral_signals/total_signals*100:>5.1f}%)")
        print(f"  总卖出信号 (-1-2): {sell_signals:>6} ({sell_signals/total_signals*100:>5.1f}%)")
        
        # 信号质量指标
        print(f"\n信号质量指标:")
        strong_signals = signal_counts.get(2, 0) + signal_counts.get(-2, 0)
        print(f"  强信号占比: {strong_signals/total_signals*100:.1f}%")
        print(f"  信号多样性: {len(signal_counts)} 种信号")
        
        # 按因子区间的信号分布
        print(f"\n因子-信号对应关系:")
        for sig in [2, 1, 0, -1, -2]:
            sig_data = signals[signals['signal'] == sig]
            if not sig_data.empty:
                label = {2: '强烈买入', 1: '买入', 0: '中性', -1: '卖出', -2: '强烈卖出'}[sig]
                avg_factor = sig_data['factor'].mean()
                min_factor = sig_data['factor'].min()
                max_factor = sig_data['factor'].max()
                print(f"  {label:>6}: 因子范围 [{min_factor:>6.2f}, {max_factor:>6.2f}], 均值 {avg_factor:>6.2f}")
        
        print(f"{'─'*60}")
        
        # 获取最新日期的Top 10股票
        if isinstance(signals.index, pd.MultiIndex):
            latest_date = signals.index.get_level_values('trade_date').max()
            print(f"\n最新日期 ({latest_date.strftime('%Y-%m-%d')}) 选股分析:")
            print("-" * 80)
            
            # 当日信号统计
            latest_signals = signals.xs(latest_date, level='trade_date')
            latest_counts = latest_signals['signal'].value_counts().sort_index(ascending=False)
            
            print(f"\n当日信号分布:")
            for sig in [2, 1, 0, -1, -2]:
                if sig in latest_counts.index:
                    count = latest_counts[sig]
                    label = {2: '强烈买入', 1: '买入', 0: '中性', -1: '卖出', -2: '强烈卖出'}[sig]
                    print(f"  {label}: {count} 只")
            
            total_stocks = len(latest_signals)
            buy_stocks = latest_counts.get(2, 0) + latest_counts.get(1, 0)
            print(f"\n可选股票池: {buy_stocks}/{total_stocks} ({buy_stocks/total_stocks*100:.1f}%)")
            
            top_stocks = get_top_stocks(signals, latest_date, top_n=10, signal_filter=1)
            
            if not top_stocks.empty:
                print(f"\nTop 10 推荐股票:")
                print(f"{'排名':<6}{'股票代码':<12}{'因子值':<10}{'标准化':<10}{'信号':<10}")
                print("-" * 80)
                
                factor_values = []
                for idx, (index, row) in enumerate(top_stocks.iterrows(), 1):
                    ts_code = index[1] if isinstance(index, tuple) else index
                    factor_val = row['factor']
                    factor_values.append(factor_val)
                    
                    # 添加因子排名指示
                    if factor_val >= 2:
                        rank_indicator = "🔥🔥"
                    elif factor_val >= 1.5:
                        rank_indicator = "🔥"
                    elif factor_val >= 1:
                        rank_indicator = "⭐"
                    else:
                        rank_indicator = ""
                    
                    print(f"{idx:<6}{ts_code:<12}{factor_val:<10.4f}{factor_val:<10.2f}"
                          f"{row['signal_label']:<10}{rank_indicator}")
                
                # Top 10统计
                print("\n" + "-" * 80)
                print(f"Top 10 统计:")
                print(f"  因子均值: {np.mean(factor_values):.3f}")
                print(f"  因子标准差: {np.std(factor_values):.3f}")
                print(f"  最高因子: {max(factor_values):.3f}")
                print(f"  最低因子: {min(factor_values):.3f}")
                
                strong_buy_count = sum(1 for _, row in top_stocks.iterrows() if row['signal'] == 2)
                print(f"  强烈买入: {strong_buy_count}/10")
                
            else:
                print("\n⚠️  当前无符合条件的推荐股票")
                print(f"  建议：降低signal_filter或调整信号阈值")
            
            # 生成每日信号统计
            print("\n每日信号趋势分析:")
            print("-" * 80)
            daily_stats = generate_trading_signals(factor_data)
            
            if not daily_stats.empty:
                # 计算总买入卖出信号
                if 'signal_strong_buy_count' in daily_stats.columns:
                    daily_stats['total_buy'] = (
                        daily_stats['signal_strong_buy_count'] + 
                        daily_stats['signal_buy_count']
                    )
                    daily_stats['total_sell'] = (
                        daily_stats['signal_sell_count'] + 
                        daily_stats['signal_strong_sell_count']
                    )
                
                # 显示最近10天
                recent_stats = daily_stats.tail(10)
                
                print(f"\n最近10个交易日:")
                print(f"{'日期':<12}{'强买':<6}{'买入':<6}{'中性':<6}{'卖出':<6}{'强卖':<6}{'买入合计':<8}{'因子均值':<10}")
                print("-" * 80)
                
                for date, row in recent_stats.iterrows():
                    date_str = date.strftime('%Y-%m-%d')
                    strong_buy = int(row.get('signal_strong_buy_count', 0))
                    buy = int(row.get('signal_buy_count', 0))
                    neutral = int(row.get('signal_neutral_count', 0))
                    sell = int(row.get('signal_sell_count', 0))
                    strong_sell = int(row.get('signal_strong_sell_count', 0))
                    total_buy = int(row.get('total_buy', strong_buy + buy))
                    factor_mean = row.get('factor_mean', 0)
                    
                    print(f"{date_str:<12}{strong_buy:<6}{buy:<6}{neutral:<6}"
                          f"{sell:<6}{strong_sell:<6}{total_buy:<8}{factor_mean:<10.3f}")
                
                # 趋势分析
                print("\n" + "-" * 80)
                print(f"趋势分析 (最近10天):")
                
                if 'total_buy' in daily_stats.columns:
                    recent_buy = recent_stats['total_buy']
                    print(f"  买入信号:")
                    print(f"    均值: {recent_buy.mean():.1f} 只/天")
                    print(f"    最高: {recent_buy.max():.0f} 只 ({recent_buy.idxmax().strftime('%Y-%m-%d')})")
                    print(f"    最低: {recent_buy.min():.0f} 只 ({recent_buy.idxmin().strftime('%Y-%m-%d')})")
                    print(f"    波动: {recent_buy.std():.1f}")
                    
                    # 趋势判断
                    first_half_mean = recent_buy[:5].mean()
                    second_half_mean = recent_buy[5:].mean()
                    if second_half_mean > first_half_mean * 1.2:
                        trend = "📈 上升趋势 (信号增多)"
                    elif second_half_mean < first_half_mean * 0.8:
                        trend = "📉 下降趋势 (信号减少)"
                    else:
                        trend = "➡️  平稳趋势"
                    print(f"    趋势: {trend}")
                
                if 'factor_mean' in recent_stats.columns:
                    factor_means = recent_stats['factor_mean']
                    print(f"\n  因子均值:")
                    print(f"    近期均值: {factor_means.mean():.4f}")
                    print(f"    波动性: {factor_means.std():.4f}")
                    print(f"    最新值: {factor_means.iloc[-1]:.4f}")
            else:
                print("  无法生成每日统计数据")
        
    except Exception as e:
        print(f"演示运行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()
