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


def get_index_components(data_manager: DataManager, index_code: str = '000852.SH', trade_date: Optional[str] = None) -> List[str]:
    """
    获取指定指数的成分股列表
    
    Parameters
    ----------
    data_manager : DataManager
        数据管理器实例
    index_code : str
        指数代码，默认为中证1000 (000852.SH)
        常用指数代码:
        - 000300.SH: 沪深300
        - 000905.SH: 中证500
        - 000852.SH: 中证1000
        - 000016.SH: 上证50
        - 399006.SZ: 创业板指
    trade_date : Optional[str]
        指定日期，格式 'YYYY-MM-DD' 或 'YYYYMMDD'
        如果为None，使用最新一期数据
    
    Returns
    -------
    List[str]
        成分股代码列表
        
    Raises
    ------
    FileNotFoundError
        如果找不到指数权重数据文件
    ValueError
        如果指数代码无效或数据为空
        
    Examples
    --------
    >>> dm = DataManager()
    >>> # 获取沪深300成分股（最新）
    >>> stocks = get_index_components(dm, index_code='000300.SH')
    >>> # 获取中证500成分股（指定日期）
    >>> stocks = get_index_components(dm, index_code='000905.SH', trade_date='2024-01-01')
    """
    try:
        # 直接从raw_data加载指数权重数据（该数据不需要清洗）
        raw_data_path = Path(__file__).resolve().parent.parent.parent / 'data_manager' / 'raw_data' / 'index_weight_data.parquet'
        
        if not raw_data_path.exists():
            raise FileNotFoundError(
                f"指数权重数据文件不存在: {raw_data_path}\n"
                f"请先运行: python data_manager/data_loader/index_weight_data_loader.py"
            )
        
        try:
            index_weights = pd.read_parquet(raw_data_path)
        except Exception as e:
            raise IOError(f"读取指数权重数据失败: {str(e)}")
        
        if index_weights is None or index_weights.empty:
            raise ValueError("指数权重数据为空，请检查数据加载器是否正常运行")
        
        # 验证必要字段
        required_cols = ['index_code', 'trade_date', 'con_code']
        missing_cols = [col for col in required_cols if col not in index_weights.columns]
        if missing_cols:
            raise ValueError(f"指数权重数据缺少必要字段: {missing_cols}")
        
        # 筛选指定指数
        index_data = index_weights[index_weights['index_code'] == index_code].copy()
        
        if index_data.empty:
            available_indices = index_weights['index_code'].unique().tolist()
            raise ValueError(
                f"未找到指数 '{index_code}' 的权重数据\n"
                f"可用指数代码: {', '.join(available_indices[:10])}"
                f"{'...' if len(available_indices) > 10 else ''}"
            )
        
        # 如果指定了日期，筛选该日期的数据
        if trade_date is not None:
            try:
                # 转换日期格式为YYYYMMDD
                if '-' in trade_date:
                    trade_date = trade_date.replace('-', '')
                
                # 验证日期格式
                if not trade_date.isdigit() or len(trade_date) != 8:
                    raise ValueError(f"日期格式无效: {trade_date}，应为 'YYYYMMDD' 或 'YYYY-MM-DD'")
                
                index_data = index_data[index_data['trade_date'] == trade_date]
                
                if index_data.empty:
                    # 如果指定日期没有数据，使用最接近的日期
                    all_dates = index_weights[index_weights['index_code'] == index_code]['trade_date'].unique()
                    if len(all_dates) == 0:
                        raise ValueError(f"指数 {index_code} 没有任何日期的数据")
                    
                    closest_date = min(all_dates, key=lambda x: abs(int(x) - int(trade_date)))
                    
                    warnings.warn(
                        f"指定日期 {trade_date} 没有数据，使用最接近日期 {closest_date}",
                        UserWarning
                    )
                    index_data = index_weights[
                        (index_weights['index_code'] == index_code) & 
                        (index_weights['trade_date'] == closest_date)
                    ].copy()
            except Exception as e:
                if isinstance(e, ValueError):
                    raise
                warnings.warn(f"日期处理失败: {str(e)}，将使用最新日期", UserWarning)
                latest_date = index_data['trade_date'].max()
                index_data = index_data[index_data['trade_date'] == latest_date]
        else:
            # 使用最新一期数据
            latest_date = index_data['trade_date'].max()
            index_data = index_data[index_data['trade_date'] == latest_date]
        
        # 提取成分股代码
        components = index_data['con_code'].unique().tolist()
        
        if not components:
            raise ValueError(f"指数 {index_code} 的成分股列表为空")
        
        print(f"✅ 获取指数成分股:")
        print(f"   指数代码: {index_code}")
        print(f"   日期: {index_data['trade_date'].iloc[0] if not index_data.empty else 'N/A'}")
        print(f"   成分股数量: {len(components)}")
        
        return components
        
    except (FileNotFoundError, ValueError, IOError) as e:
        # 重新抛出已知异常
        raise
    except Exception as e:
        # 捕获未预期的异常
        raise RuntimeError(f"获取指数成分股时发生未预期错误: {str(e)}") from e


def validate_date_format(date_str: str, param_name: str = 'date') -> str:
    """
    验证并标准化日期格式
    
    Parameters
    ----------
    date_str : str
        日期字符串，支持 'YYYY-MM-DD' 或 'YYYYMMDD' 格式
    param_name : str
        参数名称（用于错误提示）
    
    Returns
    -------
    str
        标准化的日期字符串 'YYYY-MM-DD'
    
    Raises
    ------
    ValueError
        如果日期格式无效
    """
    try:
        # 尝试解析日期
        date_obj = pd.to_datetime(date_str, errors='coerce')
        if pd.isna(date_obj):
            # 尝试YYYYMMDD格式
            date_obj = pd.to_datetime(str(date_str), format='%Y%m%d', errors='coerce')
        
        if pd.isna(date_obj):
            raise ValueError(f"无法解析日期格式: '{date_str}'")
        
        # 验证日期范围合理性
        if date_obj.year < 2000 or date_obj.year > 2100:
            raise ValueError(f"日期年份超出合理范围 (2000-2100): {date_obj.year}")
        
        # 返回标准格式
        return date_obj.strftime('%Y-%m-%d')
    
    except ValueError as e:
        # 重新抛出ValueError，保留原始错误信息
        raise ValueError(
            f"参数 '{param_name}' 的日期格式无效: '{date_str}'\n"
            f"请使用 'YYYY-MM-DD' 或 'YYYYMMDD' 格式，例如: '2024-01-01' 或 '20240101'\n"
            f"错误详情: {str(e)}"
        )
    except Exception as e:
        # 捕获其他未预期错误
        raise RuntimeError(
            f"处理日期 '{date_str}' 时发生未预期错误: {str(e)}"
        ) from e


def calculate_atr_factor(
    data_manager: DataManager,
    start_date: str,
    end_date: str,
    stock_codes: Optional[List[str]] = None,
    index_code: Optional[str] = None,
    period: int = 14,
    use_selection_signals: bool = True,
    atr_percentile_low: float = 30,
    atr_percentile_high: float = 70,
    price_position_threshold: float = 0.7,
    volume_ratio_threshold: float = 1.2,
    filter_limit: bool = True,
    filter_st: bool = True,
    filter_valuation: bool = True,
    pe_min: float = 0.0,
    pe_max: float = 100.0,
    pb_min: float = 0.0,
    pb_max: float = 10.0,
    use_dynamic_weighting: bool = True,
    signal_score_threshold: int = 50,
    n1_threshold: int = 20,
    n2_threshold: int = 50
) -> pd.DataFrame:
    """
    计算ATR因子并构建选股信号,使用过去period天的价格波动计算平均真实波幅。
    
    选股信号逻辑：
    1. ATR扩张信号：ATR从低位快速上升（波动性增加）
    2. 价格位置信号：股价处于近期高位（趋势向上）
    3. 成交量确认：放量突破
    4. ATR相对强度：ATR在所有股票中的相对位置

    Parameters
    ----------
    data_manager : DataManager
        数据管理器实例
    start_date : str
        起始日期
    end_date : str
        结束日期
    stock_codes : Optional[List[str]]
        股票代码列表,如果为None则使用所有可用股票
        注意: stock_codes 和 index_code 只能指定一个
    index_code : Optional[str]
        指数代码,如果指定则使用该指数的成分股作为股票池
        常用指数:
        - '000300.SH': 沪深300
        - '000905.SH': 中证500
        - '000852.SH': 中证1000 (推荐用于小盘成长策略)
        - '000016.SH': 上证50
        - '399006.SZ': 创业板指
        注意: stock_codes 和 index_code 只能指定一个
    period : int
        ATR计算周期,默认14天
    use_selection_signals : bool
        是否使用选股信号筛选,默认True
    atr_percentile_low : float
        ATR分位数下限(用于识别从低波动启动的股票),默认30%
    atr_percentile_high : float
        ATR分位数上限(用于识别高波动股票),默认70%
    price_position_threshold : float
        价格位置阈值(股价在近期区间的位置),默认0.7(70%)
    volume_ratio_threshold : float
        成交量放大倍数阈值,默认1.2倍
    filter_limit : bool
        是否过滤涨停/跌停股票,默认True
        过滤规则:
        - 当日涨停(pct_chg > 9.8%)
        - 次日开盘涨停(next_open/close - 1 > 9.8%)
        - 跌停股票(pct_chg < -9.8%)
    filter_st : bool
        是否过滤ST/退市风险股票,默认True
        过滤规则: 股票名称包含'ST'、'*ST'、'退'等关键词
    filter_valuation : bool
        是否使用估值指标筛选,默认True
        使用PE-TTM和PB市净率进行估值筛选
    pe_min : float
        市盈率(PE-TTM)最小值,默认0.0
        排除亏损股票(PE<0)和估值过低的异常股票
    pe_max : float
        市盈率(PE-TTM)最大值,默认100.0
        排除估值过高的股票,降低泡沫风险
    pb_min : float
        市净率(PB)最小值,默认0.0
        排除破净或异常低估值股票
    pb_max : float
        市净率(PB)最大值,默认10.0
        排除市净率过高的股票
    use_dynamic_weighting : bool
        是否使用动态权重机制,默认True
        True: 根据每日样本数量动态调整筛选标准
        False: 使用固定的ATR分位数权重
    signal_score_threshold : int
        信号得分阈值,默认50分
        只有综合得分>=该阈值的股票才会被选中
    n1_threshold : int
        动态筛选第一阈值,默认20
        当样本数<n1时,不进行筛选(数据不足)
    n2_threshold : int
        动态筛选第二阈值,默认50
        当样本数>=n2时,进行双维度精选(数据充足)

    Returns
    -------
    DataFrame
        MultiIndex (trade_date, ts_code) with single column 'factor'
        如果use_selection_signals=True,返回综合选股信号得分
        如果use_selection_signals=False,返回原始归一化ATR值
    
    Notes
    -----
    动态权重机制说明:
    - 样本数 < n1 (20): 不筛选,使用所有信号
    - n1 <= 样本数 < n2: 单一维度筛选(信号得分>阈值)
    - 样本数 >= n2 (50): 双维度精选(信号得分+ATR相对强度)
    
    这种机制可以在不同市场环境下灵活调整选股标准。
    
    估值筛选说明:
    - PE-TTM (市盈率): Price/Earnings Trailing Twelve Months
      合理范围通常为 0-100, 不同行业有差异
    - PB (市净率): Price/Book Value
      合理范围通常为 0-10, 破净股票(PB<1)需谨慎
    - 过滤掉估值异常的股票可以提高策略稳健性
    """
    # ===== 参数验证 =====
    try:
        # 验证日期格式
        start_date = validate_date_format(start_date, 'start_date')
        end_date = validate_date_format(end_date, 'end_date')
    except (ValueError, RuntimeError) as e:
        raise ValueError(f"日期参数验证失败: {str(e)}") from e
    
    # 验证日期逻辑
    if pd.to_datetime(start_date) > pd.to_datetime(end_date):
        raise ValueError(f"开始日期 ({start_date}) 不能晚于结束日期 ({end_date})")
    
    # 验证period参数
    if not isinstance(period, int) or period < 2:
        raise ValueError(f"period 参数必须为大于等于2的整数，当前值: {period}")
    if period > 100:
        warnings.warn(
            f"period={period} 过大可能导致计算不稳定，建议使用2-100之间的值",
            UserWarning
        )
    
    # 验证百分位数参数
    if not (0 <= atr_percentile_low < atr_percentile_high <= 100):
        raise ValueError(
            f"ATR百分位数参数无效: low={atr_percentile_low}, high={atr_percentile_high}\n"
            f"要求: 0 <= low < high <= 100"
        )
    
    # 验证阈值参数
    if not (0 < price_position_threshold <= 1):
        raise ValueError(f"price_position_threshold 必须在 (0, 1] 范围内，当前值: {price_position_threshold}")
    if volume_ratio_threshold <= 0:
        raise ValueError(f"volume_ratio_threshold 必须大于0，当前值: {volume_ratio_threshold}")
    
    # 验证估值筛选参数
    if filter_valuation:
        if pe_min < 0 or pe_max <= pe_min:
            raise ValueError(f"PE参数无效: min={pe_min}, max={pe_max}，要求: 0 <= min < max")
        if pb_min < 0 or pb_max <= pb_min:
            raise ValueError(f"PB参数无效: min={pb_min}, max={pb_max}，要求: 0 <= min < max")
    
    # 验证动态权重参数
    if use_dynamic_weighting:
        if not isinstance(n1_threshold, int) or n1_threshold < 1:
            raise ValueError(f"n1_threshold 必须为正整数，当前值: {n1_threshold}")
        if not isinstance(n2_threshold, int) or n2_threshold <= n1_threshold:
            raise ValueError(
                f"n2_threshold ({n2_threshold}) 必须大于 n1_threshold ({n1_threshold})"
            )
        if not (0 <= signal_score_threshold <= 100):
            raise ValueError(
                f"signal_score_threshold 必须在 [0, 100] 范围内，当前值: {signal_score_threshold}"
            )
    
    print("="*80)
    print("ATR因子计算 - 参数验证通过")
    print("="*80)
    
    # ===== 股票池确定 =====
    # 验证stock_codes和index_code互斥
    if stock_codes is not None and index_code is not None:
        raise ValueError("stock_codes 和 index_code 不能同时指定，请只选择一个")
    
    # 股票池处理
    try:
        if stock_codes is None and index_code is None:
            # 既没指定股票池，也没指定指数，使用全市场
            try:
                all_daily = data_manager.load_data('daily', start_date=start_date, end_date=end_date, cleaned=True)
                if all_daily is None or all_daily.empty:
                    raise ValueError("无法获取全市场数据")
                stock_codes = all_daily['ts_code'].unique().tolist()
                print(f"✅ 使用全市场股票池: {len(stock_codes)} 只股票")
            except Exception as e:
                warnings.warn(f"无法获取全市场数据: {str(e)}，使用默认样本股票池", UserWarning)
                stock_codes = ['000001.SZ', '000002.SZ', '000858.SZ', '600000.SH', '600036.SH', '600519.SH']
                print(f"⚠️  使用默认样本股票池: {len(stock_codes)} 只股票")
        elif index_code is not None:
            # 使用指数成分股
            print(f"\n{'='*60}")
            print(f"指数成分股选择")
            print(f"{'='*60}")
            try:
                stock_codes = get_index_components(data_manager, index_code=index_code)
                if not stock_codes:
                    raise ValueError(f"无法获取指数 {index_code} 的成分股数据")
            except (FileNotFoundError, ValueError, IOError) as e:
                raise ValueError(f"获取指数成分股失败: {str(e)}") from e
            
            print(f"{'='*60}\n")
        else:
            # 使用指定的股票池
            if not isinstance(stock_codes, list) or len(stock_codes) == 0:
                raise ValueError("stock_codes 必须为非空列表")
            print(f"✅ 使用指定股票池: {len(stock_codes)} 只股票")
    except ValueError as e:
        # 重新抛出ValueError
        raise
    except Exception as e:
        raise RuntimeError(f"股票池确定过程发生未预期错误: {str(e)}") from e

    # 计算历史数据缓冲期：向前扩展以确保有足够的数据计算初始ATR和选股信号
    # 需要考虑：
    # 1. ATR计算需要period天数据
    # 2. ATR_long_ma需要20天ATR数据（总共period+20天）
    # 3. 价格位置信号需要2*period天数据
    # 4. 安全起见，再增加50%缓冲
    buffer_days = int(period * 3.5)  # period + 20 + period + 余量
    start_date_extended = pd.to_datetime(start_date) - pd.Timedelta(days=buffer_days)
    start_date_extended = start_date_extended.strftime('%Y-%m-%d')
    
    print(f"\n{'='*60}")
    print(f"ATR因子计算 - 数据加载")
    print(f"{'='*60}")
    print(f"  目标时间范围: {start_date} ~ {end_date}")
    print(f"  历史缓冲天数: {buffer_days} 天")
    print(f"  实际加载起始: {start_date_extended}")
    print(f"  ATR计算周期: {period} 天")
    if use_selection_signals:
        print(f"  选股信号模式: 启用（需要更多历史数据）")
    else:
        print(f"  选股信号模式: 禁用（仅计算基础ATR）")

    # 加载日线数据（使用扩展的起始日期）
    try:
        daily = data_manager.load_data('daily', start_date=start_date_extended, end_date=end_date, stock_codes=stock_codes)
        if daily is None or daily.empty:
            raise ValueError(
                f"无法获取日线行情数据\n"
                f"时间范围: {start_date_extended} ~ {end_date}\n"
                f"股票数量: {len(stock_codes) if stock_codes else 0}"
            )
        
        # 验证数据质量
        required_cols = ['ts_code', 'trade_date', 'high', 'low', 'close', 'open', 'vol']
        missing_cols = [col for col in required_cols if col not in daily.columns]
        if missing_cols:
            raise ValueError(f"日线数据缺少必要字段: {missing_cols}")
        
        print(f"  ✅ 加载日线数据: {len(daily)} 条记录")
        
    except ValueError as e:
        raise
    except Exception as e:
        raise RuntimeError(f"加载日线数据时发生未预期错误: {str(e)}") from e

    # 如果需要估值筛选，加载daily_basic数据
    if filter_valuation:
        print(f"  估值筛选模式: 启用 (PE: {pe_min}-{pe_max}, PB: {pb_min}-{pb_max})")
        
        try:
            daily_basic = data_manager.load_data('daily_basic', start_date=start_date_extended, end_date=end_date, stock_codes=stock_codes)
            if daily_basic is None or daily_basic.empty:
                warnings.warn("无法加载daily_basic数据，估值筛选将被跳过", UserWarning)
                filter_valuation = False
            else:
                # 检查必要字段
                if 'pe_ttm' not in daily_basic.columns or 'pb' not in daily_basic.columns:
                    warnings.warn("daily_basic缺少pe_ttm或pb字段，估值筛选将被跳过", UserWarning)
                    filter_valuation = False
                else:
                    # 统一日期格式
                    daily_basic = daily_basic.copy()
                    daily_basic['trade_date'] = pd.to_datetime(daily_basic['trade_date'], errors='coerce')
                    if daily_basic['trade_date'].isna().any():
                        daily_basic['trade_date'] = pd.to_datetime(
                            daily_basic['trade_date'].astype(str), 
                            format='%Y%m%d', 
                            errors='coerce'
                        )
                    daily_basic = daily_basic.dropna(subset=['trade_date'])
                    
                    # 合并估值数据到日线数据
                    daily = pd.merge(
                        daily,
                        daily_basic[['ts_code', 'trade_date', 'pe_ttm', 'pb']],
                        on=['ts_code', 'trade_date'],
                        how='left'
                    )
                    print(f"  ✅ 成功加载估值数据")
        except Exception as e:
            warnings.warn(f"加载估值数据失败: {str(e)}，估值筛选将被跳过", UserWarning)
            filter_valuation = False

    # 数据质量检查
    required_columns = ['trade_date', 'ts_code', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in daily.columns]
    if missing_columns:
        raise ValueError(f'日线数据缺少必要列: {missing_columns}')

    # 统一日期格式为datetime并排序
    daily = daily.copy()
    
    # 第一步：尝试直接转换为datetime
    daily['trade_date'] = pd.to_datetime(daily['trade_date'], errors='coerce')
    
    # 第二步：如果有转换失败的，尝试YYYYMMDD格式
    if daily['trade_date'].isna().any():
        na_count = daily['trade_date'].isna().sum()
        print(f"\n⚠️  发现 {na_count} 条日期格式异常，尝试YYYYMMDD格式修复...")
        
        # 只对失败的记录重新转换
        mask = daily['trade_date'].isna()
        daily.loc[mask, 'trade_date'] = pd.to_datetime(
            daily.loc[mask, 'trade_date'].astype(str), 
            format='%Y%m%d', 
            errors='coerce'
        )
        
        # 统计修复结果
        still_na = daily['trade_date'].isna().sum()
        fixed_count = na_count - still_na
        if fixed_count > 0:
            print(f"✅ 成功修复 {fixed_count} 条日期")
        if still_na > 0:
            print(f"⚠️  仍有 {still_na} 条日期无法解析，将被删除")
    
    # 删除日期为空的记录
    before_drop = len(daily)
    daily = daily.dropna(subset=['trade_date'])
    after_drop = len(daily)
    
    if before_drop > after_drop:
        print(f"已删除 {before_drop - after_drop} 条日期无效的记录")
    
    if daily.empty:
        raise ValueError('日期转换后无有效数据')
    
    # 全局排序（重要！确保数据按时间顺序）
    daily = daily.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    
    print(f"\n数据加载完成:")
    print(f"  实际时间范围: {daily['trade_date'].min().strftime('%Y-%m-%d')} ~ {daily['trade_date'].max().strftime('%Y-%m-%d')}")
    print(f"  总记录数: {len(daily):,}")
    print(f"  股票数量: {daily['ts_code'].nunique()}")
    print(f"  交易日数: {daily['trade_date'].nunique()}")
    
    # 检查日期范围覆盖
    actual_start = daily['trade_date'].min()
    actual_end = daily['trade_date'].max()
    expected_start = pd.to_datetime(start_date_extended)
    expected_end = pd.to_datetime(end_date)
    
    if actual_start > expected_start:
        print(f"  ⚠️  实际开始日期晚于预期（预期: {start_date_extended}）")
    if actual_end < expected_end:
        print(f"  ⚠️  实际结束日期早于预期（预期: {end_date}）")
    
    # 价格数据质量检查
    original_count = len(daily)
    daily = daily[(daily['high'] > 0) & (daily['low'] > 0) & (daily['close'] > 0)]
    filtered_count = len(daily)
    
    if filtered_count < original_count:
        print(f"\n数据质量检查:")
        print(f"  原始记录: {original_count:,}")
        print(f"  有效记录: {filtered_count:,}")
        print(f"  过滤记录: {original_count - filtered_count:,} ({(original_count - filtered_count)/original_count*100:.1f}%)")
    
    if daily.empty:
        raise ValueError('所有记录的价格数据无效')

    # 涨停/ST股票过滤
    if filter_limit or filter_st:
        print(f"\n{'='*60}")
        print(f"股票筛选过滤")
        print(f"{'='*60}")
        
        before_filter = len(daily)
        filter_stats = {}
        
        # 1. 涨停/跌停过滤
        if filter_limit:
            print(f"\n1. 涨停/跌停过滤:")
            
            # 检查pct_chg字段
            if 'pct_chg' not in daily.columns:
                print(f"  ⚠️  警告: 缺少 pct_chg 字段，跳过涨跌停过滤")
            else:
                # 当日涨停
                limit_up_today = daily['pct_chg'] > 9.8
                limit_up_count = limit_up_today.sum()
                
                # 当日跌停
                limit_down_today = daily['pct_chg'] < -9.8
                limit_down_count = limit_down_today.sum()
                
                # 次日开盘涨停（需要计算next_open）
                daily['next_open'] = daily.groupby('ts_code')['open'].shift(-1)
                limit_up_next = ((daily['next_open'] / daily['close'] - 1) > 0.098)
                limit_up_next_count = limit_up_next.sum()
                
                print(f"  当日涨停: {limit_up_count:,} 条 ({limit_up_count/before_filter*100:.2f}%)")
                print(f"  当日跌停: {limit_down_count:,} 条 ({limit_down_count/before_filter*100:.2f}%)")
                print(f"  次日开盘涨停: {limit_up_next_count:,} 条 ({limit_up_next_count/before_filter*100:.2f}%)")
                
                # 过滤涨跌停
                valid_mask = ~(limit_up_today | limit_down_today | limit_up_next.fillna(False))
                daily = daily[valid_mask]
                
                filter_stats['limit'] = before_filter - len(daily)
                print(f"  ✅ 已过滤 {filter_stats['limit']:,} 条涨跌停记录")
        
        # 2. ST股票过滤
        if filter_st:
            print(f"\n2. ST/退市风险股票过滤:")
            
            # 尝试加载股票基本信息
            try:
                # 方法1: 从stock_basic加载股票名称
                stock_basic = data_manager.load_data('stock_basic')
                
                if stock_basic is not None and not stock_basic.empty and 'name' in stock_basic.columns:
                    # 识别ST股票
                    st_patterns = ['ST', '*ST', 'S*ST', 'SST', 'S', '退市', '退']
                    st_stocks = stock_basic[
                        stock_basic['name'].str.contains('|'.join(st_patterns), na=False)
                    ]['ts_code'].unique().tolist()
                    
                    if st_stocks:
                        st_before = len(daily)
                        daily = daily[~daily['ts_code'].isin(st_stocks)]
                        st_filtered = st_before - len(daily)
                        
                        filter_stats['st'] = st_filtered
                        print(f"  识别到 {len(st_stocks)} 只ST/退市风险股票")
                        print(f"  ✅ 已过滤 {st_filtered:,} 条ST股票记录 ({st_filtered/st_before*100:.2f}%)")
                    else:
                        print(f"  ✅ 未发现ST/退市风险股票")
                else:
                    print(f"  ⚠️  警告: 无法加载股票基本信息，跳过ST过滤")
                    
            except Exception as e:
                print(f"  ⚠️  警告: ST过滤失败 ({str(e)})，跳过ST过滤")
        
        # 汇总过滤统计
        after_filter = len(daily)
        total_filtered = before_filter - after_filter
        
        print(f"\n过滤汇总:")
        print(f"  原始记录: {before_filter:,}")
        print(f"  过滤后: {after_filter:,}")
        print(f"  总过滤: {total_filtered:,} ({total_filtered/before_filter*100:.2f}%)")
        
        if 'limit' in filter_stats:
            print(f"    - 涨跌停: {filter_stats['limit']:,}")
        if 'st' in filter_stats:
            print(f"    - ST股票: {filter_stats['st']:,}")
        
        print(f"{'='*60}\n")
        
        if daily.empty:
            raise ValueError('过滤后无有效数据')

    # 估值筛选
    if filter_valuation and 'pe_ttm' in daily.columns and 'pb' in daily.columns:
        print(f"\n{'='*60}")
        print(f"估值指标筛选")
        print(f"{'='*60}")
        
        before_valuation = len(daily)
        
        # PE-TTM筛选
        print(f"\n1. PE-TTM (市盈率) 筛选:")
        print(f"   筛选范围: [{pe_min}, {pe_max}]")
        
        pe_na_count = daily['pe_ttm'].isna().sum()
        pe_valid = daily['pe_ttm'].notna()
        pe_in_range = (daily['pe_ttm'] >= pe_min) & (daily['pe_ttm'] <= pe_max)
        pe_below = (daily['pe_ttm'] < pe_min) & daily['pe_ttm'].notna()
        pe_above = (daily['pe_ttm'] > pe_max) & daily['pe_ttm'].notna()
        
        print(f"   PE为空: {pe_na_count:,} 条 ({pe_na_count/before_valuation*100:.2f}%)")
        print(f"   PE<{pe_min} (亏损/异常): {pe_below.sum():,} 条 ({pe_below.sum()/before_valuation*100:.2f}%)")
        print(f"   PE>{pe_max} (高估值): {pe_above.sum():,} 条 ({pe_above.sum()/before_valuation*100:.2f}%)")
        print(f"   PE合理范围: {pe_in_range.sum():,} 条 ({pe_in_range.sum()/before_valuation*100:.2f}%)")
        
        # PB筛选
        print(f"\n2. PB (市净率) 筛选:")
        print(f"   筛选范围: [{pb_min}, {pb_max}]")
        
        pb_na_count = daily['pb'].isna().sum()
        pb_valid = daily['pb'].notna()
        pb_in_range = (daily['pb'] >= pb_min) & (daily['pb'] <= pb_max)
        pb_below = (daily['pb'] < pb_min) & daily['pb'].notna()
        pb_above = (daily['pb'] > pb_max) & daily['pb'].notna()
        
        print(f"   PB为空: {pb_na_count:,} 条 ({pb_na_count/before_valuation*100:.2f}%)")
        print(f"   PB<{pb_min} (破净/异常): {pb_below.sum():,} 条 ({pb_below.sum()/before_valuation*100:.2f}%)")
        print(f"   PB>{pb_max} (高估值): {pb_above.sum():,} 条 ({pb_above.sum()/before_valuation*100:.2f}%)")
        print(f"   PB合理范围: {pb_in_range.sum():,} 条 ({pb_in_range.sum()/before_valuation*100:.2f}%)")
        
        # 综合筛选（PE和PB都要满足）
        print(f"\n3. 综合估值筛选:")
        valuation_valid = pe_in_range & pb_in_range
        daily = daily[valuation_valid]
        after_valuation = len(daily)
        
        filtered_count = before_valuation - after_valuation
        print(f"   原始记录: {before_valuation:,}")
        print(f"   筛选后: {after_valuation:,}")
        print(f"   过滤记录: {filtered_count:,} ({filtered_count/before_valuation*100:.2f}%)")
        
        if after_valuation > 0:
            # 输出筛选后的估值统计
            pe_stats = daily['pe_ttm'].describe()
            pb_stats = daily['pb'].describe()
            
            print(f"\n4. 筛选后估值分布:")
            print(f"   PE-TTM统计: 均值={pe_stats['mean']:.2f}, 中位数={pe_stats['50%']:.2f}, "
                  f"范围=[{pe_stats['min']:.2f}, {pe_stats['max']:.2f}]")
            print(f"   PB统计: 均值={pb_stats['mean']:.2f}, 中位数={pb_stats['50%']:.2f}, "
                  f"范围=[{pb_stats['min']:.2f}, {pb_stats['max']:.2f}]")
        
        print(f"{'='*60}\n")
        
        if daily.empty:
            raise ValueError('估值筛选后无有效数据，请放宽PE/PB范围')

    print(f"\n{'='*60}")
    print(f"开始计算ATR因子及选股信号")
    print(f"{'='*60}\n")

    # 按股票分组计算ATR及选股信号
    factor_parts = []
    skipped_stocks = 0
    error_stocks = []
    
    total_stocks = daily['ts_code'].nunique()
    print(f"待计算股票数: {total_stocks}")
    
    for idx, code in enumerate(daily['ts_code'].unique(), 1):
        try:
            stock_data = daily[daily['ts_code'] == code].sort_values('trade_date')
            
            # 检查数据是否足够
            if len(stock_data) < period:
                skipped_stocks += 1
                if skipped_stocks <= 5:  # 只显示前5个警告
                    warnings.warn(
                        f"股票 {code} 数据不足 ({len(stock_data)} < {period})，已跳过",
                        UserWarning
                    )
                continue
            
            # 进度显示（每10%或每100只股票显示一次）
            if idx % max(1, total_stocks // 10) == 0 or idx % 100 == 0:
                print(f"  进度: {idx}/{total_stocks} ({idx/total_stocks*100:.1f}%) - 当前: {code}")
            
            # ATR计算
            try:
                min_required = period * 2 if use_selection_signals else period
                if len(stock_data) < min_required:
                    skipped_stocks += 1
                    continue
                    
                # 计算True Range
                stock_data = stock_data.copy()
                stock_data['HL'] = stock_data['high'] - stock_data['low']
                stock_data['HC'] = abs(stock_data['high'] - stock_data['close'].shift(1))
                stock_data['LC'] = abs(stock_data['low'] - stock_data['close'].shift(1))
                stock_data['TR'] = stock_data[['HL', 'HC', 'LC']].max(axis=1)
                
                # 计算ATR
                stock_data['ATR'] = stock_data['TR'].rolling(window=period, min_periods=period).mean()
                
                # 标准化处理：使用过去period天的收盘价均值进行归一化
                stock_data['price_mean'] = stock_data['close'].rolling(window=period, min_periods=period).mean()
                stock_data['ATR_norm'] = stock_data['ATR'] / stock_data['price_mean']
                
                if use_selection_signals:
                    # ========== 选股信号1: ATR扩张信号 ==========
                    stock_data['ATR_ma'] = stock_data['ATR_norm'].rolling(window=period).mean()
                    stock_data['ATR_expansion'] = (stock_data['ATR_norm'] / stock_data['ATR_ma'] - 1) * 100
                    
                    # ========== 选股信号2: 价格位置信号 ==========
                    stock_data['price_high'] = stock_data['high'].rolling(window=period*2).max()
                    stock_data['price_low'] = stock_data['low'].rolling(window=period*2).min()
                    stock_data['price_position'] = (stock_data['close'] - stock_data['price_low']) / \
                                                   (stock_data['price_high'] - stock_data['price_low'])
                    
                    # ========== 选股信号3: 成交量确认信号 ==========
                    if 'vol' in stock_data.columns:
                        stock_data['vol_ma'] = stock_data['vol'].rolling(window=period).mean()
                        stock_data['vol_ratio'] = stock_data['vol'] / stock_data['vol_ma']
                    else:
                        stock_data['vol_ratio'] = 1.0
                    
                    # ========== 选股信号4: ATR趋势信号 ==========
                    stock_data['ATR_short_ma'] = stock_data['ATR_norm'].rolling(window=5).mean()
                    stock_data['ATR_long_ma'] = stock_data['ATR_norm'].rolling(window=20).mean()
                    stock_data['ATR_trend'] = (stock_data['ATR_short_ma'] > stock_data['ATR_long_ma']).astype(int)
                    
                    # ========== 选股信号5: 价格突破信号 ==========
                    stock_data['prev_high'] = stock_data['high'].rolling(window=period).max().shift(1)
                    stock_data['price_breakout'] = (stock_data['close'] > stock_data['prev_high']).astype(int)
                    
                    # 提取所有信号数据
                    signal_cols = ['trade_date', 'ts_code', 'ATR_norm', 'ATR_expansion', 
                                  'price_position', 'vol_ratio', 'ATR_trend', 'price_breakout']
                    valid_data = stock_data[signal_cols].dropna()
                else:
                    # 只返回基础ATR值
                    valid_data = stock_data[['trade_date', 'ts_code', 'ATR_norm']].dropna()
                
                if not valid_data.empty:
                    factor_parts.append(valid_data)
                    
            except Exception as e:
                error_stocks.append((code, str(e)))
                if len(error_stocks) <= 5:  # 只显示前5个错误
                    warnings.warn(f"股票 {code} 计算失败: {str(e)}", UserWarning)
                continue
                
        except Exception as e:
            error_stocks.append((code, str(e)))
            warnings.warn(f"处理股票 {code} 时发生错误: {str(e)}", UserWarning)
            continue

    # 汇总统计
    print(f"\n{'='*60}")
    print(f"ATR计算完成统计")
    print(f"{'='*60}")
    print(f"  总股票数: {total_stocks}")
    print(f"  成功计算: {len(factor_parts)}")
    print(f"  数据不足跳过: {skipped_stocks}")
    print(f"  计算错误: {len(error_stocks)}")
    if error_stocks and len(error_stocks) > 5:
        print(f"  (仅显示前5个错误，共 {len(error_stocks)} 个)")
    
    # 合并所有股票的数据
    if not factor_parts:
        raise ValueError(
            f"没有足够的数据计算ATR因子\n"
            f"  跳过: {skipped_stocks} 只 (数据不足)\n"
            f"  错误: {len(error_stocks)} 只\n"
            f"建议: 1) 检查数据质量 2) 减小period参数 3) 扩大日期范围"
        )
    
    try:
        factor_df = pd.concat(factor_parts, axis=0, ignore_index=True)
    except Exception as e:
        raise RuntimeError(f"合并因子数据失败: {str(e)}") from e
    
    if use_selection_signals:
        print("\n" + "="*60)
        print("构建ATR选股信号")
        print("="*60)
        
        # 按日期计算截面信号并应用动态权重
        def calculate_cross_sectional_signals(group):
            """计算每日截面的选股信号（动态权重版本）"""
            n = len(group)  # 当日样本数量
            
            # ========== 第一步: 计算各项信号得分 ==========
            
            # 信号1: ATR扩张强度分数 (0-25分)
            atr_exp_score = np.where(group['ATR_expansion'] > 0, 
                                    np.clip(group['ATR_expansion'] / 2, 0, 25), 0)
            
            # 信号2: 价格位置分数 (0-25分)
            price_pos_score = np.where(group['price_position'] > price_position_threshold,
                                      25 * (group['price_position'] - price_position_threshold) / (1 - price_position_threshold),
                                      0)
            
            # 信号3: 成交量确认分数 (0-20分)
            vol_score = np.where(group['vol_ratio'] > volume_ratio_threshold,
                                20 * np.clip((group['vol_ratio'] - volume_ratio_threshold) / volume_ratio_threshold, 0, 1),
                                0)
            
            # 信号4: ATR趋势分数 (0-15分)
            trend_score = group['ATR_trend'] * 15
            
            # 信号5: 价格突破分数 (0-15分)
            breakout_score = group['price_breakout'] * 15
            
            # 综合信号得分 (0-100分)
            group['signal_score'] = (atr_exp_score + price_pos_score + 
                                    vol_score + trend_score + breakout_score)
            
            # ATR相对强度 (在当日所有股票中的百分位排名)
            group['ATR_percentile'] = group['ATR_norm'].rank(pct=True) * 100
            
            # ========== 第二步: 应用动态权重机制 ==========
            
            if use_dynamic_weighting:
                # 动态权重：根据样本数量调整筛选策略
                
                if n < n1_threshold:
                    # 样本数不足：不筛选，给所有股票基础权重
                    group['factor'] = group['signal_score']
                    group['selection_reason'] = 'insufficient_samples'
                    
                elif n < n2_threshold:
                    # 样本数适中：单维度筛选（只看信号得分）
                    # 信号得分达标的股票权重=1.0，否则=0.3
                    signal_weight = np.where(
                        group['signal_score'] >= signal_score_threshold,
                        1.0,
                        0.3
                    )
                    group['factor'] = group['signal_score'] * signal_weight
                    group['selection_reason'] = 'single_dimension'
                    
                else:
                    # 样本数充足：双维度精选（信号得分 + ATR相对强度）
                    # 计算信号得分和ATR分位数的中位数
                    signal_median = group['signal_score'].median()
                    atr_in_range = (
                        (group['ATR_percentile'] >= atr_percentile_low) & 
                        (group['ATR_percentile'] <= atr_percentile_high)
                    )
                    
                    # 三档权重分配
                    # 最优组：高信号得分 + ATR在合理区间
                    optimal_mask = (group['signal_score'] > signal_median) & atr_in_range
                    # 次优组：高信号得分 或 ATR在合理区间（满足一个）
                    suboptimal_mask = (
                        ((group['signal_score'] > signal_median) & ~atr_in_range) |
                        ((group['signal_score'] <= signal_median) & atr_in_range)
                    )
                    # 其他组：都不满足
                    
                    group['factor'] = np.where(
                        optimal_mask,
                        group['signal_score'] * 1.5,  # 最优权重1.5
                        np.where(
                            suboptimal_mask,
                            group['signal_score'] * 1.0,  # 次优权重1.0
                            group['signal_score'] * 0.3   # 其他权重0.3
                        )
                    )
                    group['selection_reason'] = np.where(
                        optimal_mask, 'optimal',
                        np.where(suboptimal_mask, 'suboptimal', 'other')
                    )
                    
            else:
                # 固定权重：原始逻辑（仅基于ATR分位数）
                atr_weight = np.where(
                    (group['ATR_percentile'] >= atr_percentile_low) & 
                    (group['ATR_percentile'] <= atr_percentile_high),
                    1.0,
                    0.5
                )
                group['factor'] = group['signal_score'] * atr_weight
                group['selection_reason'] = 'fixed_weight'
            
            return group
        
        factor_df = factor_df.groupby('trade_date', group_keys=False).apply(calculate_cross_sectional_signals)
        
        # 打印信号统计
        print(f"\n选股信号统计:")
        print(f"  平均信号得分: {factor_df['signal_score'].mean():.2f}")
        print(f"  信号得分范围: [{factor_df['signal_score'].min():.2f}, {factor_df['signal_score'].max():.2f}]")
        print(f"  高分信号(>{signal_score_threshold}分)占比: {(factor_df['signal_score'] > signal_score_threshold).mean()*100:.1f}%")
        print(f"  ATR扩张信号触发率: {(factor_df['ATR_expansion'] > 0).mean()*100:.1f}%")
        print(f"  价格突破信号触发率: {factor_df['price_breakout'].mean()*100:.1f}%")
        print(f"  成交量放大信号触发率: {(factor_df['vol_ratio'] > volume_ratio_threshold).mean()*100:.1f}%")
        
        if use_dynamic_weighting and 'selection_reason' in factor_df.columns:
            print(f"\n动态权重分配统计:")
            reason_counts = factor_df['selection_reason'].value_counts()
            for reason, count in reason_counts.items():
                pct = count / len(factor_df) * 100
                reason_map = {
                    'insufficient_samples': '样本不足(全部入选)',
                    'single_dimension': '单维筛选(信号得分)',
                    'optimal': '最优组(双维精选)',
                    'suboptimal': '次优组(单维达标)',
                    'other': '其他组(降权)',
                    'fixed_weight': '固定权重'
                }
                print(f"  {reason_map.get(reason, reason)}: {count:,} ({pct:.1f}%)")
        
        print("="*60 + "\n")
        
        result = factor_df.set_index(['trade_date', 'ts_code'])[['factor']]
    else:
        result = factor_df.set_index(['trade_date', 'ts_code'])[['ATR_norm']].rename(columns={'ATR_norm': 'factor'})
    
    # 只保留在指定日期范围内的数据（去除缓冲期数据）
    result = result.loc[result.index.get_level_values('trade_date') >= pd.to_datetime(start_date)]
    result = result.loc[result.index.get_level_values('trade_date') <= pd.to_datetime(end_date)]
    
    print(f"\n{'='*60}")
    print(f"ATR因子计算完成")
    print(f"{'='*60}")
    print(f"  有效记录数: {len(result):,}")
    print(f"  覆盖股票数: {result.index.get_level_values('ts_code').nunique()}")
    print(f"  覆盖交易日: {result.index.get_level_values('trade_date').nunique()}")
    print(f"  因子值范围: [{result['factor'].min():.4f}, {result['factor'].max():.4f}]")
    print(f"  因子均值: {result['factor'].mean():.4f}")
    print(f"{'='*60}\n")
    
    return result

def run_atr_factor_backtest(
    start_date: str = '2024-01-01',
    end_date: str = '2024-02-29',
    stock_codes: Optional[List[str]] = None,
    index_code: Optional[str] = None,
    rebalance_freq: str = 'weekly',
    transaction_cost: float = 0.0003,
    long_direction: str = 'high',
    atr_period: int = 14,
    use_selection_signals: bool = True,
    atr_percentile_low: float = 30,
    atr_percentile_high: float = 70,
    price_position_threshold: float = 0.7,
    volume_ratio_threshold: float = 1.2,
    filter_limit: bool = True,
    filter_st: bool = True,
    filter_valuation: bool = True,
    pe_min: float = 0.0,
    pe_max: float = 100.0,
    pb_min: float = 0.0,
    pb_max: float = 10.0,
    use_dynamic_weighting: bool = True,
    signal_score_threshold: int = 50,
    n1_threshold: int = 20,
    n2_threshold: int = 50
) -> dict:
    """
    ATR因子策略回测
    
    Parameters
    ----------
    start_date, end_date : str
        回测周期
    stock_codes : Optional[List[str]]
        股票代码列表,与index_code互斥
    index_code : Optional[str]
        指数代码,使用指数成分股作为股票池,与stock_codes互斥
        推荐: '000852.SH' (中证1000) 用于小盘成长策略
    rebalance_freq : str
        调仓频率
    transaction_cost : float
        单边交易成本
    long_direction : str
        多头方向
    atr_period : int
        ATR计算周期
    use_selection_signals : bool
        是否使用选股信号系统,默认True
        True: 使用综合选股信号(推荐)
        False: 仅使用原始ATR值
    atr_percentile_low : float
        ATR分位数下限,默认30%
    atr_percentile_high : float
        ATR分位数上限,默认70%
    price_position_threshold : float
        价格位置阈值,默认0.7
    volume_ratio_threshold : float
        成交量放大倍数,默认1.2
    filter_limit : bool
        是否过滤涨停/跌停股票,默认True
    filter_st : bool
        是否过滤ST/退市风险股票,默认True
    filter_valuation : bool
        是否使用估值指标筛选,默认True
    pe_min : float
        市盈率(PE-TTM)最小值,默认0.0
    pe_max : float
        市盈率(PE-TTM)最大值,默认100.0
    pb_min : float
        市净率(PB)最小值,默认0.0
    pb_max : float
        市净率(PB)最大值,默认10.0
    use_dynamic_weighting : bool
        是否使用动态权重机制,默认True
    signal_score_threshold : int
        信号得分阈值,默认50分
    n1_threshold : int
        动态筛选第一阈值,默认20
    n2_threshold : int
        动态筛选第二阈值,默认50
        
    Returns
    -------
    dict
        回测结果字典
        
    Raises
    ------
    ValueError
        参数验证失败或数据异常
    RuntimeError
        回测执行过程中发生未预期错误
    """
    try:
        # 验证日期格式
        start_date = validate_date_format(start_date, 'start_date')
        end_date = validate_date_format(end_date, 'end_date')
    except (ValueError, RuntimeError) as e:
        raise ValueError(f"日期参数验证失败: {str(e)}") from e
    
    # 验证日期逻辑
    if pd.to_datetime(start_date) > pd.to_datetime(end_date):
        raise ValueError(f"开始日期 ({start_date}) 不能晚于结束日期 ({end_date})")
    
    # 初始化数据管理器
    try:
        data_manager = DataManager()
    except Exception as e:
        raise RuntimeError(f"初始化数据管理器失败: {str(e)}") from e

    # 计算因子(包含选股信号)
    print(f"\n{'='*60}")
    print(f"ATR因子计算 - {'使用选股信号系统' if use_selection_signals else '基础ATR值'}")
    print(f"{'='*60}")
    
    try:
        factor_data = calculate_atr_factor(
            data_manager=data_manager,
            start_date=start_date,
            end_date=end_date,
            stock_codes=stock_codes,
            index_code=index_code,
            period=atr_period,
            use_selection_signals=use_selection_signals,
            atr_percentile_low=atr_percentile_low,
            atr_percentile_high=atr_percentile_high,
            price_position_threshold=price_position_threshold,
            volume_ratio_threshold=volume_ratio_threshold,
            filter_limit=filter_limit,
            filter_st=filter_st,
            filter_valuation=filter_valuation,
            pe_min=pe_min,
            pe_max=pe_max,
            pb_min=pb_min,
            pb_max=pb_max,
            use_dynamic_weighting=use_dynamic_weighting,
            signal_score_threshold=signal_score_threshold,
            n1_threshold=n1_threshold,
            n2_threshold=n2_threshold
        )
    except (ValueError, RuntimeError) as e:
        raise ValueError(f"ATR因子计算失败: {str(e)}") from e
    except Exception as e:
        raise RuntimeError(f"ATR因子计算过程发生未预期错误: {str(e)}") from e

    # 使用BacktestEngine主路径
    try:
        from backtest_engine.engine import BacktestEngine
    except ImportError as e:
        raise ImportError(
            f"无法导入BacktestEngine: {str(e)}\n"
            f"请确保backtest_engine模块已正确安装"
        ) from e
    
    try:
        engine = BacktestEngine(
            data_manager=data_manager,
            fee=transaction_cost,
            long_direction=long_direction,
            rebalance_freq=rebalance_freq,
            factor_name='factor',
        )
    except Exception as e:
        raise RuntimeError(f"初始化回测引擎失败: {str(e)}") from e
    
    # 直接设置因子数据
    engine.factor_data = factor_data
    
    # 准备收益率数据
    try:
        stock_list = factor_data.index.get_level_values('ts_code').unique().tolist()
        stock_data = data_manager.load_data(
            'daily',
            start_date=start_date,
            end_date=end_date,
            stock_codes=stock_list
        )
        
        if stock_data is None or stock_data.empty:
            raise ValueError(
                f"无法加载用于回测的股票数据\n"
                f"时间范围: {start_date} ~ {end_date}\n"
                f"股票数量: {len(stock_list)}"
            )
        
        # 统一日期格式（确保与因子数据一致）
        stock_data = stock_data.copy()
        stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'], errors='coerce')
        if stock_data['trade_date'].isna().any():
            stock_data['trade_date'] = pd.to_datetime(
                stock_data['trade_date'].astype(str), 
                format='%Y%m%d', 
                errors='coerce'
            )
        stock_data = stock_data.dropna(subset=['trade_date'])
        
        if stock_data.empty:
            raise ValueError("日期转换后无有效数据")
            
    except ValueError as e:
        raise
    except Exception as e:
        raise RuntimeError(f"加载回测数据失败: {str(e)}") from e
    
    # 计算次日收益率和合并数据
    try:
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
        
        if engine.combined_data.empty:
            raise ValueError("因子数据与收益率数据合并后为空，请检查日期对齐")
        
        engine.combined_data.dropna(subset=['factor', 'next_return'], inplace=True)
        
        if engine.combined_data.empty:
            raise ValueError("清除缺失值后无有效数据")
            
    except ValueError as e:
        raise
    except Exception as e:
        raise RuntimeError(f"数据合并处理失败: {str(e)}") from e
    
    # 运行回测
    try:
        portfolio_returns = engine.run()
    except Exception as e:
        raise RuntimeError(f"回测执行失败: {str(e)}") from e

    # 计算基本业绩指标
    try:
        if not isinstance(portfolio_returns, pd.DataFrame):
            raise ValueError(f"回测结果类型异常: 期望DataFrame，实际{type(portfolio_returns)}")
        
        if 'Long_Only' not in portfolio_returns.columns:
            raise ValueError(
                f"回测结果缺少Long_Only列\n"
                f"可用列: {portfolio_returns.columns.tolist()}"
            )

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

        # 集成PerformanceAnalyzer
        analyzer = engine.get_performance_analysis()
        metrics_df = analyzer.calculate_metrics()
        ic_series = analyzer.ic_series

        return {
            'factor_data': engine.factor_data,
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
            'analysis_results': {
                'metrics': metrics_df,
                'ic_series': ic_series
            }
        }
    except Exception as e:
        raise RuntimeError(f"计算业绩指标失败: {str(e)}") from e

def main():
    """
    主函数：演示ATR因子计算和回测
    
    包含完整的异常处理和错误提示
    """
    print("ATR因子策略演示")
    print("=" * 50)

    try:
        # 配置参数
        config = {
            'start_date': '2015-09-30',
            'end_date': '2025-09-30',
            'rebalance_freq': 'weekly',
            'transaction_cost': 0.0003,
            'long_direction': 'high',  # 做多高因子值(高信号得分)
            'atr_period': 14,
            # 股票池选择（三选一）
            'stock_codes': None,            # 方式1: 手动指定股票列表
            'index_code': '000852.SH',      # 方式2: 使用指数成分股（中证1000）
            # 'index_code': None,           # 方式3: 使用全市场（耗时较长）
            # 选股信号参数
            'use_selection_signals': True,  # 启用选股信号系统
            'atr_percentile_low': 30,       # ATR分位数下限
            'atr_percentile_high': 70,      # ATR分位数上限
            'price_position_threshold': 0.7, # 价格位置阈值(70%)
            'volume_ratio_threshold': 1.2,   # 成交量放大倍数(1.2倍)
            # 股票筛选参数
            'filter_limit': True,           # 过滤涨停/跌停股票
            'filter_st': True,              # 过滤ST/退市风险股票
            'filter_valuation': True,       # 估值筛选
            'pe_min': 0.0,                  # PE-TTM最小值(排除亏损股)
            'pe_max': 100.0,                # PE-TTM最大值(排除高估值股)
            'pb_min': 0.0,                  # PB最小值(排除破净异常股)
            'pb_max': 10.0,                 # PB最大值(排除高PB股)
            # 动态权重参数（新增）
            'use_dynamic_weighting': True,  # 启用动态权重机制
            'signal_score_threshold': 50,   # 信号得分阈值
            'n1_threshold': 20,             # 样本数阈值1（不筛选）
            'n2_threshold': 50,             # 样本数阈值2（双维精选）
        }

        print("\n回测配置:")
        for key, value in config.items():
            if value is not None or key in ['stock_codes', 'index_code']:
                print(f"  {key}: {value}")
        
        print("\n" + "="*60)
        print("指数成分股说明:")
        print("="*60)
        print("可选指数代码:")
        print("  000300.SH - 沪深300 (大盘蓝筹)")
        print("  000905.SH - 中证500 (中盘股)")
        print("  000852.SH - 中证1000 (小盘成长，推荐用于ATR策略)")
        print("  000016.SH - 上证50 (超大盘)")
        print("  399006.SZ - 创业板指")
        print("="*60)
        
        print("\n" + "="*60)
        print("ATR选股信号说明:")
        print("="*60)
        print("【股票池选择】")
        if config.get('index_code'):
            print(f"  使用指数成分股: {config['index_code']}")
        elif config.get('stock_codes'):
            print(f"  使用指定股票池: {len(config['stock_codes'])} 只")
        else:
            print(f"  使用全市场股票（不推荐，耗时较长）")
        
        print("\n【动态权重机制】")
        if config.get('use_dynamic_weighting'):
            print(f"  启用状态: 开启（推荐）")
            print(f"  信号得分阈值: {config['signal_score_threshold']} 分")
            print(f"  样本数阈值: n1={config['n1_threshold']}, n2={config['n2_threshold']}")
            print(f"  权重策略:")
            print(f"    - 样本数 < {config['n1_threshold']}: 全部入选（基础权重）")
            print(f"    - {config['n1_threshold']} ≤ 样本数 < {config['n2_threshold']}: 单维筛选（信号得分）")
            print(f"    - 样本数 ≥ {config['n2_threshold']}: 双维精选（信号+ATR）")
            print(f"      · 最优组（权重1.5）: 高信号 + ATR合理区间")
            print(f"      · 次优组（权重1.0）: 满足一个条件")
            print(f"      · 其他组（权重0.3）: 都不满足")
        else:
            print(f"  启用状态: 关闭（使用固定权重）")
        
        print("\n【信号评分系统】")
        print("1. ATR扩张信号 (0-25分): ATR快速上升,波动性增加")
        print("2. 价格位置信号 (0-25分): 股价处于近期高位")
        print("3. 成交量确认信号 (0-20分): 放量突破确认")
        print("4. ATR趋势信号 (0-15分): ATR短期均线上穿长期均线")
        print("5. 价格突破信号 (0-15分): 突破近期高点")
        print("\n【股票筛选】")
        print("- 涨停/跌停过滤: 排除当日涨跌停及次日开盘涨停股票")
        print("- ST股票过滤: 排除ST、*ST、退市风险等特殊处理股票")
        print("- 估值筛选: PE-TTM ∈ [0, 100], PB ∈ [0, 10]")
        print("  · 排除亏损股票 (PE < 0)")
        print("  · 排除高估值股票 (PE > 100)")
        print("  · 排除破净异常股票 (PB < 0)")
        print("  · 排除极高市净率股票 (PB > 10)")
        print("\n综合得分 = 各信号得分之和 × 动态权重")
        print("="*60 + "\n")

        # 运行回测
        results = run_atr_factor_backtest(**config)

        # 结果总结
        print("\n回测结果总结 (Long_Only):")
        metrics = results['performance_metrics']
        print(f"  夏普比率: {metrics['sharpe_ratio']:.3f}")
        print(f"  总收益: {metrics['total_return']:.2%}")
        print(f"  年化收益: {metrics['annualized_return']:.2%}")
        print(f"  年化波动: {metrics['volatility']:.2%}")
        print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"  调仓次数: {metrics['rebalance_count']}")

        # IC分析
        if results['analysis_results']['ic_series'] is not None:
            ic = results['analysis_results']['ic_series']
            print(f"\nIC分析:")
            print(f"  IC均值: {ic.mean():.4f}")
            print(f"  IC标准差: {ic.std():.4f}")
            print(f"  ICIR: {ic.mean() / ic.std():.4f}" if ic.std() > 0 else "  ICIR: N/A")
            print(f"  IC>0占比: {(ic > 0).mean():.2%}")

        print("\nATR因子策略演示完成!")
        print("\n💡 策略特点:")
        print("  ✅ 捕捉波动性扩张带来的趋势机会")
        print("  ✅ 结合价格位置和成交量进行多维度确认")
        print("  ✅ 动态权重机制适应不同市场环境")
        print("  ✅ 优选处于合理波动区间的股票")
        print("  ✅ 自动过滤涨跌停和ST风险股票")
        print("  ✅ 估值筛选避免高风险股票")
        print("  ✅ 支持指数成分股选择,聚焦特定市场风格")
        print("  ✅ 适合趋势突破型策略")
        print("\n⚠️  风险提示:")
        print("  - ATR因子主要捕捉波动性变化,需结合趋势判断")
        print("  - 高波动期可能对应市场震荡,注意风险控制")
        print("  - 建议配合止损策略使用")
        print("\n📊 使用建议:")
        print("  - 动态权重: 根据样本数量自动调整选股标准")
        print("    · 样本少时放宽，样本多时严选")
        print("    · 提高策略的适应性和稳健性")
        print("  - 中证1000成分股: 适合小盘成长策略")
        print("  - 沪深300成分股: 适合大盘稳健策略")
        print("  - 估值筛选: PE-TTM [0,100], PB [0,10]")
        print("  - 可根据市场风格调整参数")

    except ValueError as e:
        print(f"\n❌ 参数错误: {e}")
        print("\n建议:")
        print("  1. 检查日期格式是否正确 (YYYY-MM-DD 或 YYYYMMDD)")
        print("  2. 检查参数范围是否合理")
        print("  3. 查看上方详细错误信息")
        raise
    except FileNotFoundError as e:
        print(f"\n❌ 文件未找到: {e}")
        print("\n建议:")
        print("  1. 检查数据文件是否存在")
        print("  2. 运行数据加载器: python data_manager/data_loader/...")
        print("  3. 检查文件路径是否正确")
        raise
    except ImportError as e:
        print(f"\n❌ 模块导入失败: {e}")
        print("\n建议:")
        print("  1. 检查backtest_engine模块是否已安装")
        print("  2. 检查Python路径配置")
        print("  3. 安装缺失的依赖包")
        raise
    except RuntimeError as e:
        print(f"\n❌ 运行时错误: {e}")
        print("\n这通常是由内部错误引起的，请查看详细堆栈信息")
        import traceback
        traceback.print_exc()
        raise
    except KeyboardInterrupt:
        print(f"\n\n⚠️  用户中断执行")
        print("程序已安全退出")
        return
    except Exception as e:
        print(f"\n❌ 未预期的错误: {type(e).__name__}: {e}")
        print("\n详细堆栈信息:")
        import traceback
        traceback.print_exc()
        print("\n建议:")
        print("  1. 检查上方错误堆栈，定位问题位置")
        print("  2. 确认数据文件完整性")
        print("  3. 尝试缩小日期范围或减少股票数量")
        print("  4. 如问题持续，请报告此错误")
        raise

if __name__ == "__main__":
    # 测试日期验证功能
    print("\n" + "="*60)
    print("测试日期验证功能")
    print("="*60)
    
    test_dates = [
        ('2024-01-01', '正确格式: YYYY-MM-DD'),
        ('20240101', '正确格式: YYYYMMDD'),
        ('2024/01/01', '错误格式: YYYY/MM/DD'),
        ('invalid', '错误格式: 非日期字符串'),
    ]
    
    for date_str, desc in test_dates:
        try:
            result = validate_date_format(date_str, 'test_date')
            print(f"✅ {desc}: '{date_str}' -> '{result}'")
        except ValueError as e:
            print(f"❌ {desc}: '{date_str}' -> 验证失败")
    
    print("="*60 + "\n")
    
    # 运行主函数
    main()
