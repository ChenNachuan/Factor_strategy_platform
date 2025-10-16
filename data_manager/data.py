import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple
import warnings
from functools import lru_cache

# ===== 数据管理器类 =====
class DataManager:
    """
    量化研究数据管理器
    
    专为因子计算和回测设计的数据加载和管理工具
    支持数据缓存、时间过滤、股票筛选等功能
    """
    
    def __init__(self, data_root: Optional[str] = None):
        """
        初始化数据管理器
        
        Args:
            data_root: 数据根目录，默认为当前目录
        """
        if data_root is None:
            self.data_root = Path(__file__).parent
        else:
            self.data_root = Path(data_root)
            
        self.clean_data_path = self.data_root / "clean_data"
        self.raw_data_path = self.data_root / "raw_data"
        
        # 数据缓存
        self._cache = {}
        self._load_message_shown = set()
        
        # 文件映射
        self.file_mapping = {
            'daily': 'a_stock_daily_data',
            'daily_basic': 'a_stock_daily_basic_data',
            'cashflow': 'a_stock_cashflow_data', 
            'balancesheet': 'a_stock_balancesheet_data',
            'income': 'a_stock_income_data',
            'index': 'a_index_daily_data',
            'index_weight': 'index_weight_data'
        }
    
    def load_data(self, 
                  data_type: str, 
                  cleaned: bool = True,
                  use_cache: bool = True,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  stock_codes: Optional[List[str]] = None,
                  verbose: bool = True) -> Optional[pd.DataFrame]:
        """
        加载数据 - 量化研究专用版本
        
        Args:
            data_type: 数据类型 ('daily', 'daily_basic', 'cashflow', 'balancesheet', 'income', 'index', 'index_weight')
            cleaned: 是否加载清洗后的数据
            use_cache: 是否使用缓存
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            stock_codes: 股票代码列表，None表示加载所有
            verbose: 是否显示加载信息
            
        Returns:
            pd.DataFrame: 加载的数据
        """
        # 生成缓存键
        cache_key = f"{data_type}_{cleaned}_{start_date}_{end_date}_{hash(tuple(stock_codes) if stock_codes else None)}"
        
        # 检查缓存
        if use_cache and cache_key in self._cache:
            if verbose:
                print(f" 从缓存加载 {data_type} 数据")
            return self._cache[cache_key].copy()
        
        # 加载原始数据
        df = self._load_raw_data(data_type, cleaned, verbose)
        if df is None:
            return None
            
        # 应用过滤条件
        df = self._apply_filters(df, start_date, end_date, stock_codes, data_type)
        
        # 缓存结果
        if use_cache:
            self._cache[cache_key] = df.copy()
            
        return df
    
    def _load_raw_data(self, data_type: str, cleaned: bool, verbose: bool) -> Optional[pd.DataFrame]:
        """加载原始数据文件"""
        if data_type not in self.file_mapping:
            raise ValueError(f"不支持的数据类型: {data_type}. 支持的类型: {list(self.file_mapping.keys())}")

        base_filename = self.file_mapping[data_type]
        
        if cleaned:
            filepath = self.clean_data_path / f"{base_filename}_clean.parquet"
        else:
            filepath = self.raw_data_path / f"{base_filename}.parquet"

        try:
            df = pd.read_parquet(filepath)
            
            # 确保日期列为datetime格式
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
            if 'end_date' in df.columns:
                df['end_date'] = pd.to_datetime(df['end_date'])
            if 'ann_date' in df.columns:
                df['ann_date'] = pd.to_datetime(df['ann_date'])
                
            if verbose:
                status = "清洗后" if cleaned else "原始"
                log_key = (data_type, cleaned)
                if log_key not in self._load_message_shown:
                    print(f" {status}数据加载成功！")
                    print(f"  数据类型: {data_type}")
                    print(f"  数据量: {len(df):,} 条记录，{len(df.columns)} 列")
                    self._load_message_shown.add(log_key)
                else:
                    print(f" {status}数据加载成功（重复调用，已省略详细信息）")
                
            return df
        except FileNotFoundError:
            if verbose:
                print(f" 错误：找不到文件 '{filepath}'")
            return None
        except Exception as e:
            if verbose:
                print(f" 加载出错: {e}")
            return None
    
    def _apply_filters(self, 
                      df: pd.DataFrame, 
                      start_date: Optional[str], 
                      end_date: Optional[str],
                      stock_codes: Optional[List[str]],
                      data_type: str) -> pd.DataFrame:
        """应用过滤条件"""
        original_len = len(df)
        
        # 时间过滤
        if start_date or end_date:
            date_col = 'trade_date' if 'trade_date' in df.columns else 'end_date'
            if date_col in df.columns:
                if start_date:
                    df = df[df[date_col] >= start_date]
                if end_date:
                    df = df[df[date_col] <= end_date]
        
        # 股票代码过滤
        if stock_codes:
            # 尝试不同的股票代码列名
            code_cols = ['ts_code', 'code', 'stock_code', 'symbol']
            code_col = None
            for col in code_cols:
                if col in df.columns:
                    code_col = col
                    break
                    
            if code_col:
                df = df[df[code_col].isin(stock_codes)]
            else:
                warnings.warn(f"在{data_type}数据中找不到股票代码列，跳过股票过滤")
        
        filtered_len = len(df)
        if original_len != filtered_len:
            print(f"  过滤后: {filtered_len:,} 条记录 (过滤了 {original_len - filtered_len:,} 条)")
            
        return df
    
    def get_stock_list(self, 
                      trade_date: Optional[str] = None,
                      exclude_st: bool = True) -> List[str]:
        """
        获取股票列表
        
        Args:
            trade_date: 指定日期的股票列表，None表示所有日期
            exclude_st: 是否排除ST股票
            
        Returns:
            List[str]: 股票代码列表
        """
        try:
            # 从stock_basic.parquet加载，效率更高
            basic_data_path = self.raw_data_path / 'stock_basic.parquet'
            if not basic_data_path.exists():
                print("警告: stock_basic.parquet 不存在，回退到加载日线数据")
                df = self.load_data('daily', cleaned=True, verbose=False)
            else:
                df = pd.read_parquet(basic_data_path)
                if trade_date:
                    # 如果需要按日期筛选，仍需加载日线数据
                    daily_df = self.load_data('daily', cleaned=True, verbose=False, start_date=trade_date, end_date=trade_date)
                    if daily_df is not None and not daily_df.empty:
                        valid_codes = daily_df['ts_code'].unique()
                        df = df[df['ts_code'].isin(valid_codes)]
                    else:
                        return []

            if df is None:
                return []
                
            # 排除ST股票
            if exclude_st and 'name' in df.columns:
                df = df[~df['name'].str.contains('ST', na=False)]
            
            # 获取股票代码
            code_cols = ['ts_code', 'code', 'stock_code', 'symbol']
            for col in code_cols:
                if col in df.columns:
                    return sorted(df[col].unique().tolist())
                    
            return []
        except Exception as e:
            print(f"获取股票列表时出错: {e}")
            return []
    
    def get_trading_dates(self, 
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> List[str]:
        """
        获取交易日列表
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            List[str]: 交易日列表
        """
        try:
            df = self.load_data('daily', cleaned=True, verbose=False)
            if df is None:
                return []
                
            dates = pd.to_datetime(df['trade_date']).dt.date.unique()
            dates = sorted([d.strftime('%Y-%m-%d') for d in dates])
            
            # 过滤日期范围
            if start_date:
                dates = [d for d in dates if d >= start_date]
            if end_date:
                dates = [d for d in dates if d <= end_date]
                
            return dates
        except Exception as e:
            print(f"获取交易日列表时出错: {e}")
            return []
    
    def get_data_summary(self) -> Dict[str, Dict]:
        """获取数据概览"""
        summary = {}
        
        for data_type in self.file_mapping.keys():
            try:
                df = self.load_data(data_type, cleaned=True, verbose=False)
                if df is not None:
                    info = {
                        'rows': len(df),
                        'columns': len(df.columns),
                        'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
                    }
                    
                    # 时间范围
                    date_cols = ['trade_date', 'end_date', 'ann_date']
                    for col in date_cols:
                        if col in df.columns:
                            info['date_range'] = [
                                df[col].min().strftime('%Y-%m-%d'),
                                df[col].max().strftime('%Y-%m-%d')
                            ]
                            break
                    
                    summary[data_type] = info
            except Exception as e:
                summary[data_type] = {'error': str(e)}
                
        return summary
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        print("缓存已清空")

# ===== 全局数据管理器实例 =====
_global_data_manager = None

def get_data_manager() -> DataManager:
    """获取全局数据管理器实例"""
    global _global_data_manager
    if _global_data_manager is None:
        _global_data_manager = DataManager()
    return _global_data_manager

# ===== 便捷函数 =====
def load_stock_data(start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   stock_codes: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """
    快速加载股票日线数据
    
    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)  
        stock_codes: 股票代码列表
        
    Returns:
        pd.DataFrame: 股票日线数据
    """
    dm = get_data_manager()
    return dm.load_data('daily', start_date=start_date, end_date=end_date, 
                       stock_codes=stock_codes)

def load_index_data(start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    快速加载指数数据
    
    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        
    Returns:
        pd.DataFrame: 指数数据
    """
    dm = get_data_manager()
    return dm.load_data('index', start_date=start_date, end_date=end_date)

def load_daily_basic_data(start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         stock_codes: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """
    快速加载每日指标数据
    
    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        stock_codes: 股票代码列表
        
    Returns:
        pd.DataFrame: 每日指标数据（包含换手率、市盈率、市净率、市值等）
    """
    dm = get_data_manager()
    return dm.load_data('daily_basic', start_date=start_date, end_date=end_date,
                       stock_codes=stock_codes)

def load_index_weight_data(start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          index_codes: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """
    快速加载指数成分权重数据（月度数据）
    
    Args:
        start_date: 开始日期 (YYYY-MM-DD)，建议输入当月第一天
        end_date: 结束日期 (YYYY-MM-DD)，建议输入当月最后一天
        index_codes: 指数代码列表，如 ['000300.SH', '000905.SH']
        
    Returns:
        pd.DataFrame: 指数成分权重数据
            - index_code: 指数代码
            - con_code: 成分股代码
            - trade_date: 交易日期
            - weight: 权重
    
    Examples:
        >>> # 获取2024年全年的沪深300权重数据
        >>> weights = load_index_weight_data(
        ...     start_date='2024-01-01',
        ...     end_date='2024-12-31',
        ...     index_codes=['000300.SH']
        ... )
    """
    dm = get_data_manager()
    df = dm.load_data('index_weight', start_date=start_date, end_date=end_date,
                     verbose=True)
    
    # 如果指定了指数代码，进行过滤
    if df is not None and index_codes is not None:
        if 'index_code' in df.columns:
            df = df[df['index_code'].isin(index_codes)]
        else:
            warnings.warn("index_weight数据中没有'index_code'列，无法按指数代码过滤")
    
    return df

def load_financial_data(data_type: str,
                       end_date: Optional[str] = None,
                       stock_codes: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """
    快速加载财务数据
    
    Args:
        data_type: 财务数据类型 ('balancesheet', 'income', 'cashflow')
        end_date: 截止日期 (YYYY-MM-DD)
        stock_codes: 股票代码列表
        
    Returns:
        pd.DataFrame: 财务数据
    """
    if data_type not in ['balancesheet', 'income', 'cashflow']:
        raise ValueError("财务数据类型必须是: 'balancesheet', 'income', 'cashflow'")
        
    dm = get_data_manager()
    return dm.load_data(data_type, end_date=end_date, stock_codes=stock_codes)

# ===== 数据质量检查函数 =====
def validate_data_quality(df: pd.DataFrame, data_type: str) -> Dict[str, Union[int, str, List, Dict]]:
    """
    数据质量检查
    
    Args:
        df: 要检查的数据
        data_type: 数据类型
        
    Returns:
        Dict: 检查结果
    """
    results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': {},
        'duplicate_rows': 0,
        'data_types': {},
        'date_range': None,
        'issues': []
    }
    
    # 检查缺失值
    missing = df.isnull().sum()
    results['missing_values'] = {col: int(count) for col, count in missing.items() if count > 0}
    
    # 检查重复行
    results['duplicate_rows'] = df.duplicated().sum()
    
    # 检查数据类型
    results['data_types'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # 检查日期范围
    date_cols = ['trade_date', 'end_date', 'ann_date']
    for col in date_cols:
        if col in df.columns:
            try:
                dates = pd.to_datetime(df[col])
                results['date_range'] = [dates.min().strftime('%Y-%m-%d'), 
                                       dates.max().strftime('%Y-%m-%d')]
                break
            except:
                pass
    
    # 特定数据类型的检查
    if data_type == 'daily':
        # 检查价格数据的合理性
        price_cols = ['open', 'high', 'low', 'close', 'pre_close']
        for col in price_cols:
            if col in df.columns:
                if (df[col] <= 0).any():
                    results['issues'].append(f"{col}列存在非正值")
                    
        # 检查价格关系
        if all(col in df.columns for col in ['high', 'low', 'close']):
            if (df['high'] < df['low']).any():
                results['issues'].append("存在最高价小于最低价的异常数据")
    
    elif data_type in ['balancesheet', 'income', 'cashflow']:
        # 检查财务数据的报告期
        if 'end_date' in df.columns:
            end_dates = pd.to_datetime(df['end_date']).dt.strftime('%m-%d').value_counts()
            if not any(date in ['03-31', '06-30', '09-30', '12-31'] for date in end_dates.index):
                results['issues'].append("财务数据报告期不符合季报/年报标准")
    
    return results

def print_data_summary(summary: Dict[str, Dict]):
    """打印数据概览"""
    print("=" * 60)
    print("数据概览")
    print("=" * 60)
    
    for data_type, info in summary.items():
        if 'error' in info:
            print(f"{data_type}: {info['error']}")
        else:
            print(f"{data_type}:")
            print(f"   数据量: {info['rows']:,} 行 × {info['columns']} 列")
            print(f"   内存占用: {info['memory_mb']:.1f} MB")
            if 'date_range' in info:
                print(f"   时间范围: {info['date_range'][0]} ~ {info['date_range'][1]}")
        print()