# data_manager/data_cleaner/cleaning_steps.py
import pandas as pd

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """处理收益率离群值"""
    # ... (此函数不变) ...
    print("开始处理离群值...")
    df['pct_chg'] = df['pct_chg'].clip(-10, 10)
    print("离群值处理完成。")
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """处理缺失值"""
    # ... (此函数不变) ...
    print("开始处理缺失值...")
    df = df.sort_values(by=['ts_code', 'trade_date'])
    df = df.groupby('ts_code').ffill()
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount'])
    print("缺失值处理完成。")
    return df

def filter_blacklist(df: pd.DataFrame) -> pd.DataFrame:
    """过滤ST股票和上市不足一年的次新股"""
    print("开始过滤'黑名单'股票...")
    
    # 1. 过滤ST股票，并显式创建副本以避免警告
    df_filtered = df[~df['name'].str.contains('ST')].copy()
    
    # 2. 过滤上市不足一年的次新股
    df_filtered['days_on_market'] = (df_filtered['trade_date'] - df_filtered['list_date']).dt.days
    
    df_final = df_filtered[df_filtered['days_on_market'] > 365]
    
    # drop操作默认会返回一个新副本，所以这里不需要.copy()
    df_final = df_final.drop(columns=['days_on_market'])
    
    print("黑名单股票过滤完成。")
    return df_final