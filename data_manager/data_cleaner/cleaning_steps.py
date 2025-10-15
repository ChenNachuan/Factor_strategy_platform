import pandas as pd

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理收益率离群值
    """
    print("开始处理离群值...")
    print(f"   处理前收益率范围: [{df['pct_chg'].min():.2f}%, {df['pct_chg'].max():.2f}%]")
    
    # 统计离群值数量
    outliers_count = ((df['pct_chg'] < -10) | (df['pct_chg'] > 10)).sum()
    
    # 压缩到[-10, 10]区间
    df['pct_chg'] = df['pct_chg'].clip(-10, 10)
    
    print(f"   压缩了 {outliers_count:,} 个离群值")
    print(f"   处理后收益率范围: [{df['pct_chg'].min():.2f}%, {df['pct_chg'].max():.2f}%]")
    print("离群值处理完成。")
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理缺失值 - 使用向前填充方法
    """
    print("开始处理缺失值...")
    print(f"   处理前数据量: {len(df):,} 条")
    
    # 按股票代码和日期排序，确保向前填充的正确性
    df = df.sort_values(by=['ts_code', 'trade_date'])
    
    # 按股票分组进行向前填充
    price_cols = ['open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
    df[price_cols] = df.groupby('ts_code')[price_cols].ffill()
    
    # 删除仍然有缺失值的行
    before_drop = len(df)
    df = df.dropna(subset=price_cols)
    after_drop = len(df)
    
    print(f"   向前填充完成，删除了 {before_drop - after_drop:,} 条无法填充的记录")
    print(f"   处理后数据量: {after_drop:,} 条")
    print("缺失值处理完成。")
    return df

def filter_blacklist(df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤ST股票和上市不足一年的次新股
    """
    print("开始过滤'黑名单'股票...")
    print(f"   过滤前数据量: {len(df):,} 条")
    
    # 检查必需的列是否存在
    required_cols = ['name', 'ts_code', 'trade_date', 'list_date']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"   缺少必需的列: {missing_cols}")
        return df
    
    # 1. 过滤ST股票
    before_st_filter = len(df)
    df_filtered = df[~df['name'].str.contains('ST', na=False)].copy()
    after_st_filter = len(df_filtered)
    st_removed = before_st_filter - after_st_filter
    print(f"   剔除ST股票: {st_removed:,} 条记录")
    
    # 2. 过滤上市不足一年的次新股
    df_filtered['days_on_market'] = (df_filtered['trade_date'] - df_filtered['list_date']).dt.days
    
    before_new_stock_filter = len(df_filtered)
    df_final = df_filtered[df_filtered['days_on_market'] > 365].copy()
    after_new_stock_filter = len(df_final)
    new_stocks_removed = before_new_stock_filter - after_new_stock_filter
    print(f"   剔除次新股: {new_stocks_removed:,} 条记录")
    
    # 删除临时列
    df_final = df_final.drop(columns=['days_on_market'])
    
    print(f"   过滤后数据量: {len(df_final):,} 条")
    print(f"   总计剔除: {len(df) - len(df_final):,} 条记录")
    print("黑名单股票过滤完成。")
    return df_final