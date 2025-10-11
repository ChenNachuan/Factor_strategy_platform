# data_manager/data_cleaner/data_loader.py
import pandas as pd
from pathlib import Path

def load_raw_data():
    """
    加载原始的日线行情数据和股票基本信息数据，并进行合并。
    """
    print("开始加载原始数据...")
    # 使用相对路径定位 raw_data 文件夹
    # Path(__file__).resolve().parent -> 当前文件所在目录 (data_cleaner)
    # .parent -> 上一级目录 (data_manager)
    # / 'raw_data' -> 拼接目标目录
    raw_data_path = Path(__file__).resolve().parent.parent / 'raw_data'
    daily_data_file = raw_data_path / 'a_stock_daily_data.parquet'
    basic_data_file = raw_data_path / 'stock_basic.parquet' 

    # 加载数据
    try:
        daily_df = pd.read_parquet(daily_data_file)
        if not basic_data_file.exists():
            print(f"错误：关键文件 'stock_basic.parquet' 未在以下目录中找到: {raw_data_path}")
            print("该文件包含上市日期和名称，对于清洗次新股和ST股是必需的。")
            return None
        basic_df = pd.read_parquet(basic_data_file)
        print(f"成功加载日线数据 {len(daily_df)} 条，股票基本信息 {len(basic_df)} 条。")
    except FileNotFoundError as e:
        print(f"错误：数据文件未找到，请仔细核对你的路径。 {e}")
        return None

    # 数据合并
    merged_df = pd.merge(daily_df, basic_df[['ts_code', 'name', 'list_date']], on='ts_code', how='left')
    
    # 将日期列转换为datetime对象，便于后续计算
    merged_df['trade_date'] = pd.to_datetime(merged_df['trade_date'], format='%Y%m%d')
    merged_df['list_date'] = pd.to_datetime(merged_df['list_date'], format='%Y%m%d')
    
    print("数据加载与合并完成。")
    return merged_df