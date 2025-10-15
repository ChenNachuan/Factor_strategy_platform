import pandas as pd
from pathlib import Path
import sys

# 添加项目根路径以便导入 config
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import RAW_DATA_PATH

def load_raw_data():
    """
    加载原始的日线行情数据和股票基本信息数据，并进行合并
    确保合并后的数据包含ts_code、name、list_date等关键列
    """
    print("开始加载原始数据...")
    
    # 使用相对路径定位 raw_data 文件夹
    daily_data_file = RAW_DATA_PATH / 'a_stock_daily_data.parquet'
    basic_data_file = RAW_DATA_PATH / 'stock_basic.parquet' 

    # 加载日线数据
    try:
        daily_df = pd.read_parquet(daily_data_file)
        print(f"成功加载日线数据: {len(daily_df):,} 条记录")
        print(f"日线数据列: {list(daily_df.columns)}")
        
        # 检查日线数据是否包含ts_code
        if 'ts_code' not in daily_df.columns:
            print("日线数据缺少ts_code列")
            return None
            
    except FileNotFoundError as e:
        print(f"日线数据文件未找到: {e}")
        return None

    # 加载股票基本信息
    try:
        if not basic_data_file.exists():
            print(f"关键文件 'stock_basic.parquet' 未在以下目录中找到: {RAW_DATA_PATH}")
            print("该文件包含上市日期和名称，对于清洗次新股和ST股是必需的。")
            return None
            
        basic_df = pd.read_parquet(basic_data_file)
        print(f"成功加载股票基本信息: {len(basic_df):,} 条记录")
        print(f"基本信息列: {list(basic_df.columns)}")
        
        # 检查必需的列
        required_basic_cols = ['ts_code', 'name', 'list_date']
        missing_cols = [col for col in required_basic_cols if col not in basic_df.columns]
        if missing_cols:
            print(f"股票基本信息缺少必需列: {missing_cols}")
            return None
            
    except FileNotFoundError as e:
        print(f"股票基本信息文件未找到: {e}")
        return None

    # 数据合并 - 确保保留ts_code列
    print("开始合并数据...")
    before_merge = len(daily_df)
    
    # 使用左连接，保留所有日线数据
    merged_df = pd.merge(
        daily_df, 
        basic_df[['ts_code', 'name', 'list_date']], 
        on='ts_code', 
        how='left'
    )
    
    after_merge = len(merged_df)
    print(f"数据合并完成: {after_merge:,} 条记录")
    
    if after_merge != before_merge:
        print(f"合并后数据量变化: {before_merge:,} -> {after_merge:,}")
    
    # 检查合并后缺失name或list_date的记录
    missing_name = merged_df['name'].isnull().sum()
    missing_list_date = merged_df['list_date'].isnull().sum()
    
    if missing_name > 0:
        print(f"{missing_name:,} 条记录缺少股票名称")
    if missing_list_date > 0:
        print(f"{missing_list_date:,} 条记录缺少上市日期")
    
    # 将日期列转换为datetime对象，便于后续计算
    merged_df['trade_date'] = pd.to_datetime(merged_df['trade_date'], format='%Y%m%d')
    merged_df['list_date'] = pd.to_datetime(merged_df['list_date'], format='%Y%m%d')
    
    print(f"日期格式转换完成")
    print(f"最终合并数据列: {list(merged_df.columns)}")
    
    return merged_df