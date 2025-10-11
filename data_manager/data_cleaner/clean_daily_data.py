from pathlib import Path
from data_loader import load_raw_data
from cleaning_steps import handle_outliers, handle_missing_values, filter_blacklist

def run_pipeline():
    """
    执行完整的数据清洗工作流。
    """
    # 1. 加载并合并原始数据
    df = load_raw_data()
    
    if df is None:
        return

    # 2. 依次执行清洗步骤
    df = handle_outliers(df)
    df = handle_missing_values(df)
    df = filter_blacklist(df)
    
    # 3. 存储清洗后的数据
    print("\n开始存储清洗后的数据...")
    # 使用相对路径定位 clean_data 文件夹
    clean_data_path = Path(__file__).resolve().parent.parent / 'clean_data'
    clean_data_path.mkdir(parents=True, exist_ok=True)
    save_file = clean_data_path / 'a_stock_daily_data_clean.parquet'
    
    df.to_parquet(save_file, index=False)
    
    print(f"数据清洗流程全部完成！清洗后的数据已保存至: {save_file}")
    print(f"最终剩余数据 {len(df)} 条。")

if __name__ == '__main__':
    run_pipeline()