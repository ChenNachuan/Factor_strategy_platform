import pandas as pd
from pathlib import Path
import sys

# 添加项目根路径以便导入 config
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from cleaning_steps import handle_outliers, handle_missing_values
from config import RAW_DATA_PATH, CLEAN_DATA_PATH

def clean_index_pipeline():
    """
    执行指数数据清洗工作流，复用已有的清洗步骤。
    """
    print("--- 开始指数数据清洗流程 ---")

    # --- 1. 加载原始指数数据 (使用相对路径) ---
    index_data_file = RAW_DATA_PATH / 'a_index_daily_data.parquet'

    try:
        df = pd.read_parquet(index_data_file)
        print(f"✅ 成功加载原始指数数据 {len(df)} 条。")
    except FileNotFoundError:
        print(f"❌ 错误: 找不到指数数据文件 '{index_data_file}'！")
        return

    # --- 2. 数据清洗步骤 ---

    # a. 格式化日期列
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    print("   - 日期格式已转换。")

    # b. 调用通用的离群值处理函数
    df = handle_outliers(df)
    
    # c. 调用通用的缺失值处理函数
    df = handle_missing_values(df)

    # --- 3. 存储清洗后的数据 ---
    save_file = CLEAN_DATA_PATH / 'a_index_daily_data_clean.parquet'
    
    df.to_parquet(save_file, index=False)
    
    print(f"\n✅ 指数数据清洗完成！清洗后的数据已保存至: {save_file}")
    print(f"   最终剩余数据 {len(df)} 条。")

if __name__ == '__main__':
    clean_index_pipeline()