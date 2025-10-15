import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目根路径以便导入 config
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import RAW_DATA_PATH, CLEAN_DATA_PATH

def handle_financial_outliers_mad(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """
    使用中位数绝对偏差法(MAD)处理财务数据的离群值。
    依据《因子投资：方法与实践》3.1.4节的稳健性处理方法。
    """
    print("   - 正在使用MAD方法处理离群值...")
    df_copy = df.copy()
    for col in numeric_cols:
        if df_copy[col].isnull().all() or df_copy[col].nunique() < 2:
            continue
            
        median = df_copy[col].median()
        # 计算MAD时忽略NaN
        mad = ((df_copy[col] - median).abs()).median()
        
        # 如果mad为0，说明大部分数据相同，不进行处理
        if mad == 0:
            continue
            
        # 1.4826 是为了让MAD在正态分布下近似等于一个标准差
        upper_bound = median + 3 * 1.4826 * mad
        lower_bound = median - 3 * 1.4826 * mad
        
        # 替换离群值
        df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)
    return df_copy

def clean_financial_data_pipeline():
    """
    执行完整的财务数据清洗工作流。
    """
    print("--- 开始财务数据清洗流程 ---")

    # 定义原始数据路径和文件名 (使用相对路径)
    financial_files = {
        'balancesheet': RAW_DATA_PATH / 'a_stock_balancesheet_data.parquet',
        'cashflow': RAW_DATA_PATH / 'a_stock_cashflow_data.parquet',
        'income': RAW_DATA_PATH / 'a_stock_income_data.parquet'
    }

    # 循环处理每个财务报表文件
    for name, file_path in financial_files.items():
        print(f"\n--- 正在处理 {name} ({file_path.name}) ---")
        
        # 1. 加载数据
        try:
            df = pd.read_parquet(file_path)
            print(f"成功加载原始 {name} 数据 {len(df)} 条。")
        except FileNotFoundError:
            print(f"错误: 找不到文件 '{file_path}'！")
            continue

        # 2. 识别需要清洗的数值列
        exclude_cols = ['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type', 'update_flag']
        numeric_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
        print(f"   - 识别出 {len(numeric_cols)} 个数值列进行清洗。")

        # 3. 处理缺失值 (用0填充)
        df[numeric_cols] = df[numeric_cols].fillna(0)
        print("   - 缺失值已用0填充。")

        # 4. 处理离群值 (使用MAD方法)
        df = handle_financial_outliers_mad(df, numeric_cols)
        
        # 5. 存储清洗后的数据
        save_file = CLEAN_DATA_PATH / f"a_stock_{name}_data_clean.parquet"
        df.to_parquet(save_file, index=False)
        print(f"清洗后的 {name} 数据已保存至: {save_file}")

    print("\n--- 所有财务数据清洗流程全部完成！ ---")

if __name__ == '__main__':
    clean_financial_data_pipeline()