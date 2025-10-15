import tushare as ts # type: ignore
import pandas as pd
from pathlib import Path
from config import get_tushare_token, RAW_DATA_PATH

def download_stock_basic():
    """
    使用 Tushare API 获取所有A股上市公司的基本信息，
    并将其保存为 Parquet 文件。
    """
    print("开始获取股票基本信息...")
    
    # 初始化 Tushare API
    try:
        ts.set_token(get_tushare_token())
        pro = ts.pro_api()
        print("Tushare API 初始化成功。")
    except Exception as e:
        print(f"Tushare API 初始化失败: {e}")
        return

    # 获取股票列表
    try:
        # list_status='L' 表示只获取上市状态的股票
        stock_list_df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
        if stock_list_df.empty:
            print("警告：获取到的股票列表为空，请检查网络或Token。")
            return
        print(f"成功获取到 {len(stock_list_df)} 只上市股票的基本信息。")
    except Exception as e:
        print(f"获取股票列表时出错: {e}")
        return

    # 存储数据
    # 使用相对路径构建保存路径
    save_path = RAW_DATA_PATH
    
    # 确保目录存在
    save_path.mkdir(parents=True, exist_ok=True)
    
    file_path = save_path / 'stock_basic.parquet'

    try:
        stock_list_df.to_parquet(file_path, index=False)
        print(f"股票基本信息已成功保存至: {file_path}")
    except Exception as e:
        print(f"存储文件时出错: {e}")

if __name__ == '__main__':
    download_stock_basic()