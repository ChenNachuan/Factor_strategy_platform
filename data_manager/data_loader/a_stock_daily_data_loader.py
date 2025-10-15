import tushare as ts
import pandas as pd
import time
from pathlib import Path
from config import get_tushare_token, RAW_DATA_PATH

# 初始化 Tushare API
ts.set_token(get_tushare_token())
pro = ts.pro_api()

print("Tushare API 初始化成功。")

# 获取股票列表
print("正在获取股票列表...")
stock_list = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name')
print(f"成功获取到 {len(stock_list)} 只股票。")

# 循环获取所有股票的日线行情数据
start_date = '20100101'
end_date = '20250930'

# 创建一个空的 DataFrame 用于存储所有股票的数据
all_stock_data = pd.DataFrame()

print(f"开始获取从 {start_date} 到 {end_date} 的日线数据...")

# 循环遍历股票列表中的每一只股票
for _, row in stock_list.iterrows():
    ts_code = row['ts_code']
    try:
        # 调用 daily 接口获取单只股票的日线数据
        df_single_stock = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        
        # 将新获取的数据拼接到总的 DataFrame 中
        all_stock_data = pd.concat([all_stock_data, df_single_stock])
        
        # 打印进度提示
        print(f"成功获取 {ts_code} ({row['name']}) 的数据。")

        # 为了避免触发接口频率限制，稍作等待
        time.sleep(0.2) 
        
    except Exception as e:
        print(f"获取 {ts_code} 数据时出错: {e}")

print("所有股票获取完毕！")

# 将数据存储为 Parquet 格式

# 首先，做一个安全检查，确保我们的 all_stock_data 不是空的。
if not all_stock_data.empty:
    print("\n开始将数据存储到 Parquet 文件...")
    try:
        # 构建完整的文件保存路径
        file_path = RAW_DATA_PATH / 'a_stock_daily_data.parquet'

        all_stock_data.to_parquet(file_path, engine='pyarrow', index=False)
        
        # 输出信息路径
        print(f"数据已成功保存到: {file_path}")
        
    except Exception as e:
        print(f"数据保存失败，错误信息: {e}")
else:
    print("\n未能获取到任何数据，请检查网络或 Tushare Token 设置。")