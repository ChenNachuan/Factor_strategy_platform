import tushare as ts
import pandas as pd
import time
from pathlib import Path
from datetime import datetime
from config import get_tushare_token, RAW_DATA_PATH

# 初始化 Tushare API
ts.set_token(get_tushare_token())
pro = ts.pro_api()

print("Tushare API 初始化成功。")

# 获取A股上市公司列表
print("\n正在获取最新的A股上市公司列表...")
try:
    stock_list_df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,list_date')
    print(f"成功获取 {len(stock_list_df)} 家上市公司的基本信息。")
except Exception as e:
    print(f"获取股票列表失败，请检查网络或Token设置: {e}")
    # 如果无法获取股票列表，则直接退出程序
    exit()

# 定义下载参数和数据容器
# 定义需要下载的财务报表及其对应的Tushare接口
financial_statements = {
    'income': {'api': pro.income, 'name': '利润表'},
    'balancesheet': {'api': pro.balancesheet, 'name': '资产负债表'},
    'cashflow': {'api': pro.cashflow, 'name': '现金流量表'}
}

# 创建三个空的DataFrame，用于分别存储所有公司的三种财务报表
all_financial_data = {
    'income': pd.DataFrame(),
    'balancesheet': pd.DataFrame(),
    'cashflow': pd.DataFrame()
}

# 循环获取所有A股公司的财务数据
total_stocks = len(stock_list_df)
print(f"\n准备开始下载 {total_stocks} 只股票的财务数据，这将需要较长时间，请耐心等待...")

# 遍历股票列表中的每一只股票
for index, row in stock_list_df.iterrows():
    ts_code = row['ts_code']
    stock_name = row['name']
    
    # 打印进度
    print(f"--- [{index + 1}/{total_stocks}] 正在处理: {ts_code} ({stock_name}) ---")
    
    # 遍历需要下载的三种财务报表
    for report_type, info in financial_statements.items():
        api_func = info['api']
        report_name = info['name']
        
        try:
            # 调用对应的Tushare接口获取该股票的所有报告期数据
            df_single_stock_report = api_func(ts_code=ts_code)
            
            # 如果成功获取到数据
            if not df_single_stock_report.empty:
                # 将新获取的数据拼接到对应的总DataFrame中
                all_financial_data[report_type] = pd.concat(
                    [all_financial_data[report_type], df_single_stock_report], 
                    ignore_index=True
                )
                print(f"  成功获取 {ts_code} 的 {report_name} 数据，共 {len(df_single_stock_report)} 条记录。")
            else:
                print(f"  未找到 {ts_code} 的 {report_name} 数据。")

            # API频率限制：每次调用后强制等待，Tushare对财务数据接口有更严格的限制
            time.sleep(0.6) # 建议等待0.6秒以上

        except Exception as e:
            print(f"  获取 {ts_code} 的 {report_name} 数据时出错: {e}")
            # 即使某个接口出错，也继续尝试下一个
            continue

print("\n所有股票的财务数据获取循环执行完毕！")


# --- 5. 将数据存储为 Parquet 格式 ---
print("\n开始将数据存储到 Parquet 文件...")
try:
    # 构建目标存储目录路径
    save_dir = RAW_DATA_PATH
    
    # 确保目标目录存在
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 循环存储三种财务报表
    for report_type, df_to_save in all_financial_data.items():
        # 安全检查，确保DataFrame不为空
        if not df_to_save.empty:
            # 构建完整的文件保存路径
            file_path = save_dir / f'a_stock_{report_type}_data.parquet'
            
            # 保存为Parquet文件
            df_to_save.to_parquet(file_path, engine='pyarrow', index=False)
            
            print(f"{financial_statements[report_type]['name']}数据已成功保存到: {file_path}")
        else:
            print(f"未获取到任何{financial_statements[report_type]['name']}数据，跳过保存。")

except Exception as e:
    print(f"数据保存过程中发生错误，错误信息: {e}")

print("\n所有任务完成。")