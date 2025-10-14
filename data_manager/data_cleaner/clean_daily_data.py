from pathlib import Path
from data_loader import load_raw_data
from cleaning_steps import handle_outliers, handle_missing_values, filter_blacklist

def run_pipeline():
    """
    执行完整的数据清洗工作流
    确保清洗后的数据包含ts_code列，并正确剔除ST股票和次新股
    """
    print("=" * 60)
    print("🚀 开始执行股票日线数据清洗流程")
    print("=" * 60)
    
    # 1. 加载并合并原始数据
    print("\n📂 步骤1: 数据加载")
    df = load_raw_data()
    
    if df is None:
        print("❌ 数据加载失败，清洗流程中止")
        return
    
    print(f"✅ 数据加载成功: {len(df):,} 条记录")
    print(f"📊 数据列: {list(df.columns)}")
    
    # 确保关键列存在
    required_cols = ['ts_code', 'name', 'trade_date', 'list_date', 'pct_chg']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 缺少关键列: {missing_cols}")
        return

    # 2. 依次执行清洗步骤
    print("\n🧹 步骤2: 数据清洗")
    
    # 2a. 处理离群值
    print("\n--- 2a. 处理离群值 ---")
    df = handle_outliers(df)
    
    # 2b. 处理缺失值
    print("\n--- 2b. 处理缺失值 ---") 
    df = handle_missing_values(df)
    
    # 2c. 过滤黑名单股票
    print("\n--- 2c. 过滤黑名单股票 ---")
    df = filter_blacklist(df)
    
    # 3. 数据质量检查
    print("\n🔍 步骤3: 数据质量检查")
    print(f"✅ 最终数据量: {len(df):,} 条记录")
    print(f"📊 最终列数: {len(df.columns)} 列")
    print(f"🏢 股票数量: {df['ts_code'].nunique():,} 只")
    print(f"📅 时间跨度: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
    
    # 验证关键列
    if 'ts_code' not in df.columns:
        print("❌ 警告: ts_code列丢失！")
    else:
        print("✅ ts_code列保留完整")
    
    # 验证ST股票剔除
    if 'name' in df.columns:
        st_count = df['name'].str.contains('ST', na=False).sum()
        if st_count == 0:
            print("✅ ST股票已完全剔除")
        else:
            print(f"⚠️ 仍有 {st_count} 只ST股票")
    
    # 4. 存储清洗后的数据
    print("\n💾 步骤4: 数据存储")
    clean_data_path = Path(__file__).resolve().parent.parent / 'clean_data'
    clean_data_path.mkdir(parents=True, exist_ok=True)
    save_file = clean_data_path / 'a_stock_daily_data_clean.parquet'
    
    df.to_parquet(save_file, index=False)
    
    print(f"✅ 清洗后数据已保存至: {save_file}")
    print(f"📁 文件大小: {save_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    print("\n" + "=" * 60)
    print("🎉 数据清洗流程全部完成！")
    print("=" * 60)

if __name__ == '__main__':
    run_pipeline()