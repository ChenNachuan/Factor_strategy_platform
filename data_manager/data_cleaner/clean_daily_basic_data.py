from pathlib import Path
import sys
import pandas as pd

# 添加项目根路径以便导入 config
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import RAW_DATA_PATH, CLEAN_DATA_PATH


def load_raw_daily_basic():
    """
    加载原始的 daily_basic 数据
    """
    print("正在加载原始 daily_basic 数据...")
    raw_file = RAW_DATA_PATH / 'a_stock_daily_basic_data.parquet'
    
    if not raw_file.exists():
        print(f"❌ 文件不存在: {raw_file}")
        return None
    
    df = pd.read_parquet(raw_file)
    print(f"✅ 加载成功: {len(df):,} 条记录")
    return df


def handle_daily_basic_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理 daily_basic 数据中的离群值
    - turnover_rate: 换手率范围 [0, 100]
    - pe: 市盈率范围 [-1000, 1000]
    - pb: 市净率范围 [0, 100]
    - ps: 市销率范围 [0, 100]
    - dv_ratio: 股息率范围 [0, 50]
    """
    print("开始处理离群值...")
    
    # 换手率压缩到 [0, 100]
    if 'turnover_rate' in df.columns:
        before_outliers = ((df['turnover_rate'] < 0) | (df['turnover_rate'] > 100)).sum()
        df['turnover_rate'] = df['turnover_rate'].clip(0, 100)
        print(f"   turnover_rate: 压缩了 {before_outliers:,} 个离群值到 [0, 100]")
    
    # 市盈率压缩到 [-1000, 1000] (负值可能是亏损)
    if 'pe' in df.columns:
        before_outliers = ((df['pe'] < -1000) | (df['pe'] > 1000)).sum()
        df['pe'] = df['pe'].clip(-1000, 1000)
        print(f"   pe: 压缩了 {before_outliers:,} 个离群值到 [-1000, 1000]")
    
    # 市盈率TTM压缩
    if 'pe_ttm' in df.columns:
        before_outliers = ((df['pe_ttm'] < -1000) | (df['pe_ttm'] > 1000)).sum()
        df['pe_ttm'] = df['pe_ttm'].clip(-1000, 1000)
        print(f"   pe_ttm: 压缩了 {before_outliers:,} 个离群值到 [-1000, 1000]")
    
    # 市净率压缩到 [0, 100]
    if 'pb' in df.columns:
        before_outliers = ((df['pb'] < 0) | (df['pb'] > 100)).sum()
        df['pb'] = df['pb'].clip(0, 100)
        print(f"   pb: 压缩了 {before_outliers:,} 个离群值到 [0, 100]")
    
    # 市销率压缩到 [0, 100]
    if 'ps' in df.columns:
        before_outliers = ((df['ps'] < 0) | (df['ps'] > 100)).sum()
        df['ps'] = df['ps'].clip(0, 100)
        print(f"   ps: 压缩了 {before_outliers:,} 个离群值到 [0, 100]")
    
    # 市销率TTM压缩
    if 'ps_ttm' in df.columns:
        before_outliers = ((df['ps_ttm'] < 0) | (df['ps_ttm'] > 100)).sum()
        df['ps_ttm'] = df['ps_ttm'].clip(0, 100)
        print(f"   ps_ttm: 压缩了 {before_outliers:,} 个离群值到 [0, 100]")
    
    # 股息率压缩到 [0, 50]
    if 'dv_ratio' in df.columns:
        before_outliers = ((df['dv_ratio'] < 0) | (df['dv_ratio'] > 50)).sum()
        df['dv_ratio'] = df['dv_ratio'].clip(0, 50)
        print(f"   dv_ratio: 压缩了 {before_outliers:,} 个离群值到 [0, 50]")
    
    # 股息率TTM压缩
    if 'dv_ttm' in df.columns:
        before_outliers = ((df['dv_ttm'] < 0) | (df['dv_ttm'] > 50)).sum()
        df['dv_ttm'] = df['dv_ttm'].clip(0, 50)
        print(f"   dv_ttm: 压缩了 {before_outliers:,} 个离群值到 [0, 50]")
    
    print("离群值处理完成。")
    return df


def handle_daily_basic_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理 daily_basic 数据中的缺失值
    - 对于估值指标(pe, pb等): 使用向前填充
    - 对于市值数据: 删除缺失行
    """
    print("开始处理缺失值...")
    print(f"   处理前数据量: {len(df):,} 条")
    
    # 按股票代码和日期排序
    df = df.sort_values(by=['ts_code', 'trade_date'])
    
    # 估值指标可以向前填充
    valuation_cols = ['turnover_rate', 'turnover_rate_f', 'volume_ratio', 
                      'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 
                      'dv_ratio', 'dv_ttm']
    existing_valuation_cols = [col for col in valuation_cols if col in df.columns]
    
    if existing_valuation_cols:
        df[existing_valuation_cols] = df.groupby('ts_code')[existing_valuation_cols].ffill()
        print(f"   向前填充估值指标: {existing_valuation_cols}")
    
    # 市值数据缺失则删除整行(市值是关键指标)
    market_value_cols = ['total_share', 'float_share', 'free_share', 
                        'total_mv', 'circ_mv']
    existing_mv_cols = [col for col in market_value_cols if col in df.columns]
    
    if existing_mv_cols:
        before_drop = len(df)
        df = df.dropna(subset=existing_mv_cols, how='all')  # 如果所有市值列都是NaN才删除
        after_drop = len(df)
        print(f"   删除了 {before_drop - after_drop:,} 条市值数据全部缺失的记录")
    
    # close价格缺失也删除(这是daily_basic的基准价)
    if 'close' in df.columns:
        before_drop = len(df)
        df = df.dropna(subset=['close'])
        after_drop = len(df)
        print(f"   删除了 {before_drop - after_drop:,} 条close价格缺失的记录")
    
    print(f"   处理后数据量: {len(df):,} 条")
    print("缺失值处理完成。")
    return df


def validate_daily_basic_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据有效性验证
    - 确保关键字段非负
    - 确保日期格式正确
    """
    print("开始数据有效性验证...")
    
    # 市值必须为正
    if 'total_mv' in df.columns:
        before_len = len(df)
        df = df[df['total_mv'] > 0]
        after_len = len(df)
        if before_len != after_len:
            print(f"   删除了 {before_len - after_len:,} 条total_mv≤0的无效记录")
    
    if 'circ_mv' in df.columns:
        before_len = len(df)
        df = df[df['circ_mv'] > 0]
        after_len = len(df)
        if before_len != after_len:
            print(f"   删除了 {before_len - after_len:,} 条circ_mv≤0的无效记录")
    
    # 股本必须为正
    if 'total_share' in df.columns:
        before_len = len(df)
        df = df[df['total_share'] > 0]
        after_len = len(df)
        if before_len != after_len:
            print(f"   删除了 {before_len - after_len:,} 条total_share≤0的无效记录")
    
    # 换手率不能为负
    if 'turnover_rate' in df.columns:
        before_len = len(df)
        df = df[df['turnover_rate'] >= 0]
        after_len = len(df)
        if before_len != after_len:
            print(f"   删除了 {before_len - after_len:,} 条turnover_rate<0的无效记录")
    
    print("数据有效性验证完成。")
    return df


def run_pipeline():
    """
    执行完整的 daily_basic 数据清洗工作流
    """
    print("=" * 60)
    print("🚀 开始执行 daily_basic 数据清洗流程")
    print("=" * 60)
    
    # 1. 加载原始数据
    print("\n📂 步骤1: 数据加载")
    df = load_raw_daily_basic()
    
    if df is None:
        print("❌ 数据加载失败，清洗流程中止")
        return
    
    print(f"✅ 数据加载成功: {len(df):,} 条记录")
    print(f"📊 数据列: {list(df.columns)}")
    
    # 确保关键列存在
    required_cols = ['ts_code', 'trade_date', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 缺少关键列: {missing_cols}")
        return
    
    # 2. 依次执行清洗步骤
    print("\n🧹 步骤2: 数据清洗")
    
    # 2a. 处理离群值
    print("\n--- 2a. 处理离群值 ---")
    df = handle_daily_basic_outliers(df)
    
    # 2b. 处理缺失值
    print("\n--- 2b. 处理缺失值 ---")
    df = handle_daily_basic_missing_values(df)
    
    # 2c. 数据有效性验证
    print("\n--- 2c. 数据有效性验证 ---")
    df = validate_daily_basic_data(df)
    
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
    
    # 检查市值数据覆盖率
    if 'total_mv' in df.columns:
        mv_coverage = (df['total_mv'].notna().sum() / len(df)) * 100
        print(f"✅ total_mv数据覆盖率: {mv_coverage:.2f}%")
    
    if 'turnover_rate' in df.columns:
        tr_coverage = (df['turnover_rate'].notna().sum() / len(df)) * 100
        print(f"✅ turnover_rate数据覆盖率: {tr_coverage:.2f}%")
    
    # 4. 存储清洗后的数据
    print("\n💾 步骤4: 数据存储")
    save_file = CLEAN_DATA_PATH / 'a_stock_daily_basic_data_clean.parquet'
    
    df.to_parquet(save_file, index=False)
    print(f"✅ 数据已保存至: {save_file}")
    print(f"📦 文件大小: {save_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    print("\n" + "=" * 60)
    print("🎉 daily_basic 数据清洗流程完成！")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
