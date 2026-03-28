# SwingTrade 数据采集与存储系统设计

**版本**: v1.3
**状态**: 已合并
**维护者**: bruce li
**更新日期**: 2026-03-29

---

## 一、设计概述

### 1.1 系统定位

```
SwingTrade/src/data/fetcher/     # 采集模块（代码在SwingTrade）
SwingTrade/config/                # 配置文件
SwingTrade/docs/                  # 设计文档
/Users/bruce/workspace/trade/StockData/  # 数据存储（纯数据）
```

**核心目标**：

| 目标 | 描述 | 优先级 |
|------|------|--------|
| **数据安全** | 不丢数据、不损坏数据、备份可恢复 | P0 |
| **数据质量** | 校验机制保障数据准确性，写入前必须验证 | P0 |
| **高效访问** | 支撑波段交易策略的计算需求 | P1 |
| **可维护性** | 自动化运维、监控告警 | P1 |

### 1.2 核心设计原则

| 原则 | 说明 |
|------|------|
| **幂等写入** | 同一日期数据重复采集不会产生重复 |
| **质量门槛** | 质量不达标的数据不写入 |
| **失败追溯** | 所有失败有日志，支持自动重试 |
| **日报透明** | 所有失败体现在日报中 |
| **测试数据零容忍** | 决不允许虚假数据写入真实存储 |

---

## 二、数据源策略

### 2.1 数据源分工

根据 200 积分和接口限制，制定最优数据源组合：

| 数据源 | 用途 | 调用频率 | 成本 |
|--------|------|----------|------|
| **Tushare Pro (200积分)** | 日线(daily) + 复权因子(adj_factor) | 主力，10万次/天 | 200元/年 |
| **AkShare** | 股票列表(stock_list) | 补充，免费 | 免费 |

### 2.2 Tushare Pro 200积分限制

| 接口 | 限制 | 说明 |
|------|------|------|
| `daily` (日线) | 10万次/天 | 足够全市场采集 |
| `adj_factor` (复权因子) | 共享限制 | 与日线合计 |
| `stock_basic` (股票列表) | 每小时1次 | **不用**，改用 AkShare |

### 2.3 为什么不用 Tushare 股票列表

| 方案 | 优点 | 缺点 |
|------|------|------|
| Tushare stock_basic | 数据规范 | 每小时限1次，无法实时同步 |
| **AkShare stock_list** | 免费、调用频繁 | 需处理北交所代理问题 |

**决策**：使用 AkShare 获取股票列表，Tushare 专注于日线和复权因子。

---

## 三、模块结构

```
SwingTrade/src/data/fetcher/
├── __init__.py
├── fetch_daily.py           # 日线采集主入口
├── backfill.py              # 历史数据回填
├── index_fetcher.py         # 指数数据回填
├── quality_scorer.py        # 质量评分
├── data_merger.py          # 数据合并（复权因子）
├── price_converter.py       # 前复权/后复权转换
├── validators/
│   ├── __init__.py
│   ├── stock_validator.py    # 股票列表验证
│   └── daily_validator.py   # 日线数据验证
├── sources/
│   ├── __init__.py
│   ├── base.py              # 数据源基类
│   ├── tushare_source.py    # Tushare 适配器（主力）
│   ├── akshare_source.py    # AkShare 适配器（补充）
│   └── eastmoney_source.py  # 东方财富适配器（资金流）
├── retry_handler.py         # 重试逻辑
├── report_generator.py      # 日报生成
└── exceptions.py            # 自定义异常
```

---

## 四、数据流

### 4.1 日线采集流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     日线采集主流程（每日 17:30-18:00）             │
└─────────────────────────────────────────────────────────────────┘

注意：Tushare 日线数据在交易日 15:00-17:00 更新，为确保数据完整性，
采集窗口设为 17:30 开始，给数据源留足缓冲时间。

1. [获取股票列表]
   └─ 从 AkShare 获取全市场股票列表（约5000只）
   └─ 验证后存入 SQLite stocks 表

2. [获取采集日期]
   └─ target_date = 上一交易日（自动计算）
   └─ 检查 update_log 中是否有失败日期需要重试

3. [批量拉取 Tushare 日线]
   └─ 每个股票调用 pro.daily(ts_code, start_date, end_date)
   └─ 调用 pro.adj_factor(ts_code, start_date, end_date) 获取复权因子
   └─ 并行发送，不限流（200积分足够）

4. [数据合并]
   └─ 将日线数据与复权因子合并
   └─ 计算后复权价格

5. [质量评分]
   ├─ overall_score >= 80 → 直接写入
   ├─ overall_score >= 60 → 触发 AkShare 验证
   └─ overall_score < 60 → 拒绝，标记失败

6. [幂等写入]
   └─ 按 (code, date) 去重后写入 Parquet

7. [更新索引]
   └─ 更新 SQLite daily_index 和 update_log

8. [生成日报]
   └─ 输出采集结果统计（包含所有失败详情）
```

### 4.2 复权计算

```python
def merge_daily_with_adj_factor(daily_df: pd.DataFrame, adj_df: pd.DataFrame, code: str) -> pd.DataFrame:
    """
    合并日线数据和复权因子

    Args:
        daily_df: 日线数据
        adj_df: 复权因子数据
        code: 股票代码

    Returns:
        合并后的 DataFrame
    """
    if daily_df.empty:
        return daily_df

    if adj_df.empty or len(adj_df) == 0:
        daily_df['adj_factor'] = 1.0
        daily_df['close_adj'] = daily_df['close']
        return daily_df

    # 合并
    df = daily_df.merge(adj_df[['date', 'adj_factor']], on='date', how='left')

    # 前向填充缺失值
    df['adj_factor'] = df['adj_factor'].ffill().fillna(1.0)

    # 计算后复权价
    df['close_adj'] = df['close'] * df['adj_factor']

    return df
```

---

## 五、质量评分体系

### 5.1 评分维度

| 维度 | 权重 | 说明 |
|------|------|------|
| source_consistency | 30% | 跨源一致性（双源 vs 单源） |
| field_completeness | 20% | 字段完整性 |
| range_validity | 30% | 范围合理性 |
| historical_anomaly | 20% | 历史连续性 |

### 5.2 评分标准

**source_consistency**:
- 双源完全一致：100
- 双源差异 < 0.1%：95
- 单源（Tushare 可靠）：95
- 双源差异 >= 1%：50

**field_completeness**:
- 每缺失一个字段 -10 分
- 7个必填字段：date, code, open, high, low, close, volume

**range_validity**:
- 价格范围 0.01 - 10000：100
- OHLC 关系错误：直接归零
- 涨跌幅超 ±20%：归零

**historical_anomaly**:
- 无历史数据：100
- 涨跌停范围内：100
- 超出涨跌停：30

### 5.3 阈值判断

| 分数 | 行为 |
|------|------|
| >= 80 | 直接写入 |
| 60-80 | 触发 AkShare 验证 |
| < 60 | 拒绝并告警 |

---

## 六、数据源适配器

### 6.1 Tushare 适配器（主力）

```python
class TushareSource(DataSource):
    """Tushare 数据源（主力日线数据源）"""

    name = "tushare"
    SAFETY_LIMIT = 45  # 90% of 50 calls/minute

    def __init__(self, token: str = None):
        self.token = token or os.getenv("TUSHARE_TOKEN")
        if not self.token:
            raise ConfigurationError("TUSHARE_TOKEN not set")
        import tushare as ts
        self.pro = ts.pro_api(self.token)
        self._call_times = []
```

### 6.2 AkShare 适配器（补充）

```python
class AkShareSource(DataSource):
    """AkShare 数据源（股票列表补充源）"""

    name = "akshare"

    def fetch_stock_list(self) -> pd.DataFrame:
        """拉取股票列表（沪深两市）"""
        # 从沪市、深市分别获取
```

### 6.3 东方财富适配器（资金流）

```python
class EastMoneySource(DataSource):
    """东方财富数据源（资金流专项数据源）"""

    name = "eastmoney"

    def fetch_individual_fund_flow(self, code: str, market: str) -> pd.DataFrame:
        """获取个股资金流（主力/超大单/大单/中单/小单）"""

    def fetch_industry_fund_flow(self, indicator: str = "今日") -> pd.DataFrame:
        """获取行业资金流排名"""

    def fetch_hsgt_north_flow(self) -> pd.DataFrame:
        """获取沪深港通北向资金"""
```

---

## 七、采集窗口与调度

### 7.1 采集时间窗口

| 时间 | 任务 |
|------|------|
| 17:30 | 定时任务触发，开始采集 |
| 17:30-17:50 | 主力采集（Tushare 日线 + 复权因子） |
| 17:50-18:00 | 扫尾、重试失败、处理异常 |
| 18:00 | 窗口关闭，生成日报 |

**注意**：Tushare 日线数据在交易日 15:00-17:00 更新，17:30 开始确保数据已完整更新。

### 7.2 crontab 配置

```bash
# 每日 17:30 执行（工作日）
30 17 * * 1-5 /Users/bruce/workspace/trade/SwingTrade/scripts/fetch/run_daily_fetch.py
```

---

## 八、重试机制

### 8.1 重试策略

| 失败类型 | 重试次数 | 间隔 | 处理 |
|----------|----------|------|------|
| `network` | 2次 | 0s → 30s | 等待后重试 |
| `source` | 2次 | 0s → 30s | 切换备源 |
| `quality` | 0次 | - | 拒绝，不重试 |

### 8.2 失败追溯

```python
@dataclass
class FetchResult:
    code: str
    date: str
    data: Optional[pd.DataFrame]
    fetch_status: str = "pending"   # pending / success / failed
    write_status: str = "pending"   # pending / success / rejected
    fail_type: Optional[str] = None  # None / network / quality / source
    fail_reason: Optional[str] = None
    quality_score: Optional[float] = None
    quality_dims: dict = field(default_factory=dict)
    attempts: int = 0
```

---

## 九、写入架构

### 9.1 Parquet Schema（日线数据）

```python
{
    "date": "datetime64",      # 交易日期
    "code": "str",             # 股票代码
    "open": "float32",         # 开盘价
    "high": "float32",         # 最高价
    "low": "float32",          # 最低价
    "close": "float32",       # 收盘价
    "volume": "int64",         # 成交量
    "pct_chg": "float32",     # 涨跌幅
    "adj_factor": "float32",  # 复权因子
    "close_adj": "float32"    # 后复权收盘价
}
```

### 9.2 幂等写入

```python
def write_daily(code: str, df: pd.DataFrame, stockdata_root: str) -> bool:
    """幂等写入：基于 (code, date) 去重"""
    daily_dir = os.path.join(stockdata_root, "raw", "daily")
    target_file = os.path.join(daily_dir, f"{code}.parquet")

    if os.path.exists(target_file):
        existing = pd.read_parquet(target_file)
        # 过滤已存在的日期
        new_df = df[~df["date"].isin(existing["date"])]
        if len(new_df) == 0:
            return True  # 已存在，跳过
        df = pd.concat([existing, new_df]).drop_duplicates(["date"])
        df = df.sort_values("date")

    # 写入临时文件 + 原子替换
    temp_file = os.path.join(daily_dir, f"{code}_{datetime.now().strftime('%Y%m%d%H%M%S')}.tmp.parquet")
    df.to_parquet(temp_file, engine="pyarrow", compression="snappy")
    os.replace(temp_file, target_file)

    return True
```

---

## 十、目录结构

```
SwingTrade/
├── src/data/fetcher/
│   ├── fetch_daily.py           # 日线采集
│   ├── quality_scorer.py        # 质量评分
│   ├── data_merger.py           # 数据合并
│   ├── validators/
│   │   ├── stock_validator.py
│   │   └── daily_validator.py
│   ├── sources/
│   │   ├── tushare_source.py
│   │   └── akshare_source.py
│   ├── retry_handler.py
│   └── report_generator.py
├── config/
│   └── data_quality.yaml
├── scripts/fetch/
│   └── run_daily_fetch.py
├── docs/
│   └── SwingTrade_Data_System_Design.md
└── tests/

StockData/                       # 纯数据
├── raw/daily/                   # 日线 Parquet
│   └── {code}.parquet
├── sqlite/
│   └── market.db
├── status/
│   └── daily_report_{date}.json
└── logs/
```

---

## 十一、SQLite 表结构

### 11.1 stocks 表

```sql
CREATE TABLE stocks (
    code TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    market TEXT NOT NULL,          -- 'sh' / 'sz'
    is_active INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
```

### 11.2 daily_index 表

```sql
CREATE TABLE daily_index (
    code TEXT PRIMARY KEY,
    date TEXT NOT NULL,
    file_path TEXT NOT NULL,
    row_count INTEGER,
    start_date TEXT,
    end_date TEXT,
    updated_at TEXT DEFAULT (datetime('now'))
);
```

### 11.3 update_log 表

```sql
CREATE TABLE update_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    data_type TEXT NOT NULL,       -- 'daily'
    code TEXT NOT NULL,
    update_date TEXT NOT NULL,
    fetch_status TEXT NOT NULL,   -- 'success' / 'failed'
    write_status TEXT,            -- 'success' / 'rejected'
    quality_score REAL,
    source TEXT,
    attempts INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(code, update_date, data_type)
);
```

---

## 十二、数据分层策略

### 12.1 分层定义

| 层级 | 定义 | 存储介质 | 保留周期 |
|------|------|---------|---------|
| **Hot** | 当日实时行情 | SQLite + Memory | 当日收盘后清除 |
| **Warm** | 近 1-60 天数据 | Parquet | 滚动保留 60 天 |
| **Cold** | 超过 60 天历史 | Parquet 归档 | 永久保留 |

### 12.2 热数据层 (Hot)

**用途**: 盘中实时行情读取

**存储**:
```python
# SQLite latest_quote 表
# 结构见 11.4 latest_quote 表

# Memory Cache
# 进程内 LRU 缓存，TTL: 直到当日收盘
```

**访问模式**:
```
读取: memory cache → SQLite latest_quote → 数据源
写入: 直接更新 SQLite（单点，无竞争）
```

### 12.3 温数据层 (Warm)

**用途**: 近30-60天策略计算、信号验证

**存储**:
```python
# 目录: warm/daily_summary/{YYYYMMDD}.parquet
# 每只股票一行，包含当日 OHLCV + 复权因子
# 用于快速选股扫描（全市场一文件）
```

**访问模式**:
```
读取: pandas.read_parquet() 直接加载
写入: Parquet 临时文件 → 原子替换
```

### 12.4 冷数据层 (Cold)

**用途**: 长期历史回测

**存储**:
```python
# 目录: raw/daily/{code}.parquet
# 按股票代码组织，每股一个文件
# 包含历史全量日线（后复权）
```

**访问模式**:
```
读取: pd.read_parquet(f"raw/daily/{code}.parquet")
     df[df['date'] >= start_date]
写入: 单写入器，原子替换
```

### 12.5 latest_quote 表

```sql
CREATE TABLE latest_quote (
    code TEXT PRIMARY KEY,
    date TEXT,                 -- YYYY-MM-DD
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    amount REAL,
    pct_chg REAL,
    update_time TEXT           -- YYYY-MM-DD HH:MM:SS
);
```

---

## 十三、维护任务清单

### 13.1 日常维护

| 任务 | 频率 | 执行时间 | 脚本 |
|------|------|---------|------|
| 日线数据采集 | 每日 | 17:30 | run_daily_fetch.py |
| 温数据汇总生成 | 每日 | 16:00-16:30 | warm_summary.py |
| 冷数据归档 | 每日 | 16:30-17:00 | archive.py |
| 健康检查 | 每日 | 09:00, 16:30 | health_check.py |

### 13.2 定期维护

| 任务 | 频率 | 执行时间 | 脚本 |
|------|------|---------|------|
| SQLite VACUUM | 每周 | 周五 17:00 | vacuum.py |
| SQLite REINDEX | 每月 | 1日 09:00 | reindex.py |
| 增量备份 | 每日 | 17:00-17:30 | backup.py |

### 13.3 归档策略

```python
def archive_cold_data(cutoff_days: int = 60):
    """将超过60天的数据从 warm 移动到 cold"""

    cutoff_date = datetime.now() - timedelta(days=cutoff_days)

    for code_file in Path('raw/daily').glob('*.parquet'):
        df = pd.read_parquet(code_file)

        # 分离冷数据
        cold_df = df[df['date'] < cutoff_date.strftime('%Y-%m-%d')]
        warm_df = df[df['date'] >= cutoff_date.strftime('%Y-%m-%d')]

        if len(cold_df) > 0:
            # 写入归档目录
            archive_path = f"cold/daily/{code_file.name}"
            cold_df.to_parquet(archive_path)

            # 重写温数据文件
            warm_df.to_parquet(code_file)
```

---

## 十四、备份策略

### 14.1 备份内容

| 内容 | 重要性 | 备份方式 |
|------|--------|---------|
| raw/daily/ | ✓ 重要 | 全量备份 |
| processed/adj/ | ✓ 重要 | 全量备份 |
| sqlite/*.db | ✓ 重要 | 全量备份 |
| config/ | ✓ 重要 | 增量备份 |
| scripts/ | ✓ 重要 | 增量备份 |

### 14.2 不备份内容

| 内容 | 原因 |
|------|------|
| cache/ | 可再生 |
| logs/ | 可再生 |
| warm/daily_summary/ | 可再生 |
| status/ | 可再生 |

### 14.3 备份保留

```
- 每日增量备份: 最近 7 天
- 每周全量备份: 最近 4 周
- 每月全量备份: 最近 12 个月
```

---

## 十五、健康检查

### 15.1 检查项

| 检查项 | 触发条件 | 级别 |
|--------|---------|------|
| 数据新鲜度 | 日线数据未更新超过 16:30 | ERROR |
| 采集完整性 | 成功率 < 99.5% | WARNING |
| 存储空间 | 使用率 > 80% | WARNING |
| SQLite WAL | WAL 文件 > 100MB | WARNING |

### 15.2 检查实现

```python
class HealthChecker:
    def check_freshness(self):
        """检查数据新鲜度"""
        last_update = load_status()['last_update']
        if is_overdue(last_update, threshold='16:30'):
            return Alert("ERROR", "日线数据未更新")

    def check_completeness(self):
        """检查完整性"""
        actual = load_status()['success_count']
        if actual < 4000 * 0.995:
            return Alert("WARNING", f"采集不完整: {actual}/4000")

    def check_storage(self):
        """检查存储空间"""
        usage = shutil.disk_usage('/Users/bruce/workspace/trade')
        if usage.percent > 80:
            return Alert("WARNING", f"存储空间不足: {usage.percent}%")

    def check_sqlite_health(self):
        """检查 SQLite WAL 文件大小"""
        wal_size = Path('sqlite/market.db-wal').stat().st_size / 1024 / 1024
        if wal_size > 100:
            return Alert("WARNING", f"WAL 文件过大: {wal_size}MB")
```

---

## 十六、Schema 版本管理

### 16.1 版本历史

| 版本 | 新增字段 | 说明 |
|------|---------|------|
| v1 | date, open, high, low, close, volume | 初始版本 |
| v2 | amount, turnover, adj_factor | 增加财务字段 |
| v3 | pct_chg, is_halt | 增加涨跌停标记 |

### 16.2 迁移规则

```python
MIGRATIONS = {
    ('v1', 'v2'): lambda df: df.assign(
        amount=None,
        turnover=None,
        adj_factor=None
    ),
    ('v2', 'v3'): lambda df: df.assign(
        pct_chg=None,
        is_halt=False
    ),
}
```

### 16.3 自动迁移

```python
def read_parquet_with_version(path: str) -> Tuple[pd.DataFrame, str]:
    """读取 Parquet，自动检测并迁移版本"""
    pf = pq.ParquetFile(path)
    current_version = pf.schema_arrow.metadata.get(b'schema_version', b'v1').decode()

    df = pf.read().to_pandas()

    if current_version != SCHEMA_VERSION:
        df = SchemaMigrator.migrate(df, current_version, SCHEMA_VERSION)
        write_parquet_with_metadata(df, path, SCHEMA_VERSION)

    return df, current_version
```

---

## 十七、验收标准

### 12.1 功能验收

| 验收项 | 标准 | 验证方法 |
|--------|------|---------|
| 日线采集 | 全市场成功率 ≥ 99.5% | 日报统计 |
| 复权数据 | adj_factor 正确合并 | 单元测试 |
| 质量评分 | 分数 < 60 拒绝写入 | 边界测试 |
| 失败追溯 | 所有失败在日报体现 | 日报检查 |
| 幂等写入 | 重复执行无重复数据 | 集成测试 |
| 测试隔离 | 测试使用 tempfile | 集成测试 |

### 12.2 测试覆盖

| 模块 | 测试文件 | 测试数 | 状态 |
|------|---------|-------|------|
| 数据合并 | test_data_merger.py | 9 | ✅ 通过 |
| 质量评分 | test_quality_scorer.py | 8 | ✅ 通过 |
| 质量评分边界 | test_quality_scorer_edge.py | 13 | ✅ 通过 |
| 验证器 | test_validators.py | 8 | ✅ 通过 |
| 集成测试 | test_integration.py | 12 | ✅ 通过 |
| 回填模块 | test_backfill.py | 10 | ✅ 通过 |
| 价格转换 | test_price_converter.py | 9 | ✅ 通过 |
| 东方财富数据源 | test_eastmoney_source.py | 12 | ✅ 通过 |
| 指数采集器 | test_index_fetcher.py | 4 | ✅ 通过 |
| 核心股票加载器 | test_core_stock_loader.py | 5 | ✅ 通过 |
| **合计** | | **90** | **全部通过** |

## 价格转换功能

### 前复权与后复权

| 类型 | 公式 | 用途 |
|------|------|------|
| 后复权 | `close_adj = close * adj_factor` | 历史数据连续性分析 |
| 前复权 | `forward_close = close_adj * (latest_adj / hist_adj)` | 实盘真实成本 |

```python
from src.data.fetcher.price_converter import convert_to_forward_adj

# 读取后复权数据
df = pd.read_parquet("raw/daily/600519.parquet")

# 转换为前复权（用于实盘买入参考）
df = convert_to_forward_adj(df)
```

### EastMoneySource 资金流数据

```python
from src.data.fetcher.sources.eastmoney_source import EastMoneySource

em = EastMoneySource()

# 个股资金流（主力/超大单/大单/中单/小单）
df = em.fetch_individual_fund_flow("600519", "sh")

# 行业资金流排名
df = em.fetch_industry_fund_flow("今日")

# 北向资金（沪深港通）
df = em.fetch_hsgt_north_flow()
```

---

## 回填使用方法

```bash
# 回填最近5年数据
python -m src.data.fetcher.backfill \
  --start 2021-03-29 \
  --end 2026-03-28 \
  --stockdata-root /Users/bruce/workspace/trade/StockData

# 回填指定股票
python -m src.data.fetcher.backfill \
  --start 2021-03-29 \
  --end 2026-03-28 \
  --codes 600519,000001
```

## 预计回填耗时

| 股票数 | API调用 | 预计耗时 |
|--------|---------|---------|
| 全市场(~4000) | 8000 calls | ~3.7 小时 |
| 100只 | 200 calls | ~5.5 分钟 |
| 3只 | 6 calls | ~10 秒 |

---

---

## 十八、已知问题修复记录

### 18.1 tushare_source.py 验证顺序 bug

**问题**：`_validate_daily_dataframe()` 在 `rename()` 之前执行，期望 `date`/`volume` 列但原始列为 `trade_date`/`vol`，导致有效数据被误判。

**修复**：调整顺序，先 rename 再 validation。

**验证**：
- 2026-03-27 贵州茅台（600519）真实采集成功
- 质量评分 86.5 分 ≥ 80，通过
- 复权因子 8.4464，计算正确
- 50 个单元测试全部通过

---

### 18.2 P0 稳定性修复（2026-03-29）

| 问题 | 影响 | 修复 |
|------|------|------|
| 并发写入同一股票无锁 | 竞态条件导致数据丢失 | writer.py 添加 InterProcessLock |
| SMTP 失败无回退 | 告警静默丢失 | alert.py 添加本地文件回退 |
| SQLite 备份用 shutil.copy2 | WAL 模式下备份损坏 | backup.py 改用 sqlite3.backup API |
| 裸 `except Exception` 静默失败 | 数据损坏无法诊断 | loader.py 添加 logger.error |
| checkpoint 损坏丢失历史 | 告警冷却失效 | health_check.py 保留 last_alerts |

**验证**：152 个测试通过（test_migration.py 5 个失败为 pre-existing 问题）

---

**文档版本**: v1.6
**最后更新**: 2026-03-29
**更新内容**:
- v1.6: P0 稳定性修复（文件锁、告警回退、SQLite备份、错误日志），152测试通过
- v1.5: 扩大股票池至40只，新增IndexFetcher（6个宽基指数），置信度100%验证通过
- v1.4: 新增 EastMoneySource（资金流）、price_converter（前复权转换），81个测试通过
- v1.3: 新增历史数据回填模块 (BackfillFetcher)，10个测试通过
- v1.2: 修复 tushare_source.py 验证顺序 bug，真实验证通过
- v1.1: 补充数据分层策略、维护任务清单、备份策略、健康检查、Schema版本管理
- v1.0: 合并 design.md 和 fetcher_design.md
