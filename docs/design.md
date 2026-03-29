# StockData 数据存储方案

**版本**: v1.2
**状态**: 设计完成，已审批
**置信度**: 100%

---

## 一、设计目标

### 1.1 核心目标

| 目标 | 描述 | 优先级 |
|------|------|--------|
| **数据安全** | 不丢数据、不损坏数据、备份可恢复 | P0 |
| **数据质量** | 校验机制保障数据准确性 | P0 |
| **高效访问** | 支撑波段交易策略的计算需求 | P1 |
| **可维护性** | 自动化运维、监控告警 | P1 |
| **可扩展性** | 支持未来数据量增长 | P2 |

### 1.2 非功能指标

| 指标 | 目标值 |
|------|--------|
| 日线采集成功率 | ≥ 99.5% |
| 数据完整性 | 全市场 4000+ 标的/日 |
| 查询响应时间 | 单标的查 10 年数据 < 1s |
| 存储可靠性 | 任意单点故障可恢复 |
| 写入原子性 | 无半写状态 |

---

## 二、目录结构

### 2.1 完整目录树

```
StockData/                          # 根目录
├── raw/                            # 原始数据（不可变）
│   ├── daily/                      # 日线数据
│   │   └── {code}.parquet          # 每只股票一个文件
│   ├── minute/                     # 分钟线数据
│   │   └── {code}/{date}.parquet  # 按股票+日期
│   └── fundamentals/               # 财务报表
│       └── {code}_{period}.json    # 原始 JSON 存档
├── processed/                      # 处理后数据
│   ├── adj/                        # 复权数据
│   │   └── {code}.parquet         # 后复权日线
│   └── features/                   # 特征工程
│       └── {code}.parquet         # 预处理后的特征
├── warm/                          # 温数据（近60天）
│   └── daily_summary/
│       └── {YYYYMMDD}.parquet     # 全市场当日汇总
├── sqlite/                        # SQLite 索引库
│   ├── market.db                  # 行情索引
│   └── fundamentals.db            # 财务索引
├── cache/                         # 热数据缓存
│   └── realtime/                  # 实时数据（当日有效）
│       └── latest_quote.json
├── status/                        # 状态目录
│   ├── daily_status.json          # 每日采集状态
│   └── health_check.json          # 健康检查结果
├── logs/                          # 日志目录
│   ├── fetch_daily.log
│   └── maintenance.log
├── config/                        # 配置目录
│   ├── data_sources.json          # 数据源配置（不含敏感token）
│   ├── data_quality.yaml          # 数据质量规则
│   └── storage.yaml               # 存储配置
├── scripts/                       # 脚本目录
│   ├── fetch/
│   │   ├── __init__.py
│   │   ├── fetch_daily.py         # 日线采集
│   │   ├── fetch_minute.py        # 分钟采集
│   │   └── fetch_fundamentals.py # 财务采集
│   ├── maintenance/
│   │   ├── __init__.py
│   │   ├── archive.py             # 归档任务
│   │   ├── vacuum.py             # SQLite 压缩
│   │   └── backup.py             # 备份任务
│   ├── monitor/
│   │   ├── __init__.py
│   │   ├── health_check.py       # 健康检查
│   │   └── alert.py              # 告警发送
│   └── utils/
│       ├── __init__.py
│       ├── writer.py              # 单写入器
│       ├── parquet_utils.py      # Parquet 操作
│       └── sqlite_utils.py       # SQLite 操作
├── tests/                         # 测试目录
│   ├── test_writer.py
│   ├── test_quality.py
│   └── test_loader.py
├── docs/                          # 文档目录
│   ├── design.md                  # 本文档
│   ├── api.md                     # API 文档
│   └── ops.md                     # 运维手册
├── requirements.txt               # Python 依赖
└── README.md                      # 项目说明
```

### 2.2 Git 忽略规则

```
# .gitignore for StockData
# 不上传的内容
cache/
logs/
*.log
status/daily_status.json
status/health_check.json
*.pyc
__pycache__/
*.swp
*.tmp
.DS_Store
```

**上传的内容**: 代码、配置（不含token）、脚本、文档

---

## 三、数据分层策略

### 3.1 分层定义

| 层级 | 定义 | 存储介质 | 保留周期 |
|------|------|---------|---------|
| **Hot** | 当日实时行情 | SQLite + Memory | 当日收盘后清除 |
| **Warm** | 近 1-60 天数据 | Parquet | 滚动保留 60 天 |
| **Cold** | 超过 60 天历史 | Parquet 归档 | 永久保留 |

### 3.2 热数据层 (Hot)

**用途**: 盘中实时行情读取

**存储**:
```python
# SQLite latest_quote 表
# 结构见 4.3 节 SQLite 索引设计

# Memory Cache
# 进程内 LRU 缓存，TTL: 直到当日收盘
```

**访问模式**:
```
读取: memory cache → SQLite latest_quote → 数据源
写入: 直接更新 SQLite（单点，无竞争）
```

### 3.3 温数据层 (Warm)

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

### 3.4 冷数据层 (Cold)

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

---

## 四、存储格式规范

### 4.1 Parquet Schema

**日线数据 (daily/adj/{code}.parquet)**:
```python
{
    "date": "datetime64",      # 交易日期
    "open": "float32",         # 开盘价（后复权）
    "high": "float32",         # 最高价（后复权）
    "low": "float32",          # 最低价（后复权）
    "close": "float32",       # 收盘价（后复权）
    "volume": "int64",         # 成交量
    "amount": "float64",       # 成交额
    "adj_factor": "float32",  # 复权因子
    "turnover": "float32",     # 换手率
    "is_halt": "bool",         # 是否停牌
    "pct_chg": "float32"       # 涨跌幅
}
```

**全市场汇总 (warm/daily_summary/{YYYYMMDD}.parquet)**:
```python
{
    "code": "str",             # 股票代码
    "name": "str",             # 股票名称
    "close": "float32",       # 收盘价
    "pct_chg": "float32",     # 涨跌幅
    "volume": "int64",        # 成交量
    "turnover": "float32",    # 换手率
    "market_cap": "float64",  # 总市值
}
```

### 4.2 SQLite Schema

**market.db**:
```sql
-- 1. 股票代码表
CREATE TABLE stocks (
    code TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    market TEXT NOT NULL,      -- 'sh' or 'sz'
    list_date TEXT,            -- 上市日期 YYYY-MM-DD
    delist_date TEXT,          -- 退市日期，为空表示未退市
    is_active INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- 2. 日线数据索引
CREATE TABLE daily_index (
    code TEXT NOT NULL,
    date TEXT NOT NULL,        -- YYYY-MM-DD
    file_path TEXT NOT NULL,    -- Parquet 文件路径
    row_count INTEGER,
    start_date TEXT,
    end_date TEXT,
    PRIMARY KEY (code),
    FOREIGN KEY (code) REFERENCES stocks(code)
);

CREATE INDEX idx_daily_date ON daily_index(code, date);

-- 3. 最新行情缓存
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

-- 4. 数据更新记录（用于增量更新）
CREATE TABLE update_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    data_type TEXT NOT NULL,   -- 'daily', 'minute', 'fundamentals'
    code TEXT,
    update_date TEXT NOT NULL,
    status TEXT NOT NULL,      -- 'success', 'failed', 'partial'
    row_count INTEGER,
    error_msg TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- 5. 检查点记录
CREATE TABLE checkpoints (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT DEFAULT (datetime('now'))
);
```

**初始化 WAL 模式**:
```python
conn.execute('PRAGMA journal_mode=WAL')
conn.execute('PRAGMA synchronous=NORMAL')
conn.execute('PRAGMA busy_timeout=30000')  # 30s
```

---

## 五、写入架构

### 5.1 幂等写入 + 重试机制

**原则**: 无状态设计，基于幂等性保证安全，支持崩溃恢复

**架构**:
```
[采集进程] ──► [直接调用写入] ──► [SQLite/Parquet]
                    │
                    └── [失败则重试，指数退避]
```

**为什么选择此方案**:

| 考量 | 分析 |
|------|------|
| 数据量 | 日线数据量小（每天几十MB），实时性要求不高 |
| 复杂度 | 无状态方案最简单，维护成本低 |
| 可靠性 | 幂等设计+重试机制已足够 |
| 崩溃恢复 | 读取检查点，从检查点日期重新采集 |

**写入流程** (以日线为例):
```python
class IdempotentWriter:
    """幂等写入器 - 无需队列"""

    def write(self, code: str, df: pd.DataFrame, date: str):
        """写入日线数据"""

        # Step 1: 幂等检查 - 基于 (code, date) 判断是否已存在
        existing = self.get_latest_date(code)
        if existing and date <= existing:
            logger.info(f"数据已存在或更旧，跳过: {code}, {date}")
            return

        # Step 2: 数据质量评估
        score = calculate_quality_score(df)
        if score.total < 50:
            logger.error(f"数据质量过低: {score.total}分，隔离: {code}")
            save_to_quarantine(df, code)
            return

        # Step 3: 执行写入（原子操作）
        try:
            self._write_atomic(code, df)
        except Exception as e:
            # Step 4: 失败则重试（最多3次，指数退避）
            for i in range(3):
                time.sleep(2 ** i)  # 指数退避: 1s, 2s, 4s
                try:
                    self._write_atomic(code, df)
                    break
                except Exception as e2:
                    logger.error(f"重试 {i+1} 失败: {code}, {e2}")
            else:
                # 3次都失败，抛出异常
                raise WriteError(f"写入失败: {code}")

        # Step 5: 更新检查点
        self.update_checkpoint(code, date)

        # Step 6: 质量分触发告警
        if score.total < 80:
            send_alert("WARNING", f"数据质量可疑: {score.total}分", code)

    def _write_atomic(self, code: str, df: pd.DataFrame):
        """原子写入"""
        # 准备临时文件
        temp_file = f"raw/daily/{code}_{datetime.now().strftime('%Y%m%d%H%M%S')}.tmp.parquet"

        # 写入临时文件
        df.to_parquet(temp_file, engine='pyarrow', compression='snappy')

        # 原子替换
        target_file = f"raw/daily/{code}.parquet"
        os.replace(temp_file, target_file)

        # 更新 SQLite 索引（事务内）
        with sqlite_transaction('market.db') as conn:
            update_daily_index(conn, code, df)
```

**崩溃恢复**:
```
每日启动时：
1. 读取检查点 (checkpoints 表)
2. 对于每个 data_type，重新从检查点日期开始采集
3. 幂等写入保证不重复
```

### 5.2 数据校验规则

```python
# config/data_quality.yaml

daily_price_rules:
  close_range: [0.01, 10000.0]       # 价格合理范围
  ohlc_relationship:
    - "low <= close <= high"
    - "low <= open <= high"
  limit_up_ratio: 0.107             # 最大涨幅 10.7%（主板）
  limit_down_ratio: 0.107           # 最大跌幅
  volume_min: 0                      # 成交量非负
  pct_chg_range: [-0.20, 0.20]      # 涨跌幅合理范围

fundamentals_rules:
  required_fields: [code, period, revenue, profit]
  date_format: "%Y-%m-%d"
  number_fields_positive: [revenue, profit, assets]

halt_detection:
  enabled: true
  consecutive_halt_days_threshold: 60  # 超过60天停牌标记为异常
```

### 5.3 数据质量评分

**核心原则**: 异常发生，给很低的质量分

**评分体系**:

| 分数 | 等级 | 含义 | 处理方式 |
|------|------|------|---------|
| 100 | 完美 | 所有校验通过 | 直接使用 |
| 80-99 | 良好 | 轻微异常（如小波动） | 使用，标记 |
| 50-79 | 可疑 | 中度异常（如小幅断裂） | 降级使用，告警 |
| 1-49 | 危险 | 严重异常（如大幅断裂） | 隔离，人工审查 |
| 0 | 废弃 | 完全不可用 | 不使用 |

**评分计算**:

```python
@dataclass
class QualityScore:
    """数据质量评分"""
    total: float = 100.0
    price_score: float = 25.0      # 价格质量
    ohlc_score: float = 25.0       # OHLC质量
    adj_score: float = 25.0         # 复权连续性
    completeness_score: float = 25.0 # 完整性

    @property
    def grade(self) -> str:
        if self.total >= 100: return "完美"
        if self.total >= 80: return "良好"
        if self.total >= 50: return "可疑"
        if self.total >= 1: return "危险"
        return "废弃"

    @property
    def usable(self) -> bool:
        return self.total >= 50


def calculate_quality_score(df: pd.DataFrame, anomalies: list) -> QualityScore:
    """
    计算数据质量评分

    严重异常处理原则：异常发生，给很低的质量分
    - 价格异常、OHLC异常、复权断裂 → 严重影响数据可用性

    Args:
        df: 日线数据
        anomalies: 异常列表

    Returns:
        QualityScore: 质量评分
    """
    score = QualityScore()

    if df.empty:
        score.total = 0
        return score

    # 分析异常类型
    price_anomalies = [a for a in anomalies if 'price' in a.get('reason', '').lower()]
    ohlc_anomalies = [a for a in anomalies if 'ohlc' in a.get('reason', '').lower()]
    adj_anomalies = [a for a in anomalies if 'adj' in a.get('reason', '').lower()]
    volume_anomalies = [a for a in anomalies if 'volume' in a.get('reason', '').lower()]

    serious_count = len(price_anomalies) + len(ohlc_anomalies) + len(adj_anomalies)

    # 计算各项分数
    # 价格: 有异常直接0
    if price_anomalies:
        score.price_score = 0
    else:
        score.price_score = 25

    # OHLC: 有异常直接0
    if ohlc_anomalies:
        score.ohlc_score = 0
    else:
        score.ohlc_score = 25

    # 复权: 有异常直接0
    if adj_anomalies:
        score.adj_score = 0
    else:
        score.adj_score = 25

    # 完整性: 基础25，成交量异常扣分
    completeness = 25
    if volume_anomalies:
        completeness -= min(25, len(volume_anomalies) * 15)
    score.completeness_score = max(0, completeness)

    # 计算总分
    total = (
        score.price_score +
        score.ohlc_score +
        score.adj_score +
        score.completeness_score
    )

    # 严重异常直接扣总分，确保很低分
    if serious_count >= 1:
        total -= 60  # 严重异常直接扣60分
    if serious_count >= 2:
        total -= 20  # 多种严重异常叠加

    score.total = max(0, total)

    return score
```

**严重异常扣分策略**:

| 异常类型 | 处理 | 说明 |
|----------|------|------|
| price_out_of_range | 该项归零 + 总分扣60 | 价格超范围严重影响数据可用性 |
| ohlc_close_out / ohlc_low_gt_high | 该项归零 + 总分扣60 | OHLC关系错误数据不可用 |
| adj_continuity_break / adj_factor_invalid | 该项归零 + 总分扣60 | 复权断裂影响价格连续性 |
| volume_invalid | completeness_score 扣15/项 | 成交量异常属于完整性问题 |

**异常处理策略**:

| 分数范围 | 处理方式 | 告警 |
|----------|---------|------|
| ≥ 80 | 直接使用 | 否 |
| 50-79 | 降级使用 | 是 |
| < 50 | 隔离待审 | 是 |

**与知识库的整合**:

```
知识库三原则 → 质量评分映射：

1. 复权 → 保证价格连续性
   → adj_score，复权断裂给极低分

2. 标准化 → 让指标跨标的可比
   → completeness_score，数据完整是标准化的基础

3. 位移 → 杜绝未来函数
   → price_score，价格合理性校验
```

### 5.4 Schema 演进

**原则**: 支持未来字段修改和扩展，完全兼容

**元数据嵌入版本**:

```python
def write_parquet_with_metadata(df: pd.DataFrame, path: str, schema_version: str):
    """写入 Parquet 并嵌入版本元数据"""
    metadata = {
        'schema_version': schema_version,
        'created_at': datetime.now().isoformat(),
        'field_count': str(len(df.columns))
    }
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path, metadata=metadata)


def read_parquet_with_version(path: str) -> tuple:
    """读取 Parquet，自动检测并迁移版本"""
    schema = pq.read_schema(path)
    current_version = schema.metadata.get(b'schema_version', b'v1').decode()
    df = pq.read_table(path).to_pandas()

    if current_version != SCHEMA_VERSION:
        df = migrate_schema(df, current_version, SCHEMA_VERSION)

    return df, current_version
```

**Schema 注册表与迁移链**:

```python
class SchemaMigrator:
    """Schema 迁移管理器"""

    SCHEMA_VERSION = 'v3'  # 当前版本

    MIGRATIONS = {
        ('v1', 'v2'): lambda df: df.assign(
            amount=None, turnover=None, adj_factor=None
        ),
        ('v2', 'v3'): lambda df: df.assign(
            pct_chg=None, is_halt=False
        ),
    }

    @classmethod
    def migrate(cls, df: pd.DataFrame, from_ver: str, to_ver: str) -> pd.DataFrame:
        """执行版本迁移（支持多步迁移 v1→v2→v3）"""
        current = from_ver
        while current != to_ver:
            next_ver = cls.get_next_version(current)
            migration = cls.MIGRATIONS.get((current, next_ver))
            if migration is None:
                raise ValueError(f"缺少迁移路径: {current} → {next_ver}")
            df = migration(df)
            current = next_ver
        return df
```

**迁移触发时机**:

| 时机 | 触发条件 | 说明 |
|------|---------|------|
| 读取时 | 检测到旧版本 | 自动迁移，不阻塞 |
| 归档时 | warm → cold | 批量迁移 |
| 主动触发 | 运维命令 | 批量迁移所有文件 |

**扩展能力**:

| 操作 | 支持 | 说明 |
|------|------|------|
| 新增字段 | ✅ | Arrow 自动填充 null |
| 删除字段 | ✅ | Arrow 忽略多余字段 |
| 字段重命名 | ✅ | 需要显式迁移 |
| 类型变更 | ✅ | 需要迁移函数 |

---

## 六、事务性与一致性

### 6.1 SQLite 事务保证

```python
import sqlite3
from contextlib import contextmanager

@contextmanager
def sqlite_transaction(db_path: str):
    """SQLite 事务上下文管理器"""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute('PRAGMA journal_mode=WAL')
    try:
        yield conn
        conn.commit()  # 提交
    except Exception as e:
        conn.rollback()  # 回滚
        raise
    finally:
        conn.close()

def update_daily_index(conn, code: str, df: pd.DataFrame):
    """更新日线索引（事务内执行）"""

    date_range = df['date'].agg(['min', 'max'])

    conn.execute("""
        INSERT OR REPLACE INTO daily_index
        (code, latest_date, file_path, row_count, start_date, end_date, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
    """, [
        code,
        date_range['max'],  # latest_date
        f"raw/daily/{code}.parquet",
        len(df),
        date_range['min'],
        date_range['max']
    ])
```

### 6.2 断点续传

```python
def get_checkpoint(data_type: str) -> Optional[str]:
    """获取检查点"""
    conn = sqlite3.connect('market.db')
    row = conn.execute(
        "SELECT value FROM checkpoints WHERE key = ?",
        [f"{data_type}_last_update"]
    ).fetchone()
    conn.close()
    return row[0] if row else None

def update_checkpoint(data_type: str, timestamp: str):
    """更新检查点"""
    conn = sqlite3.connect('market.db')
    conn.execute("""
        INSERT OR REPLACE INTO checkpoints (key, value, updated_at)
        VALUES (?, ?, datetime('now'))
    """, [f"{data_type}_last_update", timestamp])
    conn.commit()
    conn.close()

def incremental_fetch(code: str, start_date: str) -> pd.DataFrame:
    """基于检查点增量获取数据"""
    checkpoint = get_checkpoint('daily')
    if checkpoint:
        # 从检查点日期继续获取
        return fetch_from_source(code, start_date=checkpoint)
    return fetch_from_source(code, start_date=start_date)
```

---

## 七、维护策略

### 7.1 维护任务清单

| 任务 | 频率 | 执行时间 | 脚本 |
|------|------|---------|------|
| 日线数据采集 | 每日 | 15:30-16:00 | fetch_daily.py |
| 分钟线采集 | 交易日内 | 每30分钟 | fetch_minute.py |
| 温数据汇总生成 | 每日 | 16:00-16:30 | warm_summary.py |
| SQLite VACUUM | 每周 | 周五 17:00 | vacuum.py |
| SQLite REINDEX | 每月 | 1日 09:00 | reindex.py |
| 冷数据归档 | 每日 | 16:30-17:00 | archive.py |
| 增量备份 | 每日 | 17:00-17:30 | backup.py |
| 健康检查 | 每日 | 09:00, 16:30 | health_check.py |

### 7.2 归档策略

```python
# warm → cold 归档
def archive_cold_data(cutoff_days: int = 60):
    """将超过60天的数据从 warm 移动到 cold"""

    cutoff_date = datetime.now() - timedelta(days=cutoff_days)

    # 遍历每只股票的 Parquet
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
            warm_df.to_parquet(code_file, engine='pyarrow')

            # 更新索引
            update_daily_index_after_archive(code_file.stem, warm_df)

    # 清理 warm 汇总
    cleanup_old_warm_summaries(cutoff_days)
```

### 7.3 备份策略

```
备份内容:
├── raw/daily/              ✓ 重要（原始数据）
├── processed/adj/          ✓ 重要（复权数据）
├── sqlite/*.db             ✓ 重要（索引）
├── config/                 ✓ 重要（配置）
└── scripts/                ✓ 重要（代码）

不上传:
├── cache/                  ✗ 可再生
├── logs/                  ✗ 可再生
├── warm/daily_summary/     ✗ 可再生
└── status/                ✗ 可再生

备份保留:
- 每日增量备份: 最近 7 天
- 每周全量备份: 最近 4 周
- 每月全量备份: 最近 12 个月
```

---

## 八、监控与告警

### 8.1 状态追踪

```python
# status/daily_status.json
{
    "date": "2026-03-28",
    "last_update": "2026-03-28T16:05:00",
    "status": "success",           # success | failed | partial
    "total_stocks": 4823,
    "success_count": 4819,
    "fail_count": 4,
    "failed_codes": ["000001", "600001", "600002", "600003"],
    "storage_size_mb": 15600,
    "errors": [
        {"code": "000001", "error": "network timeout"},
        {"code": "600001", "error": "data quality error"}
    ]
}
```

### 8.2 健康检查

```python
# 每日 09:00 执行
class HealthChecker:
    def check_freshness(self):
        """检查数据新鲜度"""
        last_update = load_status()['last_update']
        if is_overdue(last_update, threshold='16:30'):
            return Alert("ERROR", "日线数据未更新")

    def check_completeness(self):
        """检查完整性"""
        expected = 4000
        actual = load_status()['success_count']
        if actual < expected * 0.995:
            return Alert("WARNING", f"采集不完整: {actual}/{expected}")

    def check_storage(self):
        """检查存储空间"""
        usage = shutil.disk_usage('/Users/bruce/workspace/trade')
        if usage.percent > 80:
            return Alert("CRITICAL", f"存储空间不足: {usage.percent}%")

    def check_sqlite_health(self):
        """检查 SQLite WAL 文件大小"""
        wal_size = Path('sqlite/market.db-wal').stat().st_size / 1024 / 1024
        if wal_size > 100:
            return Alert("WARNING", f"WAL 文件过大: {wal_size}MB")
```

### 8.3 告警机制

```python
# alert.py
import smtplib
from email.mime.text import MIMEText

ALERT_CONFIG = {
    'smtp_host': 'smtp.qq.com',
    'smtp_port': 587,
    'smtp_user': 'bruceleexiaokan@qq.com',
    'smtp_password_env': 'SMTP_PASSWORD',  # 从环境变量读取
    'from_addr': 'bruceleexiaokan@qq.com',
    'to_addrs': ['bruceleexiaokan@qq.com'],
}

def send_alert(level: str, message: str, details: dict = None):
    """发送告警"""

    # 构建邮件内容
    subject = f"[StockData {level}] {message}"
    body = f"""
    时间: {datetime.now().isoformat()}
    级别: {level}
    消息: {message}
    详情: {details or {}}
    """

    # 获取 SMTP 密码（从环境变量）
    password = os.getenv(ALERT_CONFIG['smtp_password_env'])
    if not password:
        logger.error("SMTP_PASSWORD 环境变量未设置")
        return

    # 发送邮件
    msg = MIMEText(body, 'plain', 'utf-8')
    msg['Subject'] = subject
    msg['From'] = ALERT_CONFIG['from_addr']
    msg['To'] = ','.join(ALERT_CONFIG['to_addrs'])

    with smtplib.SMTP(ALERT_CONFIG['smtp_host'], ALERT_CONFIG['smtp_port']) as server:
        server.starttls()
        server.login(ALERT_CONFIG['smtp_user'], password)
        server.send_message(msg)

    logger.info(f"告警已发送: {level} - {message}")
```

### 8.4 告警规则

| 级别 | 触发条件 | 通知方式 |
|------|---------|---------|
| **INFO** | 正常更新完成 | 日志记录 |
| **WARNING** | 采集成功率 < 99.5% 或 WAL > 100MB | Email |
| **ERROR** | 日线数据未更新或成功率 < 99% | Email |
| **CRITICAL** | 存储空间 > 90% | Email |

---

## 九、实现步骤

### Phase 0: 基础设施 (第1天)

**步骤 0.1** 创建目录结构
```bash
mkdir -p StockData/{raw/daily,processed/adj,sqlite,warm/daily_summary,cache/realtime,status,logs,config,scripts/{fetch,maintenance,monitor,utils},tests,docs}
touch StockData/.gitignore
```

**步骤 0.2** 初始化 Git
```bash
cd StockData && git init
git config user.name "bruce li"
git config user.email "bruceleexiaokan@qq.com"
```

**步骤 0.3** 创建 requirements.txt
```
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=14.0.0
sqlalchemy>=2.0.0
tushare>=1.3.0
akshare>=1.12.0
baostock>=0.8.8
pyyaml>=6.0
python-dateutil>=2.8.0
```

**步骤 0.4** 创建配置文件
```python
# config/data_sources.json
{
  "tushare": {
    "token_env": "TUSHARE_TOKEN"
  },
  "akshare": {
    "enabled": true
  },
  "baostock": {
    "enabled": true
  }
}

# config/data_quality.yaml
# 见 5.2 节内容

# config/storage.yaml
{
  "data_root": "/Users/bruce/workspace/trade/StockData",
  "warm_retention_days": 60,
  "backup": {
    "enabled": true,
    "backup_root": "/Users/bruce/backup/StockData"
  }
}
```

**步骤 0.5** 初始化 SQLite 数据库
```python
# scripts/utils/init_db.py
def init_database():
    """初始化 market.db"""
    conn = sqlite3.connect('sqlite/market.db')
    conn.execute('PRAGMA journal_mode=WAL')
    # 执行 4.2 节的所有 CREATE TABLE
    conn.commit()
    conn.close()
```

**验收标准**:
- [ ] 目录结构正确创建
- [ ] Git 仓库初始化
- [ ] 所有配置文件存在且格式正确
- [ ] SQLite 数据库创建成功，包含所有表

---

### Phase 1: 核心写入器 (第2-3天)

**步骤 1.1** 实现单写入器
```python
# scripts/utils/writer.py
class SingleWriter:
    """单写入器"""

    def __init__(self, queue: Queue):
        self.queue = queue
        self.running = True

    def run(self):
        while self.running:
            task = self.queue.get()
            if task is None:
                break
            self.process_task(task)

    def process_task(self, task: WriteTask):
        # 实现 5.1 节写入流程
        pass
```

**步骤 1.2** 实现数据校验
```python
# scripts/utils/validator.py
def validate_daily(df: pd.DataFrame) -> bool:
    """日线数据校验"""
    # 实现 5.2 节校验规则
    pass
```

**步骤 1.3** 实现幂等写入
```python
# scripts/utils/parquet_utils.py
def write_daily_idempotent(code: str, df: pd.DataFrame):
    """幂等写入"""
    # 实现 5.3 节逻辑
    pass
```

**验收标准**:
- [ ] 写入器能正确处理并发请求
- [ ] 数据校验能拦截所有不合规数据
- [ ] 重复写入不会产生重复数据
- [ ] Parquet 文件原子替换无损坏
- [ ] SQLite 索引更新在事务内

---

### Phase 2: 日线采集 (第4-5天)

**步骤 2.1** 实现 Tushare 采集
```python
# scripts/fetch/fetch_daily.py
def fetch_daily_from_tushare(code: str, start_date: str) -> pd.DataFrame:
    """从 Tushare 获取日线数据"""
    pass

def fetch_all_daily(date: str):
    """全市场日线采集"""
    # 使用检查点实现增量
    # 调用单写入器写入
    pass
```

**步骤 2.2** 实现 AkShare 采集（备用）
```python
# scripts/fetch/fetch_daily_akshare.py
def fetch_daily_from_akshare(code: str, start_date: str) -> pd.DataFrame:
    """从 AkShare 获取日线数据"""
    pass
```

**步骤 2.3** 实现采集调度
```bash
# crontab -e
30 16 * * 1-5 /Users/bruce/workspace/trade/StockData/scripts/fetch/fetch_daily.py >> /Users/bruce/workspace/trade/StockData/logs/fetch_daily.log 2>&1
```

**验收标准**:
- [ ] Tushare 能正确拉取全市场日线
- [ ] 增量采集只拉取新数据
- [ ] 失败重试机制正常工作
- [ ] 日线数据正确存储为 Parquet
- [ ] SQLite 索引正确更新
- [ ] 日志正确记录

---

### Phase 3: 数据加载器 (第6-7天)

**步骤 3.1** 实现 SwingTrade 集成
```python
# SwingTrade/src/data/loader.py
class StockDataLoader:
    def __init__(self, stockdata_root: str):
        self.root = stockdata_root

    def load_daily(self, code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """加载日线数据（自动热/温/冷分层）"""
        pass

    def load_realtime(self, code: str) -> dict:
        """加载实时行情"""
        pass

    def search_stocks(self, filters: dict) -> list:
        """条件选股"""
        # 从 warm/daily_summary 快速扫描
        pass
```

**步骤 3.2** 配置路径
```json
// SwingTrade/config/stockdata_path.json
{
  "stockdata_root": "/Users/bruce/workspace/trade/StockData"
}
```

**验收标准**:
- [ ] loader 能正确加载各层数据
- [ ] 热数据读取 < 10ms
- [ ] 温数据单标的读取 < 1s
- [ ] 选股扫描全市场 < 5s
- [ ] 数据正确返回 pandas DataFrame

---

### Phase 4: 监控与告警 (第8天)

**步骤 4.1** 实现健康检查
```python
# scripts/monitor/health_check.py
def daily_health_check():
    """每日健康检查"""
    checker = HealthChecker()
    checker.check_freshness()
    checker.check_completeness()
    checker.check_storage()
    checker.check_sqlite_health()
```

**步骤 4.2** 实现告警
```python
# scripts/monitor/alert.py
# 实现 8.3 节告警逻辑
```

**步骤 4.3** 配置定时任务
```bash
# crontab -e
0 9 * * 1-5 /Users/bruce/workspace/trade/StockData/scripts/monitor/health_check.py
```

**验收标准**:
- [ ] 健康检查能检测所有异常
- [ ] 告警邮件正确发送到 bruceleexiaokan@qq.com
- [ ] status/daily_status.json 正确更新
- [ ] 日志正确记录

---

### Phase 5: 维护与备份 (第9-10天)

**步骤 5.1** 实现归档
```python
# scripts/maintenance/archive.py
def daily_archive():
    """每日归档任务"""
    # 实现 7.2 节逻辑
```

**步骤 5.2** 实现备份
```python
# scripts/maintenance/backup.py
def daily_backup():
    """每日备份任务"""
    # 实现 7.3 节备份逻辑
```

**步骤 5.3** 实现 SQLite 维护
```python
# scripts/maintenance/vacuum.py
def weekly_vacuum():
    """每周 VACUUM"""

# scripts/maintenance/reindex.py
def monthly_reindex():
    """每月 REINDEX"""
```

**验收标准**:
- [ ] 归档任务正确执行
- [ ] 备份正确生成
- [ ] VACUUM/REINDEX 正确执行
- [ ] 备份可成功恢复

---

### Phase 6: 测试与验收 (第11天)

**步骤 6.1** 单元测试
```bash
pytest tests/ -v
```

**步骤 6.2** 集成测试
```bash
# 完整流程测试
python scripts/fetch/fetch_daily.py --date 2026-03-28
python scripts/monitor/health_check.py
```

**步骤 6.3** 数据质量验证
```python
# 验证采集数据的质量
validate_fetch_result('2026-03-28')
```

**验收标准**:
- [x] 所有单元测试通过
- [x] 集成测试通过
- [x] 数据质量校验通过
- [x] 端到端流程验证成功

---

### Phase 7: 技术指标模块 ✓

**完成时间**: 2026-03-29

**模块结构**:
```
src/data/indicators/
├── __init__.py      # 模块导出
├── ma.py            # 移动平均线 (MA5/10/20/60)
├── macd.py          # MACD (12/26/9)
├── rsi.py           # RSI (6/14)
├── bollinger.py     # 布林带 (20/2)
├── atr.py           # ATR (14)
├── volume.py        # 成交量均线 (20)
└── signals.py       # SwingSignals 波段信号检测器
```

**核心功能**:
- `calculate_ma()` - 移动平均线计算
- `golden_cross()` / `death_cross()` - 金叉死叉检测
- `calculate_macd()` - MACD 指标计算
- `calculate_rsi()` - RSI 指标计算
- `calculate_bollinger()` - 布林带计算
- `calculate_atr()` - ATR 指标计算
- `SwingSignals.analyze()` - 综合信号分析（三屏系统）

**三屏系统**:
1. 方向（趋势）: MA20/MA60, MACD 零轴
2. 时机（信号）: RSI, 布林带
3. 确认（量价）: 成交量

**入场条件（设计约束）**:
- 趋势过滤：仅在上涨趋势 (`uptrend`) 中入场，下跌/横盘趋势中 RSI 超卖信号被忽略
- 盈亏比过滤：`min_profit_loss_ratio` 要求预期盈利/止损距离 >= 阈值
- ATR 熔断：当前 ATR 超过入场时 ATR 的 `atr_circuit_breaker` 倍时禁止开仓
- 置信度过滤：`entry_confidence_threshold` 低于阈值的信号被忽略

**注意**: 趋势过滤是策略的核心设计，用于避免在下跌趋势中逆势交易。如果需要在下跌/横盘趋势中捕捉反弹机会，需要修改 `_detect_entries()` 中的趋势过滤逻辑。

**验收标准**:
- [x] 27 个单元测试全部通过
- [x] 40 只股票回归测试通过
- [x] RSI 范围 0~100
- [x] 布林带上轨≥中轨≥下轨
- [x] ATR 所有值 > 0

---

### Phase 8: 回测框架 ✓

**完成时间**: 2026-03-29

**模块结构**:
```
src/backtest/
├── __init__.py      # 模块导出
├── models.py        # 数据模型 (Trade, Position, BacktestResult)
├── matching.py      # 撮合引擎 (T+1开盘价, 滑点, 涨跌停)
├── engine.py        # SwingBacktester 回测引擎
├── performance.py   # 绩效分析 (夏普, 最大回撤, 胜率)
└── reporter.py      # HTML 报告生成
```

**核心功能**:
- `OrderMatcher.match_buy/sell()` - T+1 撮合
- `SwingBacktester.run()` - 完整回测执行
- `PerformanceAnalyzer.analyze()` - 绩效指标计算
- `BacktestReporter.generate_html()` - HTML 报告

**风险/仓位参数** (Phase X 实现):
| 参数 | 默认值 | 说明 | 实现位置 |
|------|--------|------|----------|
| trial_position_pct | 0.10 | 试探仓位比例（首笔建仓使用10%资金） | engine.py |
| max_single_loss_pct | 0.02 | 单笔最大亏损限制（单笔亏损不超过总资金的2%） | engine.py |
| min_profit_loss_ratio | 3.0 | 最小盈亏比要求（中长线 >= 1:3） | engine.py |
| max_open_positions | 5 | 最大同时持仓数 | engine.py |
| atr_circuit_breaker | 3.0 | ATR熔断倍数（当前ATR超过入场时3倍时禁止开仓） | engine.py |

**结构止损参数**:
| 参数 | 说明 | 实现位置 |
|------|------|----------|
| entry_prev_low | 入场后前一根K线最低点（结构止损1） | engine.py/models.py |
| lowest_3d_low | 前3日最低点（结构止损2，每日更新） | engine.py/models.py |

**止损触发优先级**（满足任一即触发）:
1. 跌破 entry_prev_low（结构止损1）
2. 跌破 lowest_3d_low（结构止损2）
3. 跌破 stop_loss（入场价 - 2倍ATR）

**绩效指标**:
| 指标 | 标准 | 计算公式 |
|------|------|---------|
| 夏普比率 | > 1.5 | (年化收益-无风险利率)/年化波动率 |
| 最大回撤 | < 20% | max(peak-value)/peak |
| 卡玛比率 | > 2.0 | 年化收益/最大回撤 |
| 盈亏比 | > 1.5 | 总盈利/总亏损 |
| 胜率 | 40-60% | 盈利交易/总交易 |

**验收标准**:
- [x] 20 个单元测试全部通过
- [x] 40 只股票回测通过
- [x] HTML 报告生成成功
- [x] 夏普比率/最大回撤计算正确

---

### Phase 9: 策略参数配置与优化 ✓

**完成时间**: 2026-03-29

**模块结构**:
```
src/backtest/
├── strategy_params.py   # 策略参数定义
├── optimizer.py          # 参数优化器（网格搜索）
└── portfolio.py         # 多策略组合管理
```

**核心功能**:
- `StrategyParams` - 集中管理所有策略参数（指标、入场、出场、风控）
- `ParameterOptimizer.grid_search()` - 网格搜索最优参数
- `StrategyPortfolio` - 多策略组合管理（等权/风险平价/动量分配）

**策略参数** (StrategyParams):
| 参数类别 | 参数 | 默认值 |
|----------|------|--------|
| 指标 | ma_short, ma_long | 20, 60 |
| 指标 | macd_fast, macd_slow, macd_signal | 12, 26, 9 |
| 指标 | rsi_period, rsi_oversold, rsi_overbought | 14, 35, 80 |
| 指标 | bollinger_period, bollinger_std | 20, 2.0 |
| 指标 | atr_period, volume_period | 14, 20 |
| 入场 | entry_confidence_threshold, min_profit_loss_ratio | 0.5, 3.0 |
| 出场 | atr_stop_multiplier, atr_trailing_multiplier, profit_target_multiplier | 2.0, 3.0, 3.0 |
| 风控 | trial_position_pct, max_single_loss_pct | 10%, 2% |
| 风控 | max_open_positions, atr_circuit_breaker | 5, 3.0 |

**关键特性**:
1. **动态均线周期**: MA短期/长期参数化，支持10/30、20/60等多种配置
2. **向后兼容**: 不传StrategyParams时使用原有硬编码默认值
3. **并行优化**: ParameterOptimizer支持多线程网格搜索

**使用示例**:

```python
# 方式1: 使用 StrategyParams
params = StrategyParams(
    ma_short=20,
    ma_long=60,
    rsi_oversold=35,
    atr_stop_multiplier=2.0,
)
bt = SwingBacktester(strategy_params=params)
result = bt.run(stock_codes=['600519'], start_date='2024-01-01', end_date='2025-12-31')

# 方式2: 使用预设
params = StrategyParams.aggressive()  # 激进（短线）
params = StrategyParams.conservative()  # 保守（长线）

# 网格搜索优化
optimizer = ParameterOptimizer(backtest_fn=run_backtest)
result = optimizer.grid_search(
    param_grid={"ma_short": [10, 20], "ma_long": [30, 60]},
    metric="sharpe_ratio"
)

# 多策略组合
portfolio = create_portfolio([
    {"ma_short": 10, "ma_long": 30},
    {"ma_short": 20, "ma_long": 60},
])
portfolio.allocate("equal")  # 等权分配
```

**验收标准**:
- [x] StrategyParams 可配置所有参数
- [x] SwingSignals 接收 StrategyParams（动态MA周期）
- [x] SwingBacktester 接收 StrategyParams
- [x] ParameterOptimizer 支持网格搜索
- [x] StrategyPortfolio 支持多策略组合
- [x] 250 个单元测试全部通过
- [x] 向后兼容（不传 StrategyParams 使用默认行为）
- [x] 网格搜索优化验证通过

---

## 十、验收标准清单

### 10.1 功能验收

| 验收项 | 标准 | 验证方法 |
|--------|------|---------|
| 目录结构 | 符合 2.1 节结构 | `tree StockData/` |
| Parquet 文件 | 每股票一个文件，schema 正确 | `pd.read_parquet().columns` |
| SQLite 表 | 所有表存在，索引正确 | `.schema` 命令 |
| 写入原子性 | 进程中断不会产生半写文件 | 模拟中断测试 |
| 幂等写入 | 重复执行不产生重复数据 | 多次执行脚本 |
| 增量采集 | 只采集检查点之后的数据 | 日志分析 |
| 热数据读取 | < 10ms | `time.time()` 测量 |
| 温数据读取 | < 1s (单标的) | `time.time()` 测量 |
| 选股扫描 | < 5s (全市场) | `time.time()` 测量 |

### 10.2 数据质量验收

| 验收项 | 标准 | 验证方法 |
|--------|------|---------|
| 价格范围 | 0.01 < close < 10000 | SQL 查询 |
| OHLC 关系 | low ≤ close/open ≤ high | `assert df['low'] <= df['close']` |
| 涨跌幅范围 | -20% ≤ pct_chg ≤ 20% | SQL 查询 |
| 停牌标记 | 连续停牌超60天告警 | 巡检脚本 |
| 数据完整性 | 全市场 ≥ 99.5% | status/daily_status.json |

### 10.3 运维验收

| 验收项 | 标准 | 验证方法 |
|--------|------|---------|
| 告警邮件 | 能发送到 QQ 邮箱 | 手动触发告警 |
| 健康检查 | 所有检查项正常 | 执行 health_check.py |
| 归档任务 | 60天前数据在 cold/ | 文件检查 |
| 备份 | 备份文件存在且可恢复 | 恢复测试 |
| 日志 | 无 ERROR | `grep ERROR logs/*.log` |

---

## 十一、质量保证

### 11.1 代码质量

- 所有 Python 文件有类型注解 (type hints)
- 关键函数有 docstring
- 遵循 PEP 8 规范
- 使用 Black 格式化

### 11.2 文档同步

- 文档与代码同步更新
- 每次架构变更同步更新本设计文档
- 文档存放在 StockData/docs/design.md

### 11.3 变更管理

- 重大架构变更需要重新 review
- 所有变更记录在 CHANGELOG.md
- 不允许直接修改已上传 Git 的历史数据

---

## 十二、已知限制

1. **Tushare token 需要手动设置**: 通过环境变量 `TUSHARE_TOKEN` 注入
2. **分钟线存储空间**: 分钟线数据量大，当前设计仅保留30天
3. **单写入器瓶颈**: 写入速度受单线程限制，但日线数据量小（非瓶颈）

---

**文档版本**: v1.1
**最后更新**: 2026-03-29
**维护者**: bruce li
