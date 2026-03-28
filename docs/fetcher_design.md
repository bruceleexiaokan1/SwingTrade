# StockData 数据采集模块设计

**版本**: v2.1
**状态**: 待审阅
**维护者**: bruce li
**更新日期**: 2026-03-29

---

## 一、设计概述

### 1.1 采集模块定位

```
SwingTrade/src/data/fetcher/     # 采集模块（代码在SwingTrade）
SwingTrade/config/                # 配置文件
/Users/bruce/workspace/trade/StockData/  # 数据存储（纯数据）
```

采集模块负责：
1. 从数据源（Tushare/AkShare）拉取数据
2. 质量校验和评分
3. 将高质量数据写入 StockData 目录
4. 生成采集日报

### 1.2 核心设计原则

| 原则 | 说明 |
|------|------|
| **幂等写入** | 同一日期数据重复采集不会产生重复 |
| **质量门槛** | 质量不达标的数据不写入 |
| **失败追溯** | 所有失败有日志，支持自动重试 |
| **日报透明** | 所有失败体现在日报中 |

---

## 二、数据源策略（v2.0 核心更新）

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
├── fetch_metadata.py         # 股票列表采集
├── quality_scorer.py        # 质量评分
├── validators/
│   ├── __init__.py
│   ├── stock_validator.py    # 股票列表验证
│   └── daily_validator.py   # 日线数据验证
├── sources/
│   ├── __init__.py
│   ├── base.py              # 数据源基类
│   ├── tushare_source.py    # Tushare 适配器（主力）
│   └── akshare_source.py    # AkShare 适配器（补充）
├── retry_handler.py         # 重试逻辑
├── report_generator.py      # 日报生成
└── exceptions.py            # 自定义异常
```

---

## 四、数据流（v2.0 优化）

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
   ├─ overall_score >= 60 → 记录警告但不阻塞
   └─ overall_score < 60 → 拒绝，标记失败

6. [幂等写入]
   └─ 按 (code, date) 去重后写入 Parquet

7. [更新索引]
   └─ 更新 SQLite daily_index 和 update_log

8. [生成日报]
   └─ 输出采集结果统计（包含所有失败详情）
```

### 4.2 复权计算（新增）

```python
def calculate_adj_close(daily_df: pd.DataFrame, adj_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算后复权价格

    复权公式：
    后复权价 = 原始价 × 复权因子

    Args:
        daily_df: 日线数据（包含 open, high, low, close）
        adj_df: 复权因子数据

    Returns:
        添加后复权列的 DataFrame
    """
    # 合并复权因子
    df = daily_df.merge(adj_df, on='date', how='left')

    # 填充缺失的复权因子（用最近的）
    df['adj_factor'] = df['adj_factor'].fillna(method='ffill')

    # 计算后复权价
    df['close_adj'] = df['close'] * df['adj_factor']
    df['open_adj'] = df['open'] * df['adj_factor']
    df['high_adj'] = df['high'] * df['adj_factor']
    df['low_adj'] = df['low'] * df['adj_factor']

    return df
```

### 4.3 质量评分详细流程

```python
def score_daily_record(record: dict, prev_record: dict = None) -> QualityScore:
    """计算单条日线数据的质量分"""

    # 1. source_consistency（跨源一致性）
    # 注：v2.0 以 Tushare 为主源，暂不使用双源验证
    source_score = 100.0  # Tushare 质量可靠，单源满分

    # 2. field_completeness（字段完整性）
    required = ["date", "open", "high", "low", "close", "volume"]
    present_count = sum(1 for f in required if record.get(f) is not None)
    completeness_score = (present_count / len(required)) * 100

    # 3. range_validity（范围合理性）
    validity_score = 100.0
    if not (0.01 <= record.get("close", 0) <= 10000):
        validity_score -= 20
    if not (record.get("high", 0) >= record.get("low", 0)):
        validity_score -= 20
    if not (record.get("high", 0) >= record.get("close", 0) >= record.get("low", 0)):
        validity_score -= 20

    # 4. historical_anomaly（历史连续性）
    if prev_record is None:
        anomaly_score = 100.0  # 无历史数据，无法比较
    else:
        change = abs(record.get("close", 0) - prev_record.get("close", 0)) / prev_record.get("close", 1)
        if change <= 0.107:  # 涨跌停范围内
            anomaly_score = 100.0
        else:
            anomaly_score = 30.0  # 超出涨跌停

    # 5. 综合分（加权平均）
    overall = (
        source_score * 0.30 +
        completeness_score * 0.20 +
        validity_score * 0.30 +
        anomaly_score * 0.20
    )

    return QualityScore(
        source_consistency=source_score,
        field_completeness=completeness_score,
        range_validity=validity_score,
        historical_anomaly=anomaly_score,
        overall=round(overall, 2)
    )
```

---

## 五、数据源适配器（v2.0 更新）

### 5.1 Tushare 适配器（主力）

```python
class TushareSource(DataSource):
    """Tushare 数据源（主力日线数据源）"""

    name = "tushare"

    def __init__(self, token: str = None):
        self.token = token or os.getenv("TUSHARE_TOKEN")
        if not self.token:
            raise ConfigurationError("TUSHARE_TOKEN not set")
        import tushare as ts
        self.pro = ts.pro_api(self.token)

    def fetch_daily(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        拉取日线数据（不复权）

        Args:
            code: 股票代码，如 "600519.SH"
            start_date: 开始日期 "YYYY-MM-DD"
            end_date: 结束日期 "YYYY-MM-DD"

        Returns:
            DataFrame，包含字段：date, open, high, low, close, volume, amount, pct_chg
        """
        ts_code = self._to_ts_code(code)
        df = self.pro.daily(
            ts_code=ts_code,
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", "")
        )
        if df is None or len(df) == 0:
            return pd.DataFrame()

        # 字段映射
        df = df.rename(columns={
            "ts_code": "code",
            "trade_date": "date",
            "vol": "volume"
        })
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df

    def fetch_adj_factor(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        拉取复权因子

        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame，包含字段：date, adj_factor
        """
        ts_code = self._to_ts_code(code)
        df = self.pro.adj_factor(
            ts_code=ts_code,
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", "")
        )
        if df is None or len(df) == 0:
            return pd.DataFrame()

        df = df.rename(columns={"trade_date": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df[["date", "adj_factor"]]

    def _to_ts_code(self, code: str) -> str:
        """转换为 Tushare 格式"""
        if code.endswith((".SH", ".SZ", ".BJ")):
            return code
        if code.startswith(("6", "5")):
            return f"{code}.SH"
        return f"{code}.SZ"
```

### 5.2 AkShare 适配器（补充）

```python
class AkShareSource(DataSource):
    """AkShare 数据源（股票列表补充源）"""

    name = "akshare"

    def __init__(self):
        import akshare as ak
        self.ak = ak

    def fetch_daily(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """拉取日线数据（备用）"""
        symbol = self._to_akshare_symbol(code)
        df = self.ak.stock_zh_a_daily(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        if df is None or len(df) == 0:
            return pd.DataFrame()
        df["code"] = code
        return df

    def fetch_stock_list(self) -> pd.DataFrame:
        """
        拉取股票列表

        从沪深两市分别获取（北交所因代理问题暂时跳过）

        Returns:
            DataFrame，包含字段：code, name, market, list_date
        """
        all_stocks = []

        # 沪市
        try:
            df_sh = self.ak.stock_info_sh_name_code()
            df_sh = df_sh.rename(columns={
                "证券代码": "code",
                "证券简称": "name",
                "上市日期": "list_date"
            })
            df_sh["market"] = "sh"
            all_stocks.append(df_sh[["code", "name", "market", "list_date"]])
        except Exception as e:
            print(f"Warning: SH stock list failed: {e}")

        # 深市
        try:
            df_sz = self.ak.stock_info_sz_name_code()
            df_sz = df_sz.rename(columns={
                "A股代码": "code",
                "A股简称": "name",
                "A股上市日期": "list_date"
            })
            df_sz["market"] = "sz"
            all_stocks.append(df_sz[["code", "name", "market", "list_date"]])
        except Exception as e:
            print(f"Warning: SZ stock list failed: {e}")

        if not all_stocks:
            raise SourceError("Failed to fetch stock list", source=self.name)

        return pd.concat(all_stocks, ignore_index=True)

    def _to_akshare_symbol(self, code: str) -> str:
        if code.startswith(("6", "5")):
            return f"sh{code}"
        elif code.startswith(("0", "1", "2", "3")):
            return f"sz{code}"
        return f"sh{code}"
```

---

## 六、采集窗口与调度

### 6.1 采集时间窗口

| 时间 | 任务 |
|------|------|
| 17:30 | 定时任务触发，开始采集 |
| 17:30-17:50 | 主力采集（Tushare 日线 + 复权因子） |
| 17:50-18:00 | 扫尾、重试失败、处理异常 |
| 18:00 | 窗口关闭，生成日报 |

**注意**：Tushare 日线数据在交易日 15:00-17:00 更新，17:30 开始确保数据已完整更新。

### 6.2 crontab 配置

```bash
# 每日 17:30 执行（工作日）
30 17 * * 1-5 /Users/bruce/workspace/trade/SwingTrade/scripts/fetch/run_daily_fetch.py
```

### 6.3 批量采集策略

由于 200 积分足够（10万次/天），采用**直接采集**策略：

```python
def fetch_all_daily(target_date: str):
    """
    采集全市场日线数据

    200积分限制：10万次/天
    全市场约5000只股票，每只需要2次调用（日线+复权因子）
    总计约1万次/天，远低于限制
    """
    # 1. 获取股票列表
    stock_list = get_stock_list()  # 从 AkShare

    # 2. 逐只采集（不限流）
    for code in stock_list:
        daily = tushare.fetch_daily(code, target_date, target_date)
        adj = tushare.fetch_adj_factor(code, target_date, target_date)

        # 3. 合并复权
        df = merge_and_calculate_adj(daily, adj)

        # 4. 写入
        write_daily(code, df)
```

---

## 七、重试机制

### 7.1 重试策略

| 失败类型 | 重试次数 | 间隔 | 处理 |
|----------|----------|------|------|
| `network` | 2次 | 0s → 30s | 等待后重试 |
| `source` | 2次 | 0s → 30s | 切换备源 |
| `quality` | 0次 | - | 拒绝，不重试 |

### 7.2 失败追溯

```python
@dataclass
class FetchResult:
    code: str
    date: str
    data: Optional[pd.DataFrame]
    source: str = "tushare"
    fetch_status: str = "pending"   # pending / success / failed
    write_status: str = "pending"   # pending / success / rejected
    fail_type: Optional[str] = None  # None / network / quality / source
    fail_reason: Optional[str] = None
    quality_score: Optional[float] = None
    attempts: int = 0
```

---

## 八、写入器接口

### 8.1 Parquet Schema（日线数据）

```python
{
    "date": "datetime64",      # 交易日期
    "code": "str",             # 股票代码
    "open": "float32",         # 开盘价（后复权）
    "high": "float32",         # 最高价（后复权）
    "low": "float32",          # 最低价（后复权）
    "close": "float32",       # 收盘价（后复权）
    "volume": "int64",         # 成交量
    "amount": "float64",       # 成交额
    "pct_chg": "float32",     # 涨跌幅
    "adj_factor": "float32",  # 复权因子
    "quality_score": "float32" # 质量评分
}
```

### 8.2 幂等写入

```python
def write_daily(code: str, df: pd.DataFrame):
    """幂等写入：基于 (code, date) 去重"""
    target_file = f"{STOCKDATA_ROOT}/raw/daily/{code}.parquet"

    if os.path.exists(target_file):
        existing = pd.read_parquet(target_file)
        # 过滤已存在的日期
        new_df = df[~df["date"].isin(existing["date"])]
        if len(new_df) == 0:
            return  # 已存在，跳过
        df = pd.concat([existing, new_df]).drop_duplicates(["date"])
        df = df.sort_values("date")

    # 写入临时文件 + 原子替换
    temp_file = f"{target_file}.tmp"
    df.to_parquet(temp_file, engine="pyarrow", compression="snappy")
    os.replace(temp_file, target_file)
```

---

## 九、股票列表采集与验证

### 9.1 采集策略

| 频率 | 数据源 | 说明 |
|------|--------|------|
| 每日首次采集前 | AkShare | 更新股票列表 |
| 每周 | AkShare | 全量同步 |
| 每月 | Tushare | 交叉验证 |

### 9.2 验证规则

| 验证项 | 规则 | 不通过处理 |
|--------|------|-----------|
| 代码格式 | 6位数字 | 拒绝 |
| 市场标识 | sh/sz/bj | 拒绝 |
| 前缀匹配 | 前缀与市场对应 | 拒绝 |
| 名称规范 | 无乱码、长度2-20 | 拒绝 |
| 日期逻辑 | 上市日期 ≤ 今天 | 拒绝 |

---

## 十、日报生成

### 10.1 日报格式

```json
{
  "date": "2026-03-28",
  "start_time": "17:30:00",
  "end_time": "16:23:45",
  "duration_seconds": 1425,
  "summary": {
    "total_stocks": 4587,
    "success_count": 4580,
    "quality_rejected_count": 2,
    "network_failed_count": 5,
    "success_rate": 0.9985
  },
  "successes": [...],
  "quality_rejected": [...],
  "network_failed": [...],
  "warnings": [...]
}
```

### 10.2 失败全量体现

**日报必须包含所有失败详情**，包括：
- 网络失败：code, reason, attempts
- 质量拒绝：code, score, fail_reason
- 重试失败：code, reason, attempts

---

## 十一、SQLite 表结构

### 11.1 stocks 表

```sql
CREATE TABLE stocks (
    code TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    market TEXT NOT NULL,          -- 'sh' / 'sz'
    list_date TEXT,                 -- YYYY-MM-DD
    is_active INTEGER DEFAULT 1,    -- 1=活跃
    source TEXT,                    -- 'akshare'
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
    fail_type TEXT,               -- 'network' / 'quality' / 'source'
    fail_reason TEXT,
    write_status TEXT,            -- 'success' / 'rejected'
    quality_score REAL,
    source TEXT,
    attempts INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(code, update_date, data_type)
);
```

---

## 十二、环境变量

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `TUSHARE_TOKEN` | Tushare API Token | `xxx` |
| `STOCKDATA_ROOT` | StockData 根目录 | `/Users/bruce/workspace/trade/StockData` |

---

## 十三、目录结构

```
SwingTrade/
├── src/data/fetcher/
│   ├── __init__.py
│   ├── fetch_daily.py           # 日线采集
│   ├── fetch_metadata.py         # 股票列表采集
│   ├── quality_scorer.py        # 质量评分
│   ├── validators/
│   │   ├── stock_validator.py
│   │   └── daily_validator.py
│   ├── sources/
│   │   ├── tushare_source.py   # Tushare 主力
│   │   └── akshare_source.py   # AkShare 补充
│   ├── retry_handler.py
│   ├── report_generator.py
│   └── exceptions.py
├── config/
│   └── data_quality.yaml
├── scripts/fetch/
│   └── run_daily_fetch.py
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

## 十四、验收标准

| 验收项 | 标准 | 验证方法 |
|--------|------|---------|
| 日线采集 | 全市场成功率 ≥ 99.5% | 日报统计 |
| 复权数据 | adj_factor 正确合并 | 抽样检查 |
| 质量评分 | 分数 < 60 拒绝写入 | 日报验证 |
| 失败追溯 | 所有失败在日报体现 | 日报检查 |
| 幂等写入 | 重复执行无重复数据 | 测试验证 |

---

**文档版本**: v2.1
**最后更新**: 2026-03-29
**更新内容**:
- v2.1: 修复7个问题：source_consistency=95、限流保护、深市失败告警、复权因子缺失处理、数据格式验证、采集窗口17:30开始
- v2.0: 根据 200 积分优化数据源策略，Tushare 专注日线+复权，AkShare 提供股票列表
