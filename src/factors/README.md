# 因子库架构规范

**版本**: v1.0
**创建**: 2026-03-29
**状态**: 设计规范阶段

---

## 一、整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        因子库 (Factor Library)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │  因子基类    │ -> │  因子注册表   │ -> │  批量计算    │     │
│  │ FactorBase   │    │  Registry    │    │ BatchCompute │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│         │                                       │              │
│         ▼                                       ▼              │
│  ┌──────────────┐                      ┌──────────────┐       │
│  │  因子元数据  │                      │  因子计算结果  │       │
│  │  Metadata    │                      │   DataFrame   │       │
│  └──────────────┘                      └──────────────┘       │
│                                                   │              │
│                                                   ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │  清洗处理    │ -> │  评估框架     │ -> │  因子存储    │     │
│  │  Processor   │    │  Evaluation  │    │   Storage    │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、数据流

```
原始数据 (Parquet/SQL)
        │
        ▼
┌─────────────────┐
│  输入数据验证   │  ← 验证必要字段: date, code, open, high, low, close, volume, amount
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   因子计算      │  ← 每个因子继承FactorBase，实现calculate()方法
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   因子处理      │  ← 清洗: fillna → winsorize → standardize → neutralize
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   因子存储      │  ← 因子宽表: date, code, factor1, factor2, ...
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   因子评估      │  ← IC/IR计算, 分组回测, 单调性检验
└─────────────────┘
```

---

## 三、目录结构

```
SwingTrade/src/factors/
│
├── README.md              # 本文档
├── __init__.py            # 模块导出
│
├── factor_base.py         # 因子基类定义
├── registry.py           # 因子注册表
├── exceptions.py          # 自定义异常
│
├── price_volume/          # 价量因子
│   ├── __init__.py
│   ├── momentum.py        # 动量因子: ret_3m, ret_6m, ret_12m, rs_120
│   ├── volatility.py      # 波动率因子: vol_20, atr_14_pct, beta_60
│   ├── turnover.py        # 换手率因子: turnover, turnover_ma20
│   └── amount.py          # 成交额因子: amount, amount_std20
│
├── flow/                 # 资金流因子
│   ├── __init__.py
│   ├── fund_flow.py      # 主力资金因子: fund_flow_main, fund_flow_big
│   └── north_flow.py     # 北向资金因子: north_hold_chg, north_hold_ratio
│
├── valuation/            # 估值因子
│   ├── __init__.py
│   ├── pe.py            # 市盈率因子
│   ├── pb.py            # 市净率因子
│   └── composite.py      # 综合估值
│
├── quality/             # 质量因子
│   ├── __init__.py
│   ├── roe.py           # ROE因子
│   └── growth.py        # 成长因子
│
├── evaluation/          # 评估框架
│   ├── __init__.py
│   ├── ic_ir.py         # IC/IR计算
│   ├── backtest.py      # 分组回测
│   └── report.py        # 评估报告
│
└── utils/              # 工具函数
    ├── __init__.py
    └── processing.py     # 清洗处理: fillna, winsorize, standardize, neutralize
```

---

## 四、因子分类体系

### 4.1 因子类别

| 类别代码 | 类别名称 | 说明 |
|----------|----------|------|
| `momentum` | 动量因子 | 价格动量、相对强度 |
| `volatility` | 波动率因子 | 历史波动、Beta、ATR |
| `turnover` | 换手率因子 | 换手率及其衍生 |
| `flow` | 资金流因子 | 主力资金、北向资金 |
| `valuation` | 估值因子 | PE、PB、PS |
| `quality` | 质量因子 | ROE、ROA、成长性 |
| `size` | 规模因子 | 市值、对数市值 |
| `industry` | 行业因子 | 行业哑变量 |

### 4.2 因子命名规范

```
{因子类别}_{计算周期}_{说明}

示例:
  ret_3m         - 3个月收益率 (动量)
  vol_20         - 20日波动率 (波动率)
  turnover       - 换手率 (换手率)
  rs_120         - 120日相对强度 (动量)
  fund_flow_main - 主力资金净流入占比 (资金流)
  north_hold_chg - 北向持股变化 (资金流)
  pe_ttm         - TTM市盈率 (估值)
  roe_q          - 季度ROE (质量)
```

---

## 五、输入数据格式

### 5.1 标准日线数据

```python
# 输入DataFrame格式
data = pd.DataFrame({
    'date': '2026-03-28',        # 日期 (str)
    'code': '600519',             # 股票代码 (str)
    'open': 1800.0,               # 开盘价 (float)
    'high': 1820.0,              # 最高价 (float)
    'low': 1790.0,                # 最低价 (float)
    'close': 1810.0,              # 收盘价 (float)
    'volume': 3000000,            # 成交量 (int)
    'amount': 5400000000.0,       # 成交额 (float)
    'pct_chg': 1.5,               # 涨跌幅 % (float)
    'outstanding_share': 1000000000,  # 流通股本 (int)
})
```

### 5.2 指数数据 (用于Beta、RS计算)

```python
index_data = pd.DataFrame({
    'date': '2026-03-28',
    'code': '000300.SH',          # 指数代码
    'close': 3800.0,
})
```

---

## 六、输出数据格式

### 6.1 因子值DataFrame

```python
# 输出DataFrame格式
factor_df = pd.DataFrame({
    'date': '2026-03-28',        # 日期
    'code': '600519',             # 股票代码
    'factor_value': 0.05,         # 因子值 (标准化后为Z-score)
})
```

### 6.2 因子宽表

```python
# 批量计算后的宽表
factor_wide = pd.DataFrame({
    'date': '2026-03-28',
    'code': '600519',
    'ret_3m': 0.05,
    'vol_20': 0.02,
    'turnover': 1.5,
    'rs_120': 0.03,
    # ... 更多因子
})
```

---

## 七、因子基类接口

```python
class FactorBase(ABC):
    """因子基类"""

    # 类属性
    name: str           # 因子名称 (如 "ret_3m")
    category: str      # 因子类别 (如 "momentum")
    description: str   # 因子描述

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算因子

        Args:
            data: 输入数据 (标准日线格式)

        Returns:
            DataFrame: date, code, factor_value
        """
        pass

    def get_metadata(self) -> dict:
        """返回因子元数据"""
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
        }

    def validate_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据"""
        required_cols = ['date', 'code', 'close']
        return all(col in data.columns for col in required_cols)
```

---

## 八、清洗处理流水线

```
原始因子值
    │
    ▼
┌──────────────┐
│  1. fillna   │  行业均值填充
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 2. winsorize │  MAD去极值 (±3σ截断)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 3. standardize│ Z-score标准化
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 4. neutralize │  中性化 (市值+行业回归残差)
└──────┬───────┘
       │
       ▼
   干净因子值
```

---

## 九、评估指标

### 9.1 IC (Information Coefficient)

```
IC = Spearman(factor_value, forward_return)

判定标准:
  |IC| > 0.02  → 有效因子
  |IC| > 0.05  → 强有效因子
  |IC| < 0.01  → 无效因子
```

### 9.2 IR (Information Ratio)

```
IR = IC均值 / IC标准差

判定标准:
  IR > 0.5   → 稳定有效
  IR > 1.0   → 高度稳定
```

### 9.3 分组回测

```
按因子值分10组 (Q1-Q10)
检验: Q1 < Q2 < ... < Q10 (单调递增)
多空收益: Q10 - Q1
```

---

## 十、版本历史

| 版本 | 日期 | 修改内容 |
|------|------|----------|
| v1.0 | 2026-03-29 | 初始版本 |

---

## 十一、与回测系统集成

### 核心原则

> **回测不计算因子，只调用因子**

- 离线：提前把所有因子算好，存成因子宽表
- 在线：回测引擎根据日期，直接从表中提取当天的因子值

### 两种集成模式

#### 模式A：向量化回测（适合多因子选股）

```
因子宽表 → 信号生成(加权合成) → 仓位分配(TopN) → 回测模拟 → 绩效归因
```

```python
# 1. 加载因子库
factor_db = pd.read_parquet('factor_library.parquet')

# 2. 合成信号
factor_db['Score'] = factor_db['ret_3m'] * 0.5 + factor_db['vol_20'] * 0.5

# 3. 回测循环（向量化思维）
for date in all_trade_dates:
    prev_date = get_previous_trade_date(date)
    today_signals = factor_db.loc[prev_date]
    top_stocks = today_signals.nlargest(10, 'Score').index.tolist()
    # ... 换仓逻辑
```

#### 模式B：事件驱动回测（适合复杂风控）

将因子库作为外部数据源传入回测框架。

### 时间对齐规则

| 因子类型 | 规则 | 对齐方法 |
|----------|------|----------|
| 行情因子 | T日收盘后生成，用于T+1决策 | 因子索引=Trade_Date=T |
| 财务因子 | 财报发布后才能使用 | 因子索引=Announce_Date |

### 关键避坑

1. **存活偏差**：因子库必须包含已退市股票
2. **停牌处理**：因子库增加Status列，过滤停牌股票
3. **价格对齐**：用T日因子 -> T+1日开盘价成交

### 推荐实践

先用Pandas向量化回测跑通30个因子流程，再考虑复杂框架。

---

## 十二、数据时间戳要求

每个因子值必须打上"生效日期"标签：

```python
# 因子宽表结构
factor_wide = pd.DataFrame({
    'trade_date': '2026-03-28',     # 交易日
    'code': '600519',
    'factor_value': 0.05,
    'announce_date': None,           # 公告日（财务因子用）
    'status': '正常',                 # 交易状态
})
```

回测引擎严禁使用"未来数据"，所有决策必须在生效日期之后。
