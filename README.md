# SwingTrade

个人波段量化交易系统

## 项目概述

专注于A股波段交易的量化投资系统，支持数据采集、策略回测和实盘交易。

## 技术栈

- **数据源**: tushare、akshare、baostock
- **核心库**: pandas、numpy
- **存储**: SQLite (索引) + Parquet (行情数据)
- **分析**: Jupyter、matplotlib、seaborn

## 项目结构

```
SwingTrade/
├── src/                    # 源代码
│   └── data/              # 数据访问层
├── scripts/                # 脚本
│   ├── fetch/             # 数据采集
│   ├── maintenance/       # 维护任务
│   ├── monitor/           # 监控告警
│   └── utils/             # 工具类
├── config/                 # 配置文件
├── tests/                 # 测试
├── docs/                   # 设计文档
└── requirements.txt

StockData/                  # 数据仓库（独立仓库）
├── raw/daily/              # 日线原始数据 (Parquet)
├── sqlite/                 # SQLite 索引库
├── cache/                  # 热数据缓存
├── status/                 # 状态文件
└── logs/                   # 日志
```

## 数据存储

数据存储在独立的 `StockData` 仓库，位于 `/Users/bruce/workspace/trade/StockData`。

### 环境变量配置

```bash
export STOCKDATA_ROOT="/Users/bruce/workspace/trade/StockData"
export TUSHARE_TOKEN="your_token_here"
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
export STOCKDATA_ROOT="/Users/bruce/workspace/trade/StockData"
export TUSHARE_TOKEN="your_token_here"
```

### 3. 初始化数据库

```bash
python scripts/utils/init_db.py
```

### 4. 采集日线数据

```bash
python scripts/fetch/fetch_daily.py --date 2026-03-28
```

## 开发进度

- [x] 项目初始化
- [x] 数据存储架构
- [ ] 数据采集模块
- [ ] 策略框架
- [ ] 回测系统
- [ ] 实盘接口

## 文档

- [设计文档](docs/design.md) - 数据存储完整设计方案
