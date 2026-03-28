#!/bin/bash
# StockData Cron 配置
# 使用方法: bash crontab_setup.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 环境变量文件
ENV_FILE="$PROJECT_ROOT/.env"

# 加载环境变量
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

# 设置默认值
export STOCKDATA_ROOT="${STOCKDATA_ROOT:-/Users/bruce/workspace/trade/StockData}"
export BACKUP_ROOT="${BACKUP_ROOT:-/Users/bruce/backup/StockData}"

# 创建日志目录
mkdir -p "$STOCKDATA_ROOT/logs"

# Cron 条目
CRON_ENTRIES="# StockData Cron 配置
# 采集: 每周一到周五 16:35 执行日线采集
35 16 * * 1-5 cd \"$PROJECT_ROOT\" && /usr/bin/python3 scripts/fetch/run_daily_fetch.py >> \"$STOCKDATA_ROOT/logs/fetch_daily.log\" 2>&1

# 温数据汇总: 采集完成后 16:40
40 16 * * 1-5 cd \"$PROJECT_ROOT\" && /usr/bin/python3 scripts/maintenance/warm_summary.py >> \"$STOCKDATA_ROOT/logs/warm_summary.log\" 2>&1

# 备份: 每天 17:00
0 17 * * * cd \"$PROJECT_ROOT\" && /usr/bin/python3 scripts/maintenance/backup.py >> \"$STOCKDATA_ROOT/logs/backup.log\" 2>&1

# 健康检查: 每周一到周五 09:00 发送日报
0 9 * * 1-5 cd \"$PROJECT_ROOT\" && /usr/bin/python3 scripts/monitor/health_check.py >> \"$STOCKDATA_ROOT/logs/health_check.log\" 2>&1

# 健康检查: 每周一到周五 16:30 检查当日采集
30 16 * * 1-5 cd \"$PROJECT_ROOT\" && /usr/bin/python3 scripts/monitor/health_check.py >> \"$STOCKDATA_ROOT/logs/health_check.log\" 2>&1
"

# 获取当前 crontab，移除旧的 StockData 条目，添加新的
(crontab -l 2>/dev/null | grep -v "StockData Cron 配置" | grep -v "scripts/fetch/run_daily_fetch.py" | grep -v "scripts/maintenance/warm_summary.py" | grep -v "scripts/maintenance/backup.py" | grep -v "scripts/monitor/health_check.py"; echo "$CRON_ENTRIES") | crontab -

echo "Cron 配置已添加"
echo ""
echo "当前 Crontab:"
crontab -l
