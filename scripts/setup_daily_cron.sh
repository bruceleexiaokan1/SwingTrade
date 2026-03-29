#!/bin/bash
# 日线数据定时更新 crontab 设置
#
# 使用方式:
#   ./scripts/setup_daily_cron.sh install   # 安装定时任务
#   ./scripts/setup_daily_cron.sh remove    # 移除定时任务
#   ./scripts/setup_daily_cron.sh show      # 显示当前定时任务
#
# 默认设置:
#   每个交易日下午 16:00 执行增量更新
#   每个交易日上午 9:00 获取新增股票

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 增量更新任务 (每个交易日下午4点)
INCREMENTAL_CMD="cd $PROJECT_ROOT && python3 scripts/fetch_daily_incremental.py >> $LOG_DIR/daily_fetch.log 2>&1"

# 获取新增股票任务 (每个交易日上午9点)
NEW_STOCKS_CMD="cd $PROJECT_ROOT && python3 scripts/fetch_daily_incremental.py --new >> $LOG_DIR/new_stocks.log 2>&1"

show_cron() {
    echo "当前定时任务:"
    echo "================================"
    crontab -l 2>/dev/null || echo "(无定时任务)"
}

install_cron() {
    echo "安装日线数据定时更新任务..."
    echo ""

    # 读取当前 crontab
    CURRENT=$(crontab -l 2>/dev/null || true)

    # 移除旧的任务(如果有)
    NEW_CRON=$(echo "$CURRENT" | grep -v "fetch_daily_incremental")

    # 添加新的任务
    # 每个交易日下午 16:00 执行增量更新 (周一至周五)
    NEW_CRON="$NEW_CRON
# 日线数据增量更新 - 每个交易日下午4点
0 16 * * 1-5 $INCREMENTAL_CMD
"
    # 每个交易日上午 9:00 获取新增股票 (周一至周五)
    NEW_CRON="$NEW_CRON
# 获取新增股票 - 每个交易日上午9点
0 9 * * 1-5 $NEW_STOCKS_CMD
"

    echo "$NEW_CRON" | crontab -

    echo "安装完成!"
    echo ""
    show_cron
}

remove_cron() {
    echo "移除定时任务..."
    CURRENT=$(crontab -l 2>/dev/null || true)
    NEW_CRON=$(echo "$CURRENT" | grep -v "fetch_daily_incremental")
    echo "$NEW_CRON" | crontab -
    echo "已移除!"
}

case "${1:-show}" in
    install)
        install_cron
        ;;
    remove)
        remove_cron
        ;;
    show)
        show_cron
        ;;
    *)
        echo "用法: $0 {install|remove|show}"
        exit 1
        ;;
esac
