"""回测报告生成

生成 HTML 回测报告：
- 绩效摘要仪表盘
- 权益曲线图
- 回撤图
- 月度收益统计
- 交易记录表
"""

import pandas as pd
from typing import Optional

from .models import BacktestResult


class BacktestReporter:
    """
    回测报告生成器

    生成包含以下内容的 HTML 报告：
    1. 绩效摘要仪表盘
    2. 权益曲线图
    3. 回撤图
    4. 月度收益统计
    5. 交易记录表
    """

    def generate_html(
        self,
        result: BacktestResult,
        output_path: Optional[str] = None
    ) -> str:
        """
        生成 HTML 回测报告

        Args:
            result: 回测结果
            output_path: 输出文件路径（可选）

        Returns:
            str: HTML 内容
        """
        html = self._generate_html_header()
        html += self._generate_dashboard(result)
        html += self._generate_equity_chart(result)
        html += self._generate_drawdown_chart(result)
        html += self._generate_trades_table(result)
        html += self._generate_html_footer()

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)

        return html

    def _generate_html_header(self) -> str:
        """生成 HTML 头部"""
        return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>波段交易回测报告</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: #f5f7fa;
            color: #333;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #1a1a2e;
            margin-bottom: 20px;
            font-size: 28px;
        }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .metric-card h3 {
            color: #666;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        .metric-card .value {
            font-size: 24px;
            font-weight: bold;
            color: #1a1a2e;
        }
        .metric-card .value.positive {
            color: #2ecc71;
        }
        .metric-card .value.negative {
            color: #e74c3c;
        }
        .chart-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .chart-container h2 {
            color: #1a1a2e;
            margin-bottom: 15px;
            font-size: 18px;
        }
        .chart-wrapper {
            position: relative;
            height: 300px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        th {
            background: #f8f9fa;
            color: #666;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
        }
        tr:hover {
            background: #f8f9fa;
        }
        .trade-profit {
            color: #2ecc71;
        }
        .trade-loss {
            color: #e74c3c;
        }
        .summary {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .summary pre {
            font-family: "SF Mono", Monaco, monospace;
            font-size: 14px;
            line-height: 1.6;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>波段交易回测报告</h1>
"""

    def _generate_dashboard(self, result: BacktestResult) -> str:
        """生成绩效仪表盘"""
        # 颜色判断
        def color_class(val, threshold, reverse=False):
            if reverse:
                return "positive" if val < threshold else "negative"
            return "positive" if val > threshold else ""

        return f"""
        <div class="dashboard">
            <div class="metric-card">
                <h3>总收益率</h3>
                <div class="value {color_class(result.total_return, 0)}">{result.total_return:.2%}</div>
            </div>
            <div class="metric-card">
                <h3>年化收益率</h3>
                <div class="value {color_class(result.annualized_return, 0)}">{result.annualized_return:.2%}</div>
            </div>
            <div class="metric-card">
                <h3>夏普比率</h3>
                <div class="value {color_class(result.sharpe_ratio, 1.5)}">{result.sharpe_ratio:.2f}</div>
            </div>
            <div class="metric-card">
                <h3>最大回撤</h3>
                <div class="value {color_class(result.max_drawdown, 0.2, reverse=True)}">{result.max_drawdown:.2%}</div>
            </div>
            <div class="metric-card">
                <h3>卡玛比率</h3>
                <div class="value {color_class(result.calmar_ratio, 2.0)}">{result.calmar_ratio:.2f}</div>
            </div>
            <div class="metric-card">
                <h3>胜率</h3>
                <div class="value">{result.win_rate:.2%}</div>
            </div>
            <div class="metric-card">
                <h3>盈亏比</h3>
                <div class="value {color_class(result.profit_factor, 1.5)}">{result.profit_factor:.2f}</div>
            </div>
            <div class="metric-card">
                <h3>交易次数</h3>
                <div class="value">{result.total_trades}</div>
            </div>
            <div class="metric-card">
                <h3>年均交易</h3>
                <div class="value">{result.trades_per_year:.1f} 次</div>
            </div>
            <div class="metric-card">
                <h3>平均持仓</h3>
                <div class="value">{result.avg_holding_days:.1f} 天</div>
            </div>
        </div>
        """

    def _generate_equity_chart(self, result: BacktestResult) -> str:
        """生成权益曲线图"""
        if result.equity_curve is None or result.equity_curve.empty:
            return """
            <div class="chart-container">
                <h2>权益曲线</h2>
                <p>无数据</p>
            </div>
            """

        dates = result.equity_curve["date"].tolist()
        equity = result.equity_curve["equity"].tolist()

        dates_json = str(dates).replace("'", '"')
        equity_json = str(equity)

        return f"""
        <div class="chart-container">
            <h2>权益曲线</h2>
            <div class="chart-wrapper">
                <canvas id="equityChart"></canvas>
            </div>
            <script>
                new Chart(document.getElementById('equityChart'), {{
                    type: 'line',
                    data: {{
                        labels: {dates_json},
                        datasets: [{{
                            label: '权益',
                            data: {equity_json},
                            borderColor: '#3498db',
                            backgroundColor: 'rgba(52, 152, 219, 0.1)',
                            fill: true,
                            tension: 0.1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{ display: false }}
                        }},
                        scales: {{
                            x: {{
                                display: true,
                                ticks: {{ maxTicksLimit: 10 }}
                            }},
                            y: {{
                                display: true
                            }}
                        }}
                    }}
                }});
            </script>
        </div>
        """

    def _generate_drawdown_chart(self, result: BacktestResult) -> str:
        """生成回撤图"""
        if result.equity_curve is None or result.equity_curve.empty:
            return """
            <div class="chart-container">
                <h2>回撤曲线</h2>
                <p>无数据</p>
            </div>
            """

        # 计算回撤
        equity = result.equity_curve["equity"].values
        peak = equity[0]
        drawdowns = []

        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            drawdowns.append(dd * 100)  # 转换为百分比

        dates = result.equity_curve["date"].tolist()
        dates_json = str(dates).replace("'", '"')
        drawdowns_json = str(drawdowns)

        return f"""
        <div class="chart-container">
            <h2>回撤曲线</h2>
            <div class="chart-wrapper">
                <canvas id="drawdownChart"></canvas>
            </div>
            <script>
                new Chart(document.getElementById('drawdownChart'), {{
                    type: 'line',
                    data: {{
                        labels: {dates_json},
                        datasets: [{{
                            label: '回撤 (%)',
                            data: {drawdowns_json},
                            borderColor: '#e74c3c',
                            backgroundColor: 'rgba(231, 76, 60, 0.1)',
                            fill: true,
                            tension: 0.1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{ display: false }}
                        }},
                        scales: {{
                            x: {{
                                display: true,
                                ticks: {{ maxTicksLimit: 10 }}
                            }},
                            y: {{
                                display: true,
                                ticks: {{ callback: function(v) {{ return v.toFixed(1) + '%'; }} }}
                            }}
                        }}
                    }}
                }});
            </script>
        </div>
        """

    def _generate_trades_table(self, result: BacktestResult) -> str:
        """生成交易记录表"""
        if not result.trades:
            return """
            <div class="chart-container">
                <h2>交易记录</h2>
                <p>无交易记录</p>
            </div>
            """

        # 生成交易行
        rows = []
        for trade in result.trades[:50]:  # 最多显示50条
            pnl_class = "trade-profit" if trade.pnl > 0 else "trade-loss"
            pnl_str = f"{trade.pnl:,.0f}" if trade.pnl != 0 else "-"
            if trade.pnl > 0:
                pnl_str = f"+{pnl_str}"
            exit_price_str = f"{trade.exit_price:.2f}" if trade.exit_price > 0 else "-"
            rows.append(f"""
                <tr>
                    <td>{trade.date}</td>
                    <td>{trade.code}</td>
                    <td>{trade.signal_type}</td>
                    <td>{trade.entry_price:.2f}</td>
                    <td>{exit_price_str}</td>
                    <td>{trade.shares}</td>
                    <td class="{pnl_class}">{pnl_str}</td>
                    <td>{trade.signal_reason[:30] if trade.signal_reason else '-'}</td>
                </tr>
            """)

        rows_html = "\n".join(rows)

        return f"""
        <div class="chart-container">
            <h2>交易记录 (前50条)</h2>
            <table>
                <thead>
                    <tr>
                        <th>日期</th>
                        <th>股票</th>
                        <th>信号</th>
                        <th>入场价</th>
                        <th>出场价</th>
                        <th>数量</th>
                        <th>盈亏</th>
                        <th>原因</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """

    def _generate_html_footer(self) -> str:
        """生成 HTML 尾部"""
        return """
    </div>
</body>
</html>
"""
