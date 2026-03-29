"""策略参数定义

集中管理波段交易策略的所有参数：
- 指标参数（MA、MACD、RSI、布林带、ATR）
- 入场参数（置信度阈值、盈亏比要求）
- 出场参数（止损倍数、追踪止损、盈利目标）
- 风控参数（仓位管理、最大亏损限制）
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StrategyParams:
    """
    波段策略参数

    用于配置 SwingBacktester 和 SwingSignals 的所有策略参数。
    包含三大类：
    1. 指标参数：定义技术指标的周期和阈值
    2. 入场/出场参数：定义交易信号触发条件
    3. 风控参数：定义仓位管理和风险控制
    """

    # ========== 指标参数 ==========

    # 均线参数
    ma_short: int = 20        # 短期均线周期（用于趋势判断）
    ma_long: int = 60         # 长期均线周期（用于趋势判断）

    # MACD 参数
    macd_fast: int = 12       # MACD 快线周期
    macd_slow: int = 26       # MACD 慢线周期
    macd_signal: int = 9       # MACD 信号线周期

    # RSI 参数
    rsi_period: int = 14      # RSI 周期
    rsi_oversold: int = 35     # RSI 超卖阈值（用于入场）
    rsi_overbought: int = 80  # RSI 超买阈值（用于出场）

    # 布林带参数
    bollinger_period: int = 20   # 布林带周期
    bollinger_std: float = 2.0   # 布林带标准差倍数

    # ATR 参数
    atr_period: int = 14       # ATR 周期

    # ADX 参数
    adx_period: int = 14       # ADX 周期

    # 成交量参数
    volume_period: int = 20    # 成交量均线周期
    volume_surge_threshold: float = 1.5  # 放量阈值（相对于均量）

    # ========== 入场参数 ==========

    entry_confidence_threshold: float = 0.5  # 入场置信度阈值（0.0~1.0）
    min_profit_loss_ratio: float = 3.0        # 最小盈亏比要求（知识库：中长线 >= 1:3）

    # ========== 出场参数 ==========

    atr_stop_multiplier: float = 2.0          # ATR止损倍数（入场价 - N*ATR）
    atr_trailing_multiplier: float = 3.0      # ATR追踪止损倍数（最高价 - N*ATR）
    profit_target_multiplier: float = 3.0     # 盈利目标倍数（相对于ATR）

    # ========== 风控参数 ==========

    # 仓位管理
    position_size_type: str = "fixed"   # 仓位管理类型："fixed" / "percent"
    fixed_position_value: float = 100_000.0  # 固定仓位金额
    max_position_pct: float = 0.20      # 最大仓位占比

    # 风险控制
    trial_position_pct: float = 0.10    # 试探仓位比例（首笔建仓使用较小仓位）
    max_single_loss_pct: float = 0.02   # 单笔最大亏损限制（相对于总资金）
    max_open_positions: int = 5         # 最大同时持仓数
    atr_circuit_breaker: float = 3.0    # ATR熔断倍数（当前ATR超过入场时N倍时禁止开仓）

    # 交易成本
    commission_rate: float = 0.0003      # 佣金率（默认0.03%）
    stamp_tax: float = 0.0001           # 印花税率（默认0.01%，仅卖出时）

    # 滑点
    slippage_base: float = 0.001        # 基准滑点（默认0.1%）

    # 其他
    min_trade_amount: float = 1000.0    # 最小交易金额

    def to_dict(self) -> dict:
        """转换为字典（用于日志和调试）"""
        return {
            "indicator": {
                "ma_short": self.ma_short,
                "ma_long": self.ma_long,
                "macd": f"({self.macd_fast}, {self.macd_slow}, {self.macd_signal})",
                "rsi": f"{self.rsi_period} ({self.rsi_oversold}/{self.rsi_overbought})",
                "bollinger": f"{self.bollinger_period} ({self.bollinger_std}σ)",
                "atr": self.atr_period,
                "adx": self.adx_period,
            },
            "entry": {
                "confidence_threshold": self.entry_confidence_threshold,
                "min_profit_loss_ratio": self.min_profit_loss_ratio,
            },
            "exit": {
                "atr_stop": self.atr_stop_multiplier,
                "atr_trailing": self.atr_trailing_multiplier,
                "profit_target": self.profit_target_multiplier,
            },
            "risk": {
                "trial_position_pct": f"{self.trial_position_pct:.0%}",
                "max_single_loss_pct": f"{self.max_single_loss_pct:.0%}",
                "max_open_positions": self.max_open_positions,
                "atr_circuit_breaker": self.atr_circuit_breaker,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyParams":
        """从字典创建（用于加载配置）"""
        return cls(**data)

    @classmethod
    def default(cls) -> "StrategyParams":
        """创建默认参数"""
        return cls()

    @classmethod
    def aggressive(cls) -> "StrategyParams":
        """激进参数（更适合短线）"""
        return cls(
            ma_short=10,
            ma_long=30,
            rsi_period=6,
            rsi_oversold=30,
            atr_stop_multiplier=1.5,
            atr_trailing_multiplier=2.5,
            max_open_positions=8,
        )

    @classmethod
    def conservative(cls) -> "StrategyParams":
        """保守参数（更适合长线）"""
        return cls(
            ma_short=30,
            ma_long=120,
            rsi_period=21,
            rsi_oversold=40,
            atr_stop_multiplier=2.5,
            atr_trailing_multiplier=4.0,
            max_open_positions=3,
        )
