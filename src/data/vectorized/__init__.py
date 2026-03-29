"""向量化计算模块（共享）

这些模块可在回测和实时交易模块之间共享：
- VectorizedIndicators: 向量化指标计算
- VectorizedSignals: 向量化信号检测
- VectorizedMultiCycle: 向量化多周期共振

使用方式：
    from src.data.vectorized import VectorizedIndicators, VectorizedSignals
"""

from .indicators import VectorizedIndicators, IndicatorConfig
from .signals import VectorizedSignals, SignalConfig
from .multi_cycle import VectorizedMultiCycle, MultiCycleConfig

__all__ = [
    'VectorizedIndicators',
    'IndicatorConfig',
    'VectorizedSignals',
    'SignalConfig',
    'VectorizedMultiCycle',
    'MultiCycleConfig',
]
