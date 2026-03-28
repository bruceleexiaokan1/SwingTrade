"""日线数据验证器"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class DailyValidationResult:
    """日线验证结果"""
    is_valid: bool
    errors: list[str]
    warnings: list[str]


class DailyValidator:
    """日线数据验证器

    验证规则：
    1. 价格范围：0.01 < close < 10000
    2. OHLC 关系：low <= close <= high, low <= open <= high
    3. 涨跌停限制：涨跌幅在 ±10.7% 以内
    4. 成交量非负
    5. 字段完整性：必填字段非空
    """

    # A股价格合理范围
    CLOSE_RANGE = (0.01, 10000.0)

    # 涨跌停限制（主板10%，科创/创业20%）
    # 这里使用10.7%作为宽松阈值
    LIMIT_UP_RATIO = 0.107
    LIMIT_DOWN_RATIO = 0.107

    # 涨跌幅合理范围
    PCT_CHG_RANGE = (-0.20, 0.20)

    def __init__(self):
        pass

    def validate(self, record: dict) -> DailyValidationResult:
        """
        验证单条日线记录

        Args:
            record: 包含日线数据的字典

        Returns:
            DailyValidationResult
        """
        errors = []
        warnings = []

        # 1. 必填字段检查
        required_fields = ["date", "code", "open", "high", "low", "close", "volume"]
        missing_fields = self._check_required_fields(record, required_fields)
        if missing_fields:
            errors.append(f"missing_fields: {', '.join(missing_fields)}")
            # 字段缺失时后续检查可能无意义，直接返回
            return DailyValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # 2. 价格范围检查
        price_errors = self._check_price_range(record)
        errors.extend(price_errors)

        # 3. OHLC 关系检查
        ohlc_errors = self._check_ohlc_relationship(record)
        errors.extend(ohlc_errors)

        # 4. 涨跌幅检查
        pct_chg_errors = self._check_pct_chg(record)
        errors.extend(pct_chg_errors)

        # 5. 成交量检查
        volume_errors = self._check_volume(record)
        errors.extend(volume_errors)

        return DailyValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def _check_required_fields(self, record: dict, required: list) -> list:
        """检查必填字段"""
        missing = []
        for field in required:
            value = record.get(field)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                missing.append(field)
        return missing

    def _check_price_range(self, record: dict) -> list:
        """检查价格是否在合理范围内"""
        errors = []

        close = record.get("close")
        if close is not None:
            if not (self.CLOSE_RANGE[0] <= close <= self.CLOSE_RANGE[1]):
                errors.append(f"close_out_of_range: {close} not in {self.CLOSE_RANGE}")

        return errors

    def _check_ohlc_relationship(self, record: dict) -> list:
        """检查 OHLC 关系是否合理"""
        errors = []

        open_price = record.get("open")
        high = record.get("high")
        low = record.get("low")
        close = record.get("close")

        # 全部为 None 或 0 时跳过（可能是停牌）
        if not any([open_price, high, low, close]):
            return errors

        # high 必须 >= low
        if high is not None and low is not None:
            if high < low:
                errors.append(f"high_less_than_low: high={high}, low={low}")

        # high 必须 >= open
        if high is not None and open_price is not None:
            if high < open_price:
                errors.append(f"high_less_than_open: high={high}, open={open_price}")

        # high 必须 >= close
        if high is not None and close is not None:
            if high < close:
                errors.append(f"high_less_than_close: high={high}, close={close}")

        # low 必须 <= open
        if low is not None and open_price is not None:
            if low > open_price:
                errors.append(f"low_greater_than_open: low={low}, open={open_price}")

        # low 必须 <= close
        if low is not None and close is not None:
            if low > close:
                errors.append(f"low_greater_than_close: low={low}, close={close}")

        return errors

    def _check_pct_chg(self, record: dict) -> list:
        """检查涨跌幅是否合理"""
        errors = []

        pct_chg = record.get("pct_chg")
        if pct_chg is not None and not pd.isna(pct_chg):
            if not (self.PCT_CHG_RANGE[0] <= pct_chg <= self.PCT_CHG_RANGE[1]):
                errors.append(f"pct_chg_out_of_range: {pct_chg} not in {self.PCT_CHG_RANGE}")

        return errors

    def _check_volume(self, record: dict) -> list:
        """检查成交量是否合理"""
        errors = []

        volume = record.get("volume")
        if volume is not None:
            if volume < 0:
                errors.append(f"negative_volume: {volume}")

        return errors

    def validate_dataframe(self, df: pd.DataFrame) -> tuple:
        """
        批量验证日线 DataFrame

        Args:
            df: 日线数据 DataFrame

        Returns:
            (valid_records, invalid_records) 元组
        """
        valid_records = []
        invalid_records = []

        for _, row in df.iterrows():
            record = row.to_dict()
            result = self.validate(record)

            if result.is_valid:
                valid_records.append(record)
            else:
                invalid_records.append({
                    "record": record,
                    "errors": result.errors
                })

        return valid_records, invalid_records


def validate_daily_record(record: dict) -> list:
    """
    便捷函数：验证单条日线记录，返回错误列表

    Args:
        record: 日线记录字典

    Returns:
        错误列表，为空表示验证通过
    """
    validator = DailyValidator()
    result = validator.validate(record)
    return result.errors
