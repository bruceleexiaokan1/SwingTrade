"""股票列表验证器"""

import re
from typing import Optional
from dataclasses import dataclass

# 市场前缀映射
MARKET_PREFIXES = {
    "sh": ["600", "601", "603", "605", "688", "689", "500", "510", "150", "159", "560", "508"],
    "sz": ["000", "001", "002", "003", "300", "150", "159", "560"],
    "bj": ["830", "870", "889", "430", "400"]
}


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: list[str]
    warnings: list[str]


class StockValidator:
    """股票列表验证器"""

    def __init__(self):
        self.today = None  # 将在首次调用时设置

    @property
    def current_date(self) -> str:
        """获取当前日期（惰性加载）"""
        if self.today is None:
            from datetime import date
            self.today = date.today().strftime("%Y-%m-%d")
        return self.today

    def validate(self, record: dict) -> ValidationResult:
        """
        验证单条股票记录

        Args:
            record: 包含股票信息的字典，必须包含 code, name, market 字段

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        # 1. 代码格式：6位数字
        code = record.get("code", "")
        if not self._validate_code_format(code):
            errors.append(f"invalid_code_format: {code}")

        # 2. 市场标识
        market = record.get("market", "")
        if not self._validate_market(market):
            errors.append(f"invalid_market: {market}")

        # 3. 市场前缀匹配
        if code and market:
            if not self._validate_code_market_match(code, market):
                errors.append(f"code_market_mismatch: {code} not in {market}")

        # 4. 名称验证
        name = record.get("name", "")
        if not is_valid_company_name(name):
            errors.append(f"invalid_name: {name}")

        # 5. 日期逻辑
        list_date = record.get("list_date")
        if list_date:
            date_error = self._validate_list_date(list_date)
            if date_error:
                errors.append(date_error)

        # 6. 跨源一致性检查（如果有）
        if "verify_name" in record and record["verify_name"]:
            if name != record["verify_name"]:
                warnings.append(f"name_mismatch: {name} vs {record['verify_name']}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def _validate_code_format(self, code: str) -> bool:
        """验证代码格式是否为6位数字"""
        return bool(re.match(r"^\d{6}$", code))

    def _validate_market(self, market: str) -> bool:
        """验证市场标识是否合法"""
        return market in {"sh", "sz", "bj"}

    def _validate_code_market_match(self, code: str, market: str) -> bool:
        """验证代码前缀与市场标识是否匹配"""
        prefixes = MARKET_PREFIXES.get(market, [])
        return any(code.startswith(p) for p in prefixes)

    def _validate_list_date(self, list_date: str) -> Optional[str]:
        """验证上市日期逻辑"""
        from datetime import datetime

        try:
            date_obj = datetime.strptime(list_date, "%Y-%m-%d")
        except ValueError:
            return f"invalid_date_format: {list_date}"

        # 不能是未来日期
        if date_obj.strftime("%Y-%m-%d") > self.current_date:
            return f"future_list_date: {list_date}"

        return None


def validate_stock_record(record: dict) -> list:
    """
    便捷函数：验证单条股票记录，返回错误列表

    Args:
        record: 股票记录字典

    Returns:
        错误列表，为空表示验证通过
    """
    validator = StockValidator()
    result = validator.validate(record)
    return result.errors


def is_valid_company_name(name: str) -> bool:
    """
    检查公司名称是否合法

    Args:
        name: 公司名称

    Returns:
        True if valid, False otherwise
    """
    if not name or len(name) < 2 or len(name) > 20:
        return False

    # 不可见字符（控制字符）
    if re.search(r"[\x00-\x1f\x7f-\x9f]", name):
        return False

    # 纯数字或纯特殊字符（排除纯数字代码冒充名称）
    if re.match(r"^[\d\W]+$", name):
        return False

    # 中文乱码检测（常见错误编码）
    # 如果包含 ASCII 范围内的异常字符组合
    try:
        name.encode("gbk")
    except UnicodeEncodeError:
        return False

    return True


def validate_stock_list(df: "pd.DataFrame") -> tuple:
    """
    批量验证股票列表

    Args:
        df: 股票列表 DataFrame

    Returns:
        (valid_records, invalid_records) 元组
    """
    import pandas as pd

    validator = StockValidator()
    valid_records = []
    invalid_records = []

    for _, row in df.iterrows():
        record = row.to_dict()
        result = validator.validate(record)

        if result.is_valid:
            valid_records.append(record)
        else:
            invalid_records.append({
                "record": record,
                "errors": result.errors,
                "warnings": result.warnings
            })

    return valid_records, invalid_records
