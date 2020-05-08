from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class IRR_result:
    method: str
    subjects: int
    raters: int
    irr_name: str
    value: float
    statistic: float = None
    stat_name: str = None
    pvalue: float = None
    detail: Any = None
    error: str = None

    def to_dict(self):
        return asdict(self)

    def __repr__(self):
        model_string = "=" * 50 + "\n"
        model_string += f"{self.method}".center(50, " ") + "\n"
        model_string += "=" * 50 + "\n"
        model_string += f"Subjects = {self.subjects}\n  Raters = {self.raters}\n"
        model_string += f"{self.irr_name:>8}" + f" = {self.value:.3f}\n\n"

        if self.statistic is not None:
            model_string += f"{self.stat_name:>8} = {self.statistic:.3f}\n"
            model_string += f" p-value = {self.pvalue:.3f}\n"

        if self.detail is not None:
            model_string += f"\n{self.detail}\n"

        if self.error is not None:
            model_string += f"\n{self.error}\n"
        model_string += "=" * 50 + "\n"
        return model_string
