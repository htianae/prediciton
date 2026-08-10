"""Chronological Excel loading and immutable data contracts."""

from pathlib import Path

import pandas as pd


TARGET_COL = "熔炼炉B当前批次总气耗_PLC"
FEATURE_COLS = [
    "10#熔炼炉总投料重量(kg)",
    "10#熔炼炉固体料重量比例",
    "熔炼炉B当前批次熔炼时间_PLC",
    "熔炼炉B当前批次等待时长_PLC",
    "熔炼炉B当前批次炉门打开次数_PLC",
    "熔炼炉B当前批次炉门打开时长_PLC",
]
WEIGHT_COL, SOLID_COL, MELTING_COL, WAIT_COL, DOOR_COUNT_COL, DOOR_DURATION_COL = FEATURE_COLS


def load_batch_data(path: str | Path, sheet_name: str = "Sheet1") -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")
    batch_cols = [
        col for col in raw.columns
        if isinstance(raw.iloc[0, col], str) and raw.iloc[0, col].startswith("ER")
    ]
    variable_rows = [
        row for row in raw.index
        if isinstance(raw.iloc[row, 0], str) and raw.iloc[row, 0] != "Grand Total"
    ]
    values = raw.loc[variable_rows, batch_cols].apply(pd.to_numeric, errors="coerce").T
    values.columns = raw.loc[variable_rows, 0].astype(str).tolist()
    values = values.reset_index(drop=True)
    values.insert(0, "batch_id", [str(raw.iloc[0, col]).strip() for col in batch_cols])
    required = FEATURE_COLS + [TARGET_COL]
    missing = [column for column in required if column not in values]
    if missing:
        raise ValueError(f"Excel 缺少建模列: {missing}")
    if values["batch_id"].duplicated().any():
        raise ValueError("Excel 包含重复炉次号。")
    result = values[["batch_id"] + required].copy()
    if result[TARGET_COL].isna().any():
        raise ValueError("目标总气耗包含缺失值。")
    return result


def mark_target_outliers(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    q1, q3 = result[TARGET_COL].quantile([0.25, 0.75])
    result["is_high_gas_outlier"] = result[TARGET_COL] > q3 + 1.5 * (q3 - q1)
    return result


def chronological_dev_lock_split(df: pd.DataFrame, lock_size: int = 44) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) <= lock_size:
        raise ValueError("数据量不足以建立锁定测试集。")
    split = len(df) - lock_size
    return df.iloc[:split].copy(), df.iloc[split:].copy()
