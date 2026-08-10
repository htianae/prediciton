"""Load the transposed furnace workbook into a chronological batch table."""

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


def load_batch_data(path: str | Path, sheet_name: str = "Sheet1") -> pd.DataFrame:
    """Return one chronological row per ER batch from the source workbook."""
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")
    batch_cols = [
        col
        for col in raw.columns
        if isinstance(raw.iloc[0, col], str) and raw.iloc[0, col].startswith("ER")
    ]
    variable_rows = [
        row
        for row in raw.index
        if isinstance(raw.iloc[row, 0], str) and raw.iloc[row, 0] != "Grand Total"
    ]
    values = raw.loc[variable_rows, batch_cols].apply(pd.to_numeric, errors="coerce").T
    values.columns = raw.loc[variable_rows, 0].astype(str).tolist()
    values = values.reset_index(drop=True)
    values.insert(0, "batch_id", [str(raw.iloc[0, col]).strip() for col in batch_cols])

    required = FEATURE_COLS + [TARGET_COL]
    missing = [column for column in required if column not in values.columns]
    if missing:
        raise ValueError(f"Excel 缺少建模列: {missing}")
    if values["batch_id"].duplicated().any():
        duplicates = values.loc[values["batch_id"].duplicated(), "batch_id"].tolist()
        raise ValueError(f"炉次号重复: {duplicates}")
    return values[["batch_id"] + FEATURE_COLS + [TARGET_COL]].copy()


def mark_target_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Add an IQR diagnostic flag without deleting or changing any batch."""
    result = df.copy()
    target = pd.to_numeric(result[TARGET_COL], errors="coerce")
    q1, q3 = target.quantile([0.25, 0.75])
    iqr = q3 - q1
    result["is_high_gas_outlier"] = target > q3 + 1.5 * iqr
    return result
