"""Excel-only furnace gas prediction and recommendation package."""

from .data import FEATURE_COLS, TARGET_COL, load_batch_data, mark_target_outliers

__all__ = ["FEATURE_COLS", "TARGET_COL", "load_batch_data", "mark_target_outliers"]
