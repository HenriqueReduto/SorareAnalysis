"""Dataset inspection utilities."""

from __future__ import annotations

import pandas as pd


def infer_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """Infer numerical, categorical, and date-like columns."""
    numerical = df.select_dtypes(include="number").columns.tolist()
    date_cols: list[str] = []
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            date_cols.append(col)
    categorical = [
        col for col in df.columns
        if col not in numerical and col not in date_cols
    ]
    return {
        "numerical": numerical,
        "categorical": categorical,
        "date": date_cols,
    }


def inspect_dataset(df: pd.DataFrame) -> dict[str, object]:
    """Print and return mandatory inspection information."""
    inferred = infer_column_types(df)
    report = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().sort_values(ascending=False).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "sample_rows": df.head(10),
        "descriptive_statistics": df.describe(include="all"),
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
        "inferred_columns": inferred,
    }

    print("Shape:", report["shape"])
    print("Columns:", report["columns"])
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isna().sum().sort_values(ascending=False))
    print("\nDuplicate row count:", report["duplicate_rows"])
    print("\nSample rows:\n", df.head())
    print("\nDescriptive statistics:\n", df.describe(include="all"))
    print("\nMemory usage bytes:", report["memory_usage_bytes"])
    print("\nInferred columns:", inferred)
    return report
