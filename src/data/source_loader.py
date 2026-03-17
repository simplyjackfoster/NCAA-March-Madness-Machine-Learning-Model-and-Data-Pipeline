from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


class DataSourceError(RuntimeError):
    """Raised when a configured external data source cannot be loaded."""


def load_csv_source(
    *,
    source_name: str,
    root: Path,
    year: int,
    local_path: str | None = None,
    url_template: str | None = None,
    required_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load a dataset from an optional local path or URL template.

    Priority:
      1) local_path if provided
      2) url_template formatted with {year}
    """
    df: pd.DataFrame | None = None

    if local_path:
        candidate = Path(local_path)
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.exists():
            raise DataSourceError(f"{source_name}: local_path does not exist: {candidate}")
        df = pd.read_csv(candidate)

    elif url_template:
        url = url_template.format(year=year)
        try:
            df = pd.read_csv(url)
        except Exception as exc:  # pragma: no cover - network variance
            raise DataSourceError(f"{source_name}: failed to read URL {url}: {exc}") from exc

    if df is None:
        raise DataSourceError(f"{source_name}: no source configured")

    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise DataSourceError(
                f"{source_name}: missing required columns {missing}; available={list(df.columns)}"
            )
    return df
