"""
results/persistence.py
----------------------
CSV helpers making sure records are not overwritten
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_existing_results(csv_path: Path) -> pd.DataFrame:
    """Load the persistent results CSV, or return an empty DataFrame if absent."""
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df
    return pd.DataFrame()


def append_results(
    new_rows: list[dict],
    csv_path: Path,
    run_id:   str,
) -> pd.DataFrame:
    """
    Tag new_rows with run_id, append to the persistent CSV, and return
    the full combined DataFrame (existing rows + new rows).
    """
    if not new_rows:
        return load_existing_results(csv_path)

    new_df = pd.DataFrame(new_rows)
    new_df.insert(0, 'run_id', run_id)

    existing = load_existing_results(csv_path)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(csv_path, index=False)
    print(f"New experiment has been saved")
    return combined
