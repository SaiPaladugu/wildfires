"""
Filter climate stations that have data covering the entire period 1972-2024.
Outputs a CSV with station ID, start year, end year, and name.
"""

import re
import pandas as pd
from pathlib import Path


def filter_stations(
    input_path: Path,
    output_path: Path,
    start_year: int = 1972,
    end_year: int = 2024,
) -> pd.DataFrame:
    """
    Find stations with data spanning the full period from start_year to end_year.

    Args:
        input_path: Path to the source climate-stations CSV.
        output_path: Path to write the filtered output CSV.
        start_year: First year that must be covered (inclusive).
        end_year: Last year that must be covered (inclusive).

    Returns:
        DataFrame of matching stations.
    """
    df = pd.read_csv(input_path, parse_dates=["FIRST_DATE", "LAST_DATE"])

    period_start = pd.Timestamp(f"{start_year}-01-01")
    period_end = pd.Timestamp(f"{end_year}-12-31")

    mask = (df["FIRST_DATE"] <= period_start) & (df["LAST_DATE"] >= period_end)
    filtered = df.loc[mask, ["STN_ID", "STATION_NAME", "PROV_STATE_TERR_CODE"]].copy()
    filtered = filtered.reset_index(drop=True)

    clean_name = filtered["STATION_NAME"].apply(lambda n: re.sub(r"[^A-Za-z]", "", n).strip())
    filtered["STATION_ID"] = filtered["STN_ID"]
    filtered["START_YEAR"] = start_year
    filtered["END_YEAR"] = end_year
    filtered["NAME"] = clean_name + "_" + filtered["PROV_STATE_TERR_CODE"]

    output = filtered[["STATION_ID", "START_YEAR", "END_YEAR", "NAME"]]
    output.to_csv(output_path, index=False)
    print(f"Found {len(output)} station(s) with data from {start_year} to {end_year}.")
    print(f"Results written to: {output_path}")
    return output


if __name__ == "__main__":
    project_root = Path(__file__).parent
    input_path = project_root / "data" / "climate-stations.csv"
    output_path = project_root / "data" / "stations_1972_2024.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = filter_stations(input_path, output_path)
    print(result.to_string(index=False))
