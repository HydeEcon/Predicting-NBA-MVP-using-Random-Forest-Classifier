import pandas as pd
import numpy as np
import requests
import time
import random
from pathlib import Path

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

START_YEAR = 1981       # 1980-81 season
END_YEAR = 2026         # 2025-26 season

BASE_URL = "https://www.basketball-reference.com/leagues"

# Folder where downloaded HTML files will be stored
CACHE_DIR = Path("basketball_reference_cache")
CACHE_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# REQUEST SETTINGS
# --------------------------------------------------

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Safari/537.36"
    )
}

session = requests.Session()
session.headers.update(headers)


# --------------------------------------------------
# TABLE TYPES
# --------------------------------------------------

stat_types = {
    "totals": "",
    "per_game": "per_game",
    "per_36": "per_minute",
    "per_100": "per_poss",
    "advanced": "advanced",
    "play_by_play": "play_by_play",
    "shooting": "shooting",
    "adjusted_shooting": "adj_shooting"
}


# --------------------------------------------------
# DOWNLOAD FUNCTION
# --------------------------------------------------

def download_page(url, filename, max_retries=5):

    filepath = CACHE_DIR / filename

    # Don't download it again if we already have it
    if filepath.exists():
        print(f"Already downloaded: {filename}")
        return filepath

    for attempt in range(max_retries):

        try:

            print(f"Downloading: {url}")

            response = session.get(
                url,
                timeout=30
            )

            # Successful request
            if response.status_code == 200:

                filepath.write_text(
                    response.text,
                    encoding="utf-8"
                )

                print(f"Saved: {filename}")

                # Random delay between requests
                time.sleep(random.uniform(4, 7))

                return filepath

            # Rate limited
            elif response.status_code == 429:

                wait_time = 30 * (attempt + 1)

                print(
                    f"429 rate limit. "
                    f"Waiting {wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:

                print(
                    f"HTTP {response.status_code}: {url}"
                )

                time.sleep(10)

        except requests.RequestException as e:

            print(f"Request error: {e}")

            time.sleep(20)

    print(f"FAILED: {url}")

    return None


# --------------------------------------------------
# READ THE PLAYER TABLE
# --------------------------------------------------

def read_player_table(filepath):

    if filepath is None:
        return None

    try:

        tables = pd.read_html(filepath)

        # Basketball-Reference can contain multiple
        # tables on a page. Find the player table.

        for table in tables:

            if "Player" in table.columns:

                return table

        print(f"No Player table found in {filepath}")

        return None

    except Exception as e:

        print(f"Could not read {filepath}: {e}")

        return None


# --------------------------------------------------
# MAIN SCRAPER
# --------------------------------------------------

all_data = []


for year in range(START_YEAR, END_YEAR + 1):

    season = f"{year - 1}-{str(year)[-2:]}"

    print("\n" + "=" * 60)
    print(f"SEASON: {season}")
    print("=" * 60)

    season_tables = []

    for stat_name, suffix in stat_types.items():

        if suffix == "":
            url = (
                f"{BASE_URL}/NBA_{year}_totals.html"
            )
        else:
            url = (
                f"{BASE_URL}/NBA_{year}_{suffix}.html"
            )

        filename = f"NBA_{year}_{stat_name}.html"

        filepath = download_page(
            url,
            filename
        )

        df = read_player_table(filepath)

        if df is None:
            continue

        # Add season
        df["Season"] = season

        # Add stat category
        df["stat_source"] = stat_name

        season_tables.append(
            (stat_name, df)
        )

    # --------------------------------------------------
    # MERGE STAT TABLES FOR THIS SEASON
    # --------------------------------------------------

    if not season_tables:

        print(f"No data found for {season}")
        continue

    # Start with totals
    totals = None

    for stat_name, df in season_tables:

        if stat_name == "totals":

            totals = df.copy()
            break

    if totals is None:

        print(f"No totals table for {season}")
        continue

    # --------------------------------------------------
    # CREATE MVP VARIABLE
    # --------------------------------------------------

    if "Awards" in totals.columns:

        totals["MVP"] = np.where(
            totals["Awards"].astype(str).str.contains(
                r"\bMVP-1\b",
                case=False,
                na=False,
                regex=True
            ),
            1,
            0
        )

    # --------------------------------------------------
    # MERGE OTHER STAT CATEGORIES
    # --------------------------------------------------

    for stat_name, df in season_tables:

        if stat_name == "totals":
            continue

        # Remove duplicate columns that will cause
        # conflicts during merge

        columns_to_keep = [
            col for col in df.columns
            if col not in [
                "Player",
                "Season",
                "Rk",
                "Age",
                "Team",
                "Tm",
                "Pos",
                "G",
                "GS",
                "MP"
            ]
        ]

        df_merge = df[
            ["Player"] + columns_to_keep
        ].copy()

        # Prefix columns so we know where they came from
        rename_dict = {}

        for col in columns_to_keep:

            rename_dict[col] = (
                f"{stat_name}_{col}"
            )

        df_merge.rename(
            columns=rename_dict,
            inplace=True
        )

        # Merge using Player
        totals = totals.merge(
            df_merge,
            on="Player",
            how="left"
        )

    all_data.append(totals)

    print(
        f"Finished {season}: "
        f"{len(totals)} players"
    )


# --------------------------------------------------
# COMBINE ALL SEASONS
# --------------------------------------------------

if all_data:

    combined = pd.concat(
        all_data,
        ignore_index=True
    )

    combined.to_csv(
        "NBA_player_stats_1980_2026.csv",
        index=False
    )

    print("\nDONE!")
    print(
        f"Rows: {len(combined):,}"
    )

    print(
        f"Columns: {len(combined.columns):,}"
    )

else:

    print("No data was successfully collected.")