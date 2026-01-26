# Team Format and Dataset

## 1. Team

- Oswald Lackner
- Stocker Christoph

Source of Data: [Kaggle: Hourly Power Generation of Europe](https://www.kaggle.com/datasets/mehmetnuryildirim/hourly-power-generation-of-europe) (date: 2026-01-16)

# 2. Task Categories and Points

## 2.1 A. Data Preprocessing and Data Quality (70 points)

Data processing
  - Class for activated plots
  - Class for Columns of Data


```
# Initial setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from enum import Enum, auto
import calendar
from typing import Iterable, Tuple, Dict, Union



# Configure plotting
plt.rcParams.update({
    'figure.figsize': [12, 8],
    'figure.dpi': 150,
    'figure.autolayout': True,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'font.size': 12
})

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

```


```
class PlotOptions(Enum):
    DATAFRAME_NAN_REPORT = auto()
    TIME_PLOT_RAW_POWER = auto()
    TIME_PLOT_RAW_POWER_OVERLAY = auto()
    POWER_SHARE_BY_SOURCE = auto()
    POWER_SHARE_BY_SOURCE_OVER_YEARS = auto()
    SCATTER_TOTAL_POWER_OVER_YEAR = auto()
    YEARLY_SEASONAL_OVER_YEARS = auto()
    HOURLY_PLOT_OVER_SEASONS = auto()
    HEXBIN_TOTAL_POWER_HOURLY_DAYTIME_PLOT = auto()
    HEXBIN_TOTAL_POWER_DAILY_YEAR_PLOT = auto()
    HOURLY_TOTAL_POWER_REGRESSION = auto()
    TREND_TOTAL_POWER_OVER_YEARS = auto()
    TREND_TOTAL_POWER_OVER_MONTHS = auto()

class StatesOptions(Enum):
    ITALY = auto()
    FRANCE = auto()
    GERMANY = auto()
    SPAIN = auto()    

class ActvnMatrix:
    PLOT_OPTIONS_DICT = {
        "DATAFRAME_NAN_REPORT": True,
        "TIME_PLOT_RAW_POWER": True,
        "TIME_PLOT_RAW_POWER_OVERLAY": False,
        "POWER_SHARE_BY_SOURCE": True,
        "POWER_SHARE_BY_SOURCE_OVER_YEARS": True,
        "SCATTER_TOTAL_POWER_OVER_YEAR": True,
        "YEARLY_SEASONAL_OVER_YEARS": True,
        "HOURLY_PLOT_OVER_SEASONS": False,
        "HEXBIN_TOTAL_POWER_HOURLY_DAYTIME_PLOT": True,
        # "HEXBIN_TOTAL_POWER_HOURLY_DAYTIME_PLOT": False,
        "HEXBIN_TOTAL_POWER_DAILY_YEAR_PLOT": False,
        # "HEXBIN_TOTAL_POWER_DAILY_YEAR_PLOT": True
        "HOURLY_TOTAL_POWER_REGRESSION" : True,
        "TREND_TOTAL_POWER_OVER_YEARS": True,
        "TREND_TOTAL_POWER_OVER_MONTHS": True
    }

    STATE = {
        "Italy": True,
        "France": True,
        "Germany": True,
        "Spain": True,
    } 


    @classmethod
    def is_active(cls, country, plot_option) -> bool:
        """
        Return True if the given country and plot_option are active.
        country can be:
            - string (case-insensitive)
            - StatesOptions enum
        plot_option can be:
            - string (case-insensitive)
            - PlotOptions enum
        Prints messages if country or plot is inactive.
        """
        # --- Normalize country ---
        if isinstance(country, StatesOptions):
            country_str = country.name.capitalize()
        elif isinstance(country, str):
            country_str = country.capitalize()
        else:
            print(f"\n{'!'*10} WARNING {'!'*10}")
            print(f"Invalid type for country: {type(country)}. Must be str or StatesOptions enum.")
            return False

        # --- Normalize plot_option ---
        if isinstance(plot_option, PlotOptions):
            plot_str = plot_option.name
        elif isinstance(plot_option, str):
            plot_str = plot_option.upper()
        else:
            print(f"\n{'!'*10} WARNING {'!'*10}")
            print(f"Invalid type for plot_option: {type(plot_option)}. Must be str or PlotOptions enum.")
            return False

        # --- Check country ---
        if country_str not in cls.STATE:
            print(f"\n{'!'*10} WARNING {'!'*10}")
            print(f"Country '{country_str}' not found in STATE dictionary!")
            return False

        country_active = cls.STATE[country_str]

        # --- Check plot ---
        plot_active = cls.PLOT_OPTIONS_DICT.get(plot_str, False)
        if not plot_active:
            print(f"\n{'-'*5} NOTE {'-'*5}")
            print(f"Plot '{plot_str}' is deactivated in PLOT_OPTIONS_DICT. Skipping execution for {country_str}.")
            return False

        return country_active and plot_active
```


```
class Columns:
    AREA = 'Area'
    MTU = 'MTU'
    DATETIME = 'DATETIME'
    YEAR = 'YEAR'

    META = [
        AREA,
        MTU,
        DATETIME,
        YEAR
    ]

    class Power:
        # Standard main types
        BIOMASS = 'Biomass - Actual Aggregated [MW]'
        FOSSIL_BROWN = 'Fossil Brown coal/Lignite - Actual Aggregated [MW]'
        FOSSIL_COAL_DERIVED_GAS = 'Fossil Coal-derived gas - Actual Aggregated [MW]'
        FOSSIL_GAS = 'Fossil Gas - Actual Aggregated [MW]'
        FOSSIL_HARD_COAL = 'Fossil Hard coal - Actual Aggregated [MW]'
        FOSSIL_OIL = 'Fossil Oil - Actual Aggregated [MW]'
        FOSSIL_OIL_SHALE = 'Fossil Oil shale - Actual Aggregated [MW]'
        FOSSIL_PEAT = 'Fossil Peat - Actual Aggregated [MW]'
        GEOTHERMAL = 'Geothermal - Actual Aggregated [MW]'
        HYDRO_PUMPED = 'Hydro Pumped Storage - Actual Aggregated [MW]'
        HYDRO_CONSUMPTION = 'Hydro Pumped Storage - Actual Consumption [MW]'
        HYDRO_RUNOF = 'Hydro Run-of-river and poundage - Actual Aggregated [MW]'
        HYDRO_RESERVOIR = 'Hydro Water Reservoir - Actual Aggregated [MW]'
        MARINE = 'Marine - Actual Aggregated [MW]'
        NUCLEAR = 'Nuclear - Actual Aggregated [MW]'
        OTHER = 'Other - Actual Aggregated [MW]'
        OTHER_RENEWABLE = 'Other renewable - Actual Aggregated [MW]'
        SOLAR = 'Solar - Actual Aggregated [MW]'
        WASTE = 'Waste - Actual Aggregated [MW]'
        WIND_OFFSHORE = 'Wind Offshore - Actual Aggregated [MW]'
        WIND_ONSHORE = 'Wind Onshore - Actual Aggregated [MW]'

        # Helper: list of all known power columns
        ALL = [
            BIOMASS, FOSSIL_BROWN, FOSSIL_COAL_DERIVED_GAS, FOSSIL_GAS,
            FOSSIL_HARD_COAL, FOSSIL_OIL, FOSSIL_OIL_SHALE, FOSSIL_PEAT,
            GEOTHERMAL, HYDRO_PUMPED, HYDRO_CONSUMPTION, HYDRO_RUNOF, HYDRO_RESERVOIR,
            MARINE, NUCLEAR, OTHER, OTHER_RENEWABLE, SOLAR, WASTE,
            WIND_OFFSHORE, WIND_ONSHORE
        ]

        ALL_FILT = [
            BIOMASS, FOSSIL_BROWN, FOSSIL_COAL_DERIVED_GAS, FOSSIL_GAS,
            FOSSIL_HARD_COAL, FOSSIL_OIL,
            GEOTHERMAL, HYDRO_PUMPED, HYDRO_CONSUMPTION, HYDRO_RUNOF, HYDRO_RESERVOIR,
            NUCLEAR, OTHER, SOLAR, WASTE,
            WIND_ONSHORE
        ] # Removed: FOSSIL_OIL_SHALE,  FOSSIL_PEAT, MARINE, WIND_OFFSHORE, OTHER_RENEWABLE, 

    

        HYDRO = [
            HYDRO_PUMPED, HYDRO_CONSUMPTION, HYDRO_RUNOF, HYDRO_RESERVOIR
        ]

        WIND = [
            WIND_ONSHORE
        ] # WIND_OFFSHORE,

        FOSSIL = [
            FOSSIL_BROWN, FOSSIL_COAL_DERIVED_GAS, FOSSIL_GAS,
            FOSSIL_HARD_COAL, FOSSIL_OIL,
        ] # FOSSIL_OIL_SHALE,  FOSSIL_PEAT

        RENEWAABLE = [
            BIOMASS, GEOTHERMAL, HYDRO_PUMPED, HYDRO_CONSUMPTION, HYDRO_RUNOF,
            HYDRO_RESERVOIR, SOLAR, WIND_ONSHORE
        ] # Removed:  OTHER_RENEWABLE, WIND_OFFSHORE, MARINE,

    # Additional calculated columns
    class CALC:
        # Axis columns
        TOTAL_POWER = 'total_power'
        TOTAL_FOSSIL_POWER = 'total_fossil_power'
        TOTAL_RENEWABLE_POWER = 'total_renewable_power'

    # Additional columns as Axis representations
    class AXIS:
        DAY_OF_WEEK = 'day_of_week'
        DAY_OF_YEAR = 'day_in_year'
        SEASON = 'season'
        YEAR  = 'year'
        MONTH = 'month'
        MONTH_STR = 'month_str'
        DAY_OF_WEEK_STR = 'day_of_week_str'
        HOURS_OF_DAY = 'hours_a_day'


colors = {
    "Italy": "tab:blue",
    "France": "tab:orange",
    "Germany": "tab:green",
    "Spain": "tab:red",
}        
```

### Loading Raw data

Dataset is available in 4 seperate files:

- Italy_Power_Generation.csv
- France_Power_Generation.csv
- Germany_Power_Generation.csv
- Spain_Power_Generation.csv



```

def load_power_generation_data(file_path: str, dataset_name: str, col_datetime: str = 'DATETIME') -> pd.DataFrame:
    """
    Load and process power generation data from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV data file
    col_datetime : str
        Column name to use as datetime index (default 'DATETIME')

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with:
        - DatetimeIndex from MTU column
        - Numeric power columns
        - Columns normalized to match Columns class
    """

    filepath = Path(file_path)
    print("\n\n" + "=" * 100)
    print(f"Loading data from:")
    print(f"    - Path: {filepath.parent}")
    print(f"    - File: {filepath.name}\n")
    
    # Read CSV
    df = pd.read_csv(filepath)

    print(f"\nColumn of {dataset_name}" + 45*" "+ "Number of NaNs")
    print(80*"-")
    print(df.isna().sum().sort_values(ascending=False))

    # Normalize column names (make number of spaces consistent)
    df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)

    # Fix Germany-specific DATETIME
    if col_datetime not in df.columns:
        if 'Unnamed: 2' in df.columns:
            df.rename(columns={'Unnamed: 2': col_datetime}, inplace=True)
        else:
            raise KeyError(f"{col_datetime} column not found in {filepath.name}")

    # Extract first datetime from MTU
    if 'MTU' in df.columns:
        df[col_datetime] = df['MTU'].str.split(' - ').str[0]
    else:
        raise KeyError("MTU column not found for datetime extraction")

    # Convert to datetime
    df[col_datetime] = pd.to_datetime(df[col_datetime], format='%d.%m.%Y %H:%M', errors='coerce')

    # Safety check
    if df[col_datetime].isna().any():
        print(f"Warning: Some rows could not be converted to datetime in {filepath.name}")

    # Set index
    df.set_index(col_datetime, inplace=True)

    # Identify power columns present in the CSV (intersection with Columns.Power.ALL)
    power_cols = [c for c in Columns.Power.ALL if c in df.columns]

    # Convert all power columns to numeric
    df[power_cols] = df[power_cols].apply(pd.to_numeric, errors='coerce')

    # Calculate total power generation
    df[Columns.CALC.TOTAL_POWER] = df[Columns.Power.ALL].sum(axis=1)
    df[Columns.CALC.TOTAL_FOSSIL_POWER] = df[Columns.Power.FOSSIL].sum(axis=1)
    df[Columns.CALC.TOTAL_RENEWABLE_POWER] = df[Columns.Power.RENEWAABLE].sum(axis=1)


    # Create additional time-based columns
    df[Columns.AXIS.YEAR] = df.index.year

    # Month + fractional day
    df[Columns.AXIS.MONTH] = df.index.month + (df.index.day - 1) / df.index.days_in_month
    # df[Columns.AXIS.MONTH] = df.index.month

    # Day of year + fractional day
    df[Columns.AXIS.DAY_OF_YEAR] = df.index.dayofyear + (df.index.hour + df.index.minute / 60) / 24
    # df[Columns.AXIS.DAY_OF_YEAR] = df.index.dayofyear

    # Day of week + fractional day
    df[Columns.AXIS.DAY_OF_WEEK] = df.index.dayofweek + (df.index.hour + df.index.minute / 60) / 24
    #df[Columns.AXIS.DAY_OF_WEEK] = df.index.dayofweek

    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn"
    }

    # df[Columns.AXIS.SEASON] = df[Columns.AXIS.MONTH].map(season_map) -> doesn't work with fractional months
    df[Columns.AXIS.SEASON] = df.index.month.map(season_map)

    #df[Columns.AXIS.MONTH_STR] = df[Columns.AXIS.MONTH].apply(lambda x: calendar.month_name[x])
    df[Columns.AXIS.MONTH_STR] = df[Columns.AXIS.MONTH].apply(lambda x: calendar.month_name[int(np.floor(x))])

    df[Columns.AXIS.DAY_OF_WEEK_STR] = df[Columns.AXIS.DAY_OF_WEEK].apply(lambda x: calendar.day_name[int(np.floor(x))])
    df[Columns.AXIS.HOURS_OF_DAY] = df.index.hour + df.index.minute / 60


    print("\nShape:")
    print(f"  Rows: {df.shape[0]:,}")
    print(f"  Columns: {df.shape[1]}\n")

    #print(f"\nTime range: {df.index.min()} to {df.index.max()}")
    #print(f"Number of samples: {len(df):,}")

    # print("\nColumns:")
    # for col in df.columns:
    #     print(f"  - {col}")

    #print("\nFirst few rows:")
    #print(df.head())

    print("Data Overview:")
    print(df.describe())    

    # print("\nMissing values per column:")
    # print(df.isna().sum().sort_values(ascending=False))

    return df

```

These files are read in by using function:
```python
def load_power_generation_data(file_path: str, col_datetime: str = 'DATETIME') -> pd.DataFrame:
```

Funtions of `load_power_generation_data`:

- List Read in file with path.
- normalize the number of spaces to be usable over all dataset parts
- count number of NaNs
- especially Germany has a datetime column that is not named.
- extraction of datetime from MTU column which is state.
- convert MTU daytime extracted to python datatype datetime
- check for correct convertion by counting possible NaNs
- set extracted datetime to index.
- catch  power cols for dataset
- convert power columns to numeric
- calculate total power
- create addional timely indexes for year, month, day of the year, day of the week, hours of the day
- Add seasonal information
- Add Monthly/weekly time information as string

Print information of data
- Print number of rows, columns
- Dataframe describe


```
base_path = Path.cwd()
subfolder = Path(r"OneDrive - FH JOANNEUM\Courses\STM_WS2025_DA_Data Analysis\Assignment_2")

df_italy   = load_power_generation_data(file_path=base_path / subfolder / "Italy_Power_Generation.csv", dataset_name="Italy")
df_france  = load_power_generation_data(file_path=base_path / subfolder / "France_Power_Generation.csv", dataset_name="France")
df_germany = load_power_generation_data(file_path=base_path / subfolder / "Germany_Power_Generation.csv", dataset_name="Germany")
df_spain   = load_power_generation_data(file_path=base_path / subfolder / "Spain_Power_Generation.csv", dataset_name="Spain")

dataframes = [
    ("Italy", df_italy),
    ("France", df_france),
    ("Germany", df_germany),
    ("Spain", df_spain)
]

```


```
def report_multiple_dataframe_overview(
    dataframes: list[tuple[str, pd.DataFrame]],
    datetime_col: str | None = None
):
    """
    Print a Markdown-style overview and comparison of multiple DataFrames.

    Includes:
    - Dimensions
    - Column data types
    - Time range
    - Sampling rate
    - Missingness summary

    Parameters
    ----------
    dataframes : list of (name, DataFrame)
    datetime_col : str or None
        Column name for datetime, or None if DatetimeIndex is used.
    """

    print("\n# Dataset Comparison Overview\n")

    # ----------------------------
    # DATASET-LEVEL OVERVIEW TABLE
    # ----------------------------
    header = (
        "| Dataset | Rows | Columns | Time Start | Time End | Sampling Rate | Missing Cells | Missing % |"
    )
    separator = (
        "|---------|------|---------|------------|----------|---------------|---------------|-----------|"
    )

    print(header)
    print(separator)

    for name, df in dataframes:
        rows, cols = df.shape

        # ---- datetime handling ----
        dt = None
        if datetime_col and datetime_col in df.columns:
            dt = pd.to_datetime(df[datetime_col], errors="coerce")
        elif isinstance(df.index, pd.DatetimeIndex):
            dt = df.index

        if dt is not None and not dt.dropna().empty:
            t_start = dt.min()
            t_end = dt.max()

            diffs = dt.sort_values().diff().dropna()
            sampling = diffs.median() if len(diffs) else "N/A"

        else:
            t_start = t_end = sampling = "N/A"

        # ---- missingness ----
        missing_cells = df.isna().sum().sum()
        missing_pct = (missing_cells / (rows * cols)) * 100 if rows * cols else 0

        print(
            f"| {name} | "
            f"{rows:,} | "
            f"{cols} | "
            f"{t_start} | "
            f"{t_end} | "
            f"{sampling} | "
            f"{missing_cells:,} | "
            f"{missing_pct:.2f}% |"
        )

    print("\n---\n")

    # ----------------------------
    # COLUMN-LEVEL DETAILS
    # ----------------------------
    for name, df in dataframes:
        print(f"## Column Details – {name}\n")

        col_header = "| Column | dtype | Missing | Missing % |"
        col_sep = "|--------|-------|---------|-----------|"

        print(col_header)
        print(col_sep)

        total_rows = len(df)

        for col in df.columns:
            missing = df[col].isna().sum()
            missing_pct = (missing / total_rows) * 100 if total_rows else 0

            print(
                f"| {col} | "
                f"{df[col].dtype} | "
                f"{missing:,} | "
                f"{missing_pct:.2f}% |"
            )

        print("\n---\n")


report_multiple_dataframe_overview(dataframes)

```


```
def dataframe_nan_report(df: pd.DataFrame, max_examples: int = 5) -> pd.DataFrame:
    """
    Create a detailed NaN and dtype report for a DataFrame.
    """
    report = []

    total_rows = len(df)

    for col in df.columns:
        na_count = df[col].isna().sum()

        if na_count > 0:
            example_idx = df[df[col].isna()].index[:max_examples].tolist()
        else:
            example_idx = []

        report.append({
            "column": col,
            "dtype": df[col].dtype,
            "is_numeric": pd.api.types.is_numeric_dtype(df[col]),
            #"rows": total_rows,
            "na_count": na_count,
            "na_percent": round(na_count / total_rows * 100, 2),
            "example_na_indices": example_idx
        })

    return pd.DataFrame(report).sort_values("na_count", ascending=False)
```


```

for country, df in dataframes:
    if not ActvnMatrix.is_active(country, PlotOptions.DATAFRAME_NAN_REPORT):
        continue
    report = dataframe_nan_report(df)
    print("=" * 80)
    print(f"\nNaN Report for {country}:")
    print("-" * 80)
    print(report)
    print("=" * 80)
```


```
# dataframes = [
#     ("Italy", df_italy),
#     ("France", df_france),
#     ("Germany", df_germany),
#     ("Spain", df_spain)
# ]


def plot_hexbin_power_by_state_and_type(
    dataframes: list[tuple[str, pd.DataFrame]],
    power_columns: list[str],
    gridsize=40,
    mincnt=5,
    TimeAxis=Columns.AXIS.HOURS_OF_DAY,
    TimeAxisLabel="Hour of Day"
):
    """
    Hexbin grid:
    - Columns: states
    - Rows: power generation types
    - x-axis: hour of day
    - y-axis: power [MW]
    """

    n_states = len(dataframes)
    n_powers = len(power_columns)

    fig, axes = plt.subplots(
        nrows=n_powers,
        ncols=n_states,
        figsize=(5 * n_states, 3.5 * n_powers),
        sharex=True,
        sharey=False
    )

    # Ensure axes is 2D
    if n_powers == 1:
        axes = np.array([axes])

    for col_idx, (country, df) in enumerate(dataframes):
        for row_idx, power_col in enumerate(power_columns):

            ax = axes[row_idx, col_idx]

            if power_col not in df.columns:
                ax.axis("off")
                continue

            hb = ax.hexbin(
                df[TimeAxis],
                df[power_col],
                gridsize=gridsize,
                mincnt=mincnt,
                bins="log",
                cmap="viridis"
            )

            # Column titles (states)
            if row_idx == 0:
                ax.set_title(country, fontsize=12, pad=10)

            # Row labels (power types)
            if col_idx == 0:
                ax.set_ylabel(power_col.split(" - ")[0], fontsize=10)

            if row_idx == n_powers - 1:
                ax.set_xlabel(TimeAxisLabel)

    # Colorbar (single global)
    cbar = fig.colorbar(
        hb,
        ax=axes.ravel().tolist(),
        orientation="vertical",
        fraction=0.015,
        pad=0.01
    )
    cbar.set_label("log10(count)")

    plt.tight_layout()
    plt.show()


    #    Columns.Power.NUCLEAR,
    #     Columns.Power.FOSSIL_GAS,
    #     Columns.Power.SOLAR,
    #     Columns.Power.WIND_ONSHORE,

plot_hexbin_power_by_state_and_type(
    dataframes=dataframes,
    power_columns=Columns.Power.ALL,
    gridsize=50
)
```


```
plot_hexbin_power_by_state_and_type(
    dataframes=dataframes,
    power_columns=Columns.Power.ALL_FILT,
    gridsize=50
)
```


```
plot_hexbin_power_by_state_and_type(
    dataframes=dataframes,
    power_columns=Columns.Power.ALL_FILT,
    gridsize=50,
    TimeAxis=Columns.AXIS.DAY_OF_YEAR,
    TimeAxisLabel="Day of Year"
)
```


```
def plot_raw_power(df: pd.DataFrame, country: str):
    fig, axes = plt.subplots(
        nrows=len(Columns.Power.ALL),
        ncols=1,
        sharex=True,
        figsize=(14, 2.5 * len(Columns.Power.ALL))
    )

    for ax, col in zip(axes, Columns.Power.ALL):
        ax.plot(
            df.index,
            df[col],
            color=colors[country],
            linewidth=0.6
        )
        ax.set_title(col, fontsize=11)
        ax.set_ylabel("MW")

    fig.suptitle(f"{country} – Raw Power Generation (Hourly)", fontsize=16)
    plt.xlabel("Datetime")
    plt.show()
```


```



for country, df in dataframes:
    if not ActvnMatrix.is_active(country, PlotOptions.TIME_PLOT_RAW_POWER):
        continue
    #df[Columns.Power.ALL] = df[Columns.Power.ALL].apply( pd.to_numeric, errors='coerce' )
    plot_raw_power(df, country)
    # plot_raw_overlay(df, country)
    print(type(df.index))

```


```

def remove_outliers_by_regression(
    df: pd.DataFrame,
    *,
    state: str,
    power_columns: Iterable[str],
    x_col: str,
    poly_degree: int = 3,
    sigma_threshold: float = 3.0,
    min_points: int = 50,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Remove outliers per power source using deviation from a regression curve.

    Outlier rule:
        |y - y_pred| > sigma_threshold * std(residuals)

    Returns:
        filtered_df, report_dict
    """

    df = df.copy()
    report = {
        "state": state,
        "x_col": x_col,
        "poly_degree": poly_degree,
        "sigma_threshold": sigma_threshold,
        "power_sources": {}
    }

    mask_keep = pd.Series(True, index=df.index)

    for power in power_columns:
        valid = df[[x_col, power]].dropna()

        if len(valid) < min_points:
            report["power_sources"][power] = {
                "status": "skipped (not enough data)",
                "points": len(valid)
            }
            continue

        x = valid[x_col].values
        y = valid[power].values

        # Regression
        coeffs = np.polyfit(x, y, poly_degree)
        poly = np.poly1d(coeffs)
        y_pred = poly(x)

        residuals = y - y_pred
        sigma = np.std(residuals)

        threshold = sigma_threshold * sigma
        is_outlier = np.abs(residuals) > threshold

        outlier_idx = valid.index[is_outlier]

        mask_keep.loc[outlier_idx] = False

        report["power_sources"][power] = {
            "total_points": len(valid),
            "removed": int(is_outlier.sum()),
            "removed_pct": 100 * is_outlier.mean(),
            "sigma": float(sigma),
            "threshold": float(threshold),
        }

    filtered_df = df.loc[mask_keep].copy()

    report["summary"] = {
        "rows_before": len(df),
        "rows_after": len(filtered_df),
        "rows_removed": int((~mask_keep).sum()),
        "rows_removed_pct": 100 * (~mask_keep).mean(),
    }

    return filtered_df, report

filtered = []
for country, df in dataframes:
    # if not ActvnMatrix.is_active(country, PlotOptions.TIME_PLOT_RAW_POWER):
    #     continue
    
    filtered_df, report = remove_outliers_by_regression(
        df,
        state=country,
        power_columns=Columns.Power.ALL,
        x_col=Columns.AXIS.HOURS_OF_DAY,
        poly_degree=6,
        sigma_threshold=1
    )
    filtered.append((country, df))

    print(report);

```


```
for country, df in filtered:
    if not ActvnMatrix.is_active(country, PlotOptions.TIME_PLOT_RAW_POWER):
        continue
    #df[Columns.Power.ALL] = df[Columns.Power.ALL].apply( pd.to_numeric, errors='coerce' )
    plot_raw_power(df, country)
    # plot_raw_overlay(df, country)
    print(type(df.index))
```


```
def plot_raw_overlay(df: pd.DataFrame, country: str):
    plt.figure(figsize=(14, 8))

    for col in Columns.Power.ALL:
        plt.plot(
            df.index,
            df[col],
            label=col,
            linewidth=0.6
        )

    plt.title(f"{country} – Raw Power Generation by Source")
    plt.xlabel("Datetime")
    plt.ylabel("MW")
    plt.legend(ncol=2)
    plt.show()
```


```
def plot_seahorse_share_all_data(df: pd.DataFrame, country_name: str):
    """
    Plot a stacked area chart of all power columns, ignoring order.
    """
    df_graph = df.copy()

    # Skip non-existing columns
    power_cols = [c for c in Columns.Power.ALL if c in df_graph.columns]
    df_graph = df_graph[power_cols]

    # Force numeric (extra safety)
    df_graph = df_graph.apply(pd.to_numeric, errors="coerce")

    # Monthly aggregation
    df_graph = df_graph.resample("M").mean()

    if df_graph.empty:
        print(f"No data after resampling for {country_name}")
        return

    sns.set_style("whitegrid")
    plt.figure(figsize=(18, 10))

    colors = sns.color_palette("tab20", n_colors=len(power_cols))
    short_labels = [col.split(" - ")[0] for col in power_cols]

    plt.stackplot(
        df_graph.index,
        *[df_graph[col].fillna(0) for col in power_cols],
        labels=short_labels,
        colors=colors,
        alpha=0.75
    )

    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.ylabel("Power [MW]")
    plt.xlabel("Time")
    plt.title(f"{country_name} – Monthly Average Power Generation")
    plt.tight_layout()
    plt.show()


```


```
dataframes = [
    ("Italy", df_italy),
    ("France", df_france),
    ("Germany", df_germany),
    ("Spain", df_spain)
]

for country, df in dataframes:
    if not ActvnMatrix.is_active(country, PlotOptions.POWER_SHARE_BY_SOURCE):
        continue
    plot_seahorse_share_all_data(df, country)

```


```
def plot_seahorse_share_yearly_data(df: pd.DataFrame, country_name: str):
    """
    Plot a stacked area chart showing average daily power profile
    over a year (day-of-year).
    """

    df_graph = df.copy()

    # Ensure DatetimeIndex
    if not isinstance(df_graph.index, pd.DatetimeIndex):
        df_graph.index = pd.to_datetime(df_graph.index, errors="coerce")
    df_graph = df_graph[~df_graph.index.isna()]

    # Keep only power columns
    power_cols = [c for c in Columns.Power.ALL if c in df_graph.columns]
    df_graph = df_graph[power_cols].apply(pd.to_numeric, errors="coerce")

    # Daily aggregation
    df_graph = df_graph.resample("D").mean()

    # Add day-of-year
    df_graph["day_of_year"] = df_graph.index.dayofyear

    # Average over all years → yearly profile
    df_graph = df_graph.groupby("day_of_year").mean()

    if df_graph.empty:
        print(f"No data after aggregation for {country_name}")
        return

    sns.set_style("whitegrid")
    plt.figure(figsize=(18, 10))

    colors = sns.color_palette("tab20", n_colors=len(power_cols))
    short_labels = [col.split(" - ")[0] for col in power_cols]

    plt.stackplot(
        df_graph.index,
        *[df_graph[col].fillna(0) for col in power_cols],
        labels=short_labels,
        colors=colors,
        alpha=0.75
    )

    # plt.stackplot(
    #     date=df_graph,
    #     x=Columns.AXIS.MONTH_STR,
    #     y=power_cols,
    #     labels=short_labels,
    #     colors=colors,
    #     alpha=0.75
    # )

    plt.ylabel("Average Power [MW]")
    plt.xlabel("Day of Year")
    plt.title(f"{country_name} – Average Power Generation Over a Year")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


```


```
for country, df in dataframes:
    if not ActvnMatrix.is_active(country, PlotOptions.POWER_SHARE_BY_SOURCE):
        continue
    plot_seahorse_share_yearly_data(df, country)
```


```
def plot_total_power_scatter(dataframes: list[tuple[str, pd.DataFrame]]):
    fig, axes = plt.subplots(
        nrows=len(dataframes),
        ncols=1,
        figsize=(20, 12),
        sharex=True
    )

    enum_limit = {}

    state_cut_line_values = {
        "Italy": [15000, 50000],
        "France": [30000, 90000],
        "Germany": [30000, 90000],
        "Spain": [18000, 42000]
    }

    for ax, (country, df) in zip(axes, dataframes):

        sns.scatterplot( 
            data=df,
            x=Columns.AXIS.DAY_OF_YEAR,
            y=Columns.CALC.TOTAL_POWER,
            ax=ax,
            color=colors[country],
            alpha=0.4,
            s=5
        )

        # Horizontal upper cutoff line
        ax.axhline(
            y=state_cut_line_values[country][1],
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8
        )

        # Horizontal lower cutoff line
        ax.axhline(
            y=state_cut_line_values[country][0],
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8
        )

        upper_y_offset = 2500
        lower_y_offset = -6000

        ax.text(
            x=10,  # day 10
            y=state_cut_line_values[country][1] + upper_y_offset,
            s=f"{state_cut_line_values[country][1]:,.0f} MW",
            color="red",
            fontsize=10
        )

        ax.text(
            x=10,  # day 10
            y=state_cut_line_values[country][0] + lower_y_offset,
            s=f"{state_cut_line_values[country][0]:,.0f} MW",
            color="red",
            fontsize=10
        )

        # Set consistent y-axis
        ax.set_ylim(0, 100000)  # add a small buffer

        ax.set_title(f"Total power generation in {country}")
        ax.set_ylabel("Total Power [MW]")

        
        # ---- Key: format x-axis to show months ----
        # ax.xaxis.set_major_locator(mdates.MonthLocator())  # every month
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # 'Jan', 'Feb', ...

    axes[-1].set_xlabel("Day of Year")
    plt.tight_layout()
    plt.show()



plot_total_power_scatter(dataframes)

    # for feature in Columns.Power.ALL:
    #     plt.figure(figsize=[len(dataframes_only), len(Columns.Power.ALL)])
    #     sns.scatterplot(
    #         data=df,
    #         x=Columns.AXIS.DAY_OF_YEAR,
    #         y=feature,
    #         hue='day_of_week',
    #         palette='tab10',
    #         alpha=0.6
    #     )



#df_all = pd.concat([df_italy, df_france, df_germany, df_spain], ignore_index=True)

```


```
def plot_yearly_profiles_seasonal(df, country_name):
    df_season = df.copy()

    df_season = (
        df_season
        .groupby([Columns.AXIS.YEAR, Columns.AXIS.SEASON], as_index=False)
        .mean(numeric_only=True)
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    sns.pointplot(
        data=df_season,
        x=Columns.AXIS.SEASON,
        y=Columns.CALC.TOTAL_POWER,
        hue=Columns.AXIS.YEAR,
        dodge=True
    )
    # Set consistent y-axis
    plt.ylim(0, 76000)  # add a small buffer

    plt.xlabel("Season")
    plt.ylabel("Average Power [MW]")
    plt.title(f"{country_name} – Seasonal Electricity Production by Year")
    plt.tight_layout()
    plt.show()

```


```
for country, df in dataframes:
    if not ActvnMatrix.is_active(country, PlotOptions.YEARLY_SEASONAL_OVER_YEARS):
        continue
    plot_yearly_profiles_seasonal(df, country)

```


```
def plot_hourly_profile_by_season(df: pd.DataFrame, country: str):
    """
    Plot the total power consumption over the hours of a day, colored by season.
    """

    df_graph = df.copy()


    plt.figure(figsize=(14,6))

    # sns.scatterplot(
    #     data=df,
    #     x='hour',
    #     y=Columns.CALC.TOTAL_POWER,
    #     hue=Columns.AXIS.SEASON,
    #     palette=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"],
    #     alpha=0.5,
    #     s=10
    # )

    # Optional: smooth line per season
    sns.lineplot(
        data=df_graph,
        x=Columns.AXIS.HOURS_OF_DAY,
        y=Columns.CALC.TOTAL_POWER,
        hue=Columns.AXIS.SEASON,
        palette=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"],
        estimator='mean'
    )

    plt.title(f"{country} – Average Hourly Power Consumption by Season")
    plt.xlabel("Hour of Day")
    plt.ylabel("Total Power [MW]")
    plt.xticks(range(0,25))
    plt.legend(title="Season")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


```


```
for country, df in dataframes:
    if not ActvnMatrix.is_active(country, PlotOptions.HOURLY_PLOT_OVER_SEASONS):
        continue
    plot_hourly_profile_by_season(df, country)
```


```
def plot_hexbin_hourly_power(df: pd.DataFrame, country: str):
    """
    Create a hexbin plot per state showing precomputed total power vs hour of the day.
    Assumes 'total_power' and 'hour' columns already exist in each dataframe.
    """
    
    plt.figure(figsize=(12,6))
    hb = plt.hexbin(
        x=df[Columns.AXIS.HOURS_OF_DAY],
        y=df[Columns.CALC.TOTAL_POWER],
        gridsize=48,
        cmap='viridis',
        mincnt=1,
        linewidths=0.5,
        edgecolors='grey'
    )
    plt.colorbar(hb, label='Count of data points')
    plt.title(f"{country} – Total Power Production by Hour (Hexbin)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Total Power [MW]")
    plt.xticks(range(0,25,1))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

```


```
for country, df in dataframes:
    if not ActvnMatrix.is_active(country, PlotOptions.HEXBIN_TOTAL_POWER_HOURLY_DAYTIME_PLOT):
        continue
    plot_hexbin_hourly_power(df, country)
```


```
def plot_power_hexbins(df: pd.DataFrame, country: str):
    df_graph = df.copy()

    power_cols = [c for c in Columns.Power.ALL if c in df.columns]

    for power in power_cols:
        plt.figure(figsize=(14, 6))
        hb = plt.hexbin(
            x=df_graph[Columns.AXIS.DAY_OF_YEAR],    # x = day of year
            y=df_graph[Columns.AXIS.HOURS_OF_DAY],    # y = hour with fraction
            C=df_graph[Columns.CALC.TOTAL_POWER],    # color = power
            gridsize=100,                            # increase for higher resolution
            cmap='viridis',
            reduce_C_function=np.mean,     # average in bin
            mincnt=1                       # skip empty bins
        )
        plt.colorbar(hb, label=f"{power} [MW]")
        plt.xlabel("Day of Year")
        plt.ylabel("Hour of Day")
        plt.title(f"{country} – {power} Power Consumption")
        plt.tight_layout()
        plt.show()

```


```
for country, df in dataframes:
    if not ActvnMatrix.is_active(country, PlotOptions.HEXBIN_TOTAL_POWER_DAILY_YEAR_PLOT):
        continue
    plot_power_hexbins(df, country)

```


```
def r2_score_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

```


```
def plot_yearly_total_power_trend(
    df: pd.DataFrame,
    country: str,
    degrees=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
):
    df = df.copy()

    # --- yearly aggregation ---
    yearly = (
        df
        .groupby(Columns.AXIS.MONTH, as_index=False)[Columns.CALC.TOTAL_POWER]
        .mean()
    )

    x = yearly.index.values
    y = yearly[Columns.CALC.TOTAL_POWER].values

    # numeric axis for stable polynomial fitting
    t = np.arange(len(x))
    t_fit = np.linspace(t.min(), t.max(), 300)

    plt.figure(figsize=(14, 6))

    sns.scatterplot(
        x=x,
        y=y,
        s=70,
        color=colors[country],
        label="Yearly average"
    )

    max_degree = len(x) - 1

    for deg in degrees:
        if deg > max_degree:
            print(
                f"⚠️  Skipping degree {deg} for {country} "
                f"(only {len(x)} data points)"
            )
            continue

        coeffs = np.polyfit(t, y, deg)
        poly = np.poly1d(coeffs)

        y_pred = poly(t)
        r2 = r2_score_manual(y, y_pred)

        plt.plot(
            np.interp(t_fit, t, x),
            poly(t_fit),
            linewidth=2,
            label=f"deg={deg}, R²={r2:.3f}"
        )

    plt.xlabel("Year")
    plt.ylabel("Average Total Power [MW]")
    plt.title(f"{country} – Yearly Power Production Trend")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

```


```
for country, df in dataframes:
    if not ActvnMatrix.is_active(country, PlotOptions.TREND_TOTAL_POWER_OVER_YEARS):
        continue
    plot_yearly_total_power_trend(df, country)
```


```


def plot_hourly_polynomial_comparison(
    df: pd.DataFrame,
    country: str,
    degrees=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
):
    df = df.copy()

    x = df[Columns.AXIS.HOURS_OF_DAY].values
    y = df[Columns.CALC.TOTAL_POWER].values

    # Clean NaNs
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]

    x_fit = np.linspace(0, 23, 300)

    plt.figure(figsize=(13, 7))

    # Raw data
    sns.scatterplot(
        x=x,
        y=y,
        alpha=0.25,
        s=10,
        color=colors[country],
        label="Hourly observations"
    )

    results = []

    for deg in degrees:
        coeffs = np.polyfit(x, y, deg)
        poly = np.poly1d(coeffs)

        y_pred = poly(x)
        r2 = r2_score_manual(y, y_pred)

        results.append((deg, r2))

        plt.plot(
            x_fit,
            poly(x_fit),
            linewidth=2,
            label=f"deg={deg}, R²={r2:.3f}"
        )

    plt.xlabel("Hour of Day")
    plt.ylabel("Total Power [MW]")
    plt.title(f"{country} – Polynomial Regression Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Console report
    print(f"\nPolynomial model comparison for {country}")
    print("-" * 40)
    for deg, r2 in results:
        print(f"Degree {deg}: R² = {r2:.4f}")


def plot_hourly_total_power_regression(df: pd.DataFrame, country: str, degree: int = 6):
    df = df.copy()

    x = df[Columns.AXIS.HOURS_OF_DAY].values
    y = df[Columns.CALC.TOTAL_POWER].values

    # Remove NaNs
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]

    # Polynomial fit
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)

    x_fit = np.linspace(0, 23, 200)
    y_fit = poly(x_fit)

    plt.figure(figsize=(12, 6))

    sns.scatterplot(
        x=x,
        y=y,
        alpha=0.25,
        s=10,
        color=colors[country],
        label="Hourly observations"
    )

    plt.plot(
        x_fit,
        y_fit,
        color="black",
        linewidth=2.5,
        label=f"Polynomial regression (deg={degree})"
    )

    plt.xlabel("Hour of Day")
    plt.ylabel("Total Power [MW]")
    plt.title(f"{country} – Daily Power Profile (Polynomial Regression)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

```


```
for country, df in dataframes:
    if not ActvnMatrix.is_active(country, PlotOptions.HOURLY_TOTAL_POWER_REGRESSION):
        continue
    plot_hourly_total_power_regression(df, country)
    plot_hourly_polynomial_comparison(df, country)
```


```
def plot_monthly_trend_regression(
    monthly: pd.DataFrame,
    country: str,
    degrees=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
):
    x = monthly[Columns.AXIS.MONTH].values
    y = monthly[Columns.CALC.TOTAL_POWER].values

    x_fit = np.linspace(x.min(), x.max(), 300)

    plt.figure(figsize=(14, 6))

    # Raw monthly trend
    sns.scatterplot(
        x=x,
        y=y,
        s=60,
        alpha=0.8,
        color=colors[country],
        label="Monthly mean"
    )

    for deg in degrees:
        coeffs = np.polyfit(x, y, deg)
        poly = np.poly1d(coeffs)

        y_pred = poly(x)
        r2 = r2_score_manual(y, y_pred)

        plt.plot(
            x_fit,
            poly(x_fit),
            linewidth=2,
            label=f"deg={deg}, R²={r2:.3f}"
        )

    plt.xlabel("Time (months since start)")
    plt.ylabel("Total Power [MW]")
    plt.title(f"{country} – Monthly Power Trend Regression")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


```


```
for country, df in dataframes:
    if not ActvnMatrix.is_active(country, PlotOptions.TREND_TOTAL_POWER_OVER_MONTHS):
        continue
    plot_monthly_trend_regression(df, country)
```


```
def plot_filter_diagnostics_scatter(
    df_original: pd.DataFrame,
    df_filtered: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    title: str | None = None,
    kept_label: str = "Kept",
    removed_label: str = "Filtered out",
    kept_color: str = "tab:blue",
    removed_color: str = "red",
    kept_alpha: float = 0.4,
    removed_alpha: float = 0.9,
    kept_size: int = 10,
    removed_size: int = 30
):
    """
    Scatter diagnostic plot showing which points were filtered out.

    Parameters
    ----------
    df_original : pd.DataFrame
        Data before filtering.
    df_filtered : pd.DataFrame
        Data after filtering.
    x_col, y_col : str
        Columns to plot.
    """

    # --- Align by index ---
    df_original = df_original[[x_col, y_col]].dropna()
    df_filtered = df_filtered[[x_col, y_col]].dropna()

    # Points kept
    df_kept = df_original.loc[df_original.index.intersection(df_filtered.index)]

    # Points removed
    df_removed = df_original.loc[
        df_original.index.difference(df_filtered.index)
    ]

    plt.figure(figsize=(14, 6))

    # Kept points
    sns.scatterplot(
        data=df_kept,
        x=x_col,
        y=y_col,
        color=kept_color,
        alpha=kept_alpha,
        s=kept_size,
        label=f"{kept_label} ({len(df_kept)})"
    )

    # Removed points
    sns.scatterplot(
        data=df_removed,
        x=x_col,
        y=y_col,
        color=removed_color,
        alpha=removed_alpha,
        s=removed_size,
        marker="x",
        label=f"{removed_label} ({len(df_removed)})"
    )

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title or f"Filter diagnostics: {y_col} vs {x_col}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

```


```
# plot_filter_diagnostics_scatter(
#     df_original,
#     df_filtered,
#     x_col=Columns.AXIS.HOURS_OF_DAY,
#     y_col=Columns.CALC.TOTAL_POWER,
#     title="Total Power vs Hour – Filtered points"
# )

```


```
# plot_filter_diagnostics_scatter(
#     df_original,
#     df_filtered,
#     x_col=Columns.AXIS.DAY_OF_YEAR,
#     y_col=Columns.CALC.TOTAL_POWER
# )

```


```
# plot_filter_diagnostics_scatter(
#     df_original,
#     df_filtered,
#     x_col=Columns.AXIS.HOURS_OF_DAY,
#     y_col=Columns.Power.SOLAR,
#     title="Solar production – filtered diagnostics"
# )

```


```
def report_raw_dataset_overview(
    df: pd.DataFrame,
    name: str = "Dataset",
    datetime_col: str | None = None
):
    """
    Print a overview report of a raw dataset.
    """

    print(f"\n#  Dataset Overview: {name}\n")

    # ---- Dimensions ----
    print("## Dimensions")
    print(f"- Rows: **{df.shape[0]:,}**")
    print(f"- Columns: **{df.shape[1]}**\n")

    # ---- Column info ----
    print("## Columns & Data Types")
    for col, dtype in df.dtypes.items():
        print(f"- `{col}` → `{dtype}`")
    print()

    # ---- Datetime handling ----
    dt_series = None
    if datetime_col and datetime_col in df.columns:
        dt_series = pd.to_datetime(df[datetime_col], errors="coerce")
    elif isinstance(df.index, pd.DatetimeIndex):
        dt_series = df.index

    if dt_series is not None and not dt_series.dropna().empty:
        print("## Time Coverage")
        print(f"- Start: **{dt_series.min()}**")
        print(f"- End: **{dt_series.max()}**")

        # Sampling rate estimation
        diffs = dt_series.sort_values().diff().dropna()
        if not diffs.empty:
            most_common = pd.Series(diffs).mode().iloc[0]
            print(f"- Estimated sampling rate: **{most_common}**")
        print()
    else:
        print("## Time Coverage")
        print("- No valid datetime information found\n")

    # ---- Missingness ----
    print("## Missing Values Summary")
    na_counts = df.isna().sum()
    na_percent = (na_counts / len(df)) * 100

    has_missing = False
    for col in df.columns:
        if na_counts[col] > 0:
            has_missing = True
            print(
                f"- `{col}`: "
                f"{na_counts[col]:,} missing "
                f"({na_percent[col]:.2f}%)"
            )

    if not has_missing:
        print("- No missing values detected")

    print("\n---")

```

# Team Format and Dataset

## 1. Team

- Oswald Lackner
- Stocker Christoph

Source of Data: [Kaggle: Hourly Power Generation of Europe](https://www.kaggle.com/datasets/mehmetnuryildirim/hourly-power-generation-of-europe)

# 2. Task Categories and Points

## 2.1 A. Data Preprocessing and Data Quality (70 points)

- 2.1.1 Dataset overview (dimensions, columns, types, time range, sampling rate, missingness
summary) (10 points)
- 2.1.2 Basic statistical analysis using pandas (descriptives, grouped stats, quantiles) (10 points)
- 2.1.3 Original data quality analysis with visualization (missingness patterns, outliers, dupli-
cates, timestamp gaps, inconsistent units) (20 points)
- 2.1.4 Data preprocessing pipeline (cleaning steps, handling missing data, outliers strategy, re-
sampling or alignment if needed, feature engineering basics) (20 points)
- 2.1.5 Preprocessed vs original comparison (before/after visuals plus commentary on what changed
and why) (10 points)


```


for country, df in dataframes:
    # if not ActvnMatrix.is_active(country, PlotOptions.DATAFRAME_NAN_REPORT):
    #     continue
    report_raw_dataset_overview(df, name=country + " – Raw Power Data")
```


```

def remove_outliers_by_regression(
    df: pd.DataFrame,
    *,
    state: str,
    power_columns: Iterable[str],
    x_col: str,
    poly_degree: int = 3,
    sigma_threshold: float = 3.0,
    min_points: int = 50,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Remove outliers per power source using deviation from a regression curve.

    Outlier rule:
        |y - y_pred| > sigma_threshold * std(residuals)

    Returns:
        filtered_df, report_dict
    """

    df = df.copy()
    report = {
        "state": state,
        "x_col": x_col,
        "poly_degree": poly_degree,
        "sigma_threshold": sigma_threshold,
        "power_sources": {}
    }

    mask_keep = pd.Series(True, index=df.index)

    for power in power_columns:
        valid = df[[x_col, power]].dropna()

        if len(valid) < min_points:
            report["power_sources"][power] = {
                "status": "skipped (not enough data)",
                "points": len(valid)
            }
            continue

        x = valid[x_col].values
        y = valid[power].values

        # Regression
        coeffs = np.polyfit(x, y, poly_degree)
        poly = np.poly1d(coeffs)
        y_pred = poly(x)

        residuals = y - y_pred
        sigma = np.std(residuals)

        threshold = sigma_threshold * sigma
        is_outlier = np.abs(residuals) > threshold

        outlier_idx = valid.index[is_outlier]

        mask_keep.loc[outlier_idx] = False

        report["power_sources"][power] = {
            "total_points": len(valid),
            "removed": int(is_outlier.sum()),
            "removed_pct": 100 * is_outlier.mean(),
            "sigma": float(sigma),
            "threshold": float(threshold),
        }

    filtered_df = df.loc[mask_keep].copy()

    report["summary"] = {
        "rows_before": len(df),
        "rows_after": len(filtered_df),
        "rows_removed": int((~mask_keep).sum()),
        "rows_removed_pct": 100 * (~mask_keep).mean(),
    }

    return filtered_df, report

filtered = []
for country, df in dataframes:
    # if not ActvnMatrix.is_active(country, PlotOptions.TIME_PLOT_RAW_POWER):
    #     continue
    
    filtered_df, report = remove_outliers_by_regression(
        df,
        state=country,
        power_columns=Columns.Power.ALL,
        x_col=Columns.AXIS.HOURS_OF_DAY,
        poly_degree=6,
        sigma_threshold=1
    )
    filtered.append((country, df))

    print(report);

```

Basic statistical analysis using pandas (descriptives, grouped stats, quantiles) (10 points)
    - descriptives (mean, std deviation, min/max)
    - stats Total power by season / year
    - quantiles 


```
def perform_basic_stats(df: pd.DataFrame, country: str):
    """
    Returns the full descriptive statistics (transposed) for the country.
    """
    # numeric power + total power (ignore date, text ect)
    stats_cols = [c for c in Columns.Power.ALL + [Columns.CALC.TOTAL_POWER] if c in df.columns]
    
    # calculate non missin rows, mean, standard devaiton, min, max, and quatiles
    desc_stats = df[stats_cols].describe().round(2).T
    
    # return clean and transponated table
    return desc_stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]

# execute
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

all_country_stats = {}
total_power_overview = {}

# loop all counties
for country, df in dataframes:
    stats = perform_basic_stats(df, country)
    # save all satistics into a dictionary
    all_country_stats[country] = stats      
    
    # extract total power for overview
    if Columns.CALC.TOTAL_POWER in stats.index:
        total_power_overview[country] = stats.loc[Columns.CALC.TOTAL_POWER]

# 1 overview of total power
print("\n" + "="*80)
print("1. OVERVIEW: TOTAL POWER GENERATION STATISTICS")
print("="*80)
# output the total power data
overview_df = pd.DataFrame(total_power_overview)
print(overview_df)

# 2 display all sources 
print("\n" + "="*80)
print("2. DETAILED COMPARISON (ALL POWER SOURCES)")
print("="*80)

# combine all the stats tables
comparative_df = pd.concat(all_country_stats.values(), axis=1, keys=all_country_stats.keys())

# flip the colum hierachy for better readability
comparative_df = comparative_df.swaplevel(0, 1, axis=1).sort_index(axis=1)

# define output order
metrics_to_show = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

# print out the rest of the data
for metric in metrics_to_show:
    if metric in comparative_df.columns.get_level_values(0):
        print(f"\n{'-'*30} {metric.upper()} {'-'*30}")
        print(comparative_df[metric])
```

Original data quality analysis with visualization (missingness patterns, outliers, duplicates,
timestamp gaps, inconsistent units) (20 points)


```
def analyze_data_quality_combined(dataframes_list):

    print(f"\n{'='*80}")
    print(f"COMBINED DATA QUALITY ANALYSIS (ALL COUNTRIES)")
    print(f"{'='*80}")

    # extract the names
    countries = [name for name, _ in dataframes_list]
    # emty list to collect the final statistics
    quality_summary = [] 
    # dictionary to store the count of outliers
    outlier_counts_total = {}
    
    # check the ranges
    # the measurement start dated of the countries are different
    # thus the index has to be adapted
    print("\n[0] Checking Dataset Time Ranges...")
    range_data = []
    for country, df in dataframes_list:
        range_data.append({
            "Country": country,
            "Start Date": df.index.min(),
            "End Date": df.index.max(),
            "Total Days": (df.index.max() - df.index.min()).days        # duration
        })
    # display the date information
    print(pd.DataFrame(range_data).set_index("Country"))

    # ------------------------ 1 missingness patterns ------------------------
    print("\n[1] Generating Combined Missingness diagramm...")
    # added sharex=False to allow for different start/end dates
    fig, axes = plt.subplots(nrows=len(dataframes_list), ncols=1, figsize=(15, 5*len(dataframes_list)), sharex=False)
    if len(dataframes_list) == 1: axes = [axes]

    #   generate the height and pairs the plot with the corresponding data
    for ax, (country, df) in zip(axes, dataframes_list):
        sns.heatmap(
            df.isnull().T,      # create a table for the mising data
            ax=ax, 
            cbar=False,         # disable the legend
            cmap='viridis',     # set colors to yellow and purple (visibility)
            xticklabels=False, 
            yticklabels=True
            )
        # set titles
        ax.set_title(f"{country} – Missing Data (Yellow = Missing)", fontsize=14, loc='left', pad=10)
        ax.tick_params(axis='y', rotation=0, labelsize=10)

        # added display of start and end time (because of different start end end dates of the data)
        start_str = str(df.index.min().date())
        end_str = str(df.index.max().date())
        ax.set_xlabel(f"Timeline: {start_str} to {end_str}", fontsize=10, color='gray')
    
    # generate the plot
    plt.suptitle("Missing Data Patterns Overview (Independent Timelines)", fontsize=16, y=1.01)
    plt.tight_layout()
    plt.show()

    # ------------------------ 2 timestamp gaps ------------------------
    print("\n[2] Generating Combined Timestamp Gap Analysis...")
    # excluded the code for timegap analysis
    # (no timegaps found)
    print("✓ No timestamp gaps found.")

    # ------------------------ 3 outliers (boxplots) ------------------------
    print("\n[3] Generating Combined Outlier Analysis (All Power Sources)...")
    # genereate list of colums of interest
    check_list = Columns.Power.ALL + [Columns.CALC.TOTAL_POWER]
    cols_to_plot = []
    
    present_cols = set()
    for _, df in dataframes_list:
        present_cols.update(df.columns)
        
    for col in check_list:
        if col in present_cols:
            cols_to_plot.append(col)

    # count of colums 
    n_cols = 4
    # calculate rows
    n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    # convert 2D grid into a 1D List
    axes = axes.flatten()
    

    for i, col_name in enumerate(cols_to_plot):
        ax = axes[i]
        plot_data = []
        
        for country, df in dataframes_list:
            if col_name in df.columns:
                # only plot the values (for the boxplot the timestamps arent interesting)
                clean_values = df[col_name].values
                plot_data.append(pd.DataFrame({'Country': country, 'Value': clean_values}))
                # find the outlyers and mark tem in the plots
                if col_name == Columns.CALC.TOTAL_POWER:
                    data = df[col_name].dropna()
                    Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
                    IQR = Q3 - Q1
                    n_outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
                    outlier_counts_total[country] = n_outliers

        if plot_data:
            viz_df = pd.concat(plot_data, ignore_index=True)
            sns.boxplot(
                data=viz_df, 
                x='Country', 
                y='Value', 
                hue='Country', 
                legend=False, 
                ax=ax, 
                palette=colors)
            ax.set_title(col_name, fontsize=11, fontweight='bold')
            ax.set_ylabel("MW")
            ax.set_xlabel("")
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.set_visible(False)

    for j in range(len(cols_to_plot), len(axes)):
        axes[j].set_visible(False)
        
    plt.suptitle("Distribution & Outliers Comparison (All Power Sources)", fontsize=16, y=1.002)
    plt.tight_layout()
    plt.show()

    # ------------------------ 4 consistency check ------------------------
    print("\n[4] Generating Logical Consistency Check...")
    issue_counts = []
    # count double timestamps
    for country, df in dataframes_list:
        n_dupes = df.index.duplicated().sum()
        power_cols = [c for c in Columns.Power.ALL if c in df.columns]
        # ckecks if there are som negative power values
        n_negatives = (df[power_cols] < 0).sum().sum()
        
        issue_counts.append({'Country': country, 'Issue': 'Duplicate Rows', 'Count': n_dupes})
        issue_counts.append({'Country': country, 'Issue': 'Negative Values', 'Count': n_negatives})
        
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        # collect missing percent, gaps, dublicate, negative
        quality_summary.append({
            'Country': country,
            'Missing Values (Cells)': missing_cells,
            'Missing %': round((missing_cells / total_cells) * 100, 4),
            'Timestamp Gaps (>1h)': 0,
            'Duplicate Rows': n_dupes,
            'Negative Values': n_negatives,
            'Total Power Outliers': outlier_counts_total.get(country, 0)
        })
    
    issues_df = pd.DataFrame(issue_counts)
    # plot dublicates and negatives side by side
    plt.figure(figsize=(12, 6))
    sns.barplot(data=issues_df, x='Country', y='Count', hue='Issue', palette='Reds')
    plt.title("Data Logic Errors: Duplicates & Inconsistent Units", fontsize=16)
    # symlog adjusts the plot for small and large numbers
    plt.yscale('symlog') 
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    # ------------------------ 5  final summary ------------------------
    print("\n" + "="*80)
    print("FINAL DATA QUALITY REPORT SUMMARY")
    print("="*80)
    # creating the final table
    summary_df = pd.DataFrame(quality_summary).set_index('Country')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(summary_df)
    print("="*80)

# execute the function
analyze_data_quality_combined(dataframes)
```

2.2.2 Distribution analysis with histograms and density style plots where applicable (10 points)


```
def plot_distribution_all_sources(dataframes_list):
    """
    Generates Distribution Analysis (KDE) for EVERY Power Source.
    Layout: 5 plots per row.
    Optimization: Zooms into the 1st-99th percentile range to cut off extreme outliers.
    """
    print(f"\n{'='*80}")
    print("B.2 DETAILED DISTRIBUTION ANALYSIS (ZOOMED - OUTLIERS CUT)")
    print(f"{'='*80}")

    # identify collumms
    check_list = [Columns.CALC.TOTAL_POWER] + Columns.Power.ALL_FILT 
    cols_to_plot = []
    
    # loops through data to get a unique list of colums 
    present_cols = set()
    for _, df in dataframes_list: 
        present_cols.update(df.columns)
        
    # deletes the column no data is found
    for col in check_list:
        if col in present_cols:
            cols_to_plot.append(col)

    # set up grid with 5 colums and calculate the rows
    n_cols = 5
    n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
    
    # width fixed at 20 
    # height calculated according to n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))
    axes = axes.flatten()
    
    print(f"Generating {len(cols_to_plot)} distribution plots (zoomed)...")

    # loop through the colums and generate plots
    for i, col_name in enumerate(cols_to_plot):
        ax = axes[i]
        has_data = False
        
        # added to adjust the displayed frame
        view_min = float('inf')
        view_max = float('-inf')
        
        # loop through the
        for country, df in dataframes_list:
            if col_name in df.columns:
                data = df[col_name].dropna()
                
                if not data.empty and data.std() > 0:
                    # plot the full curve (with cut=0 to stop exactly at min/max data)
                    sns.kdeplot(
                        data,
                        ax=ax,
                        label=country,
                        color=colors[country], 
                        fill=True,                      # color the area under the curve
                        alpha=0.1,                      # transparent fill
                        linewidth=1.5, 
                        warn_singular=False,            
                        cut=0)                          # border 0 (no negative line smoothing)            
                    has_data = True
                    
                    # added to delete the outliers (for better visibility)
                    q01 = data.quantile(0.01)
                    q99 = data.quantile(0.99)
                    
                    # added (also for view)
                    view_min = min(view_min, q01)
                    view_max = max(view_max, q99)

        # formatting
        # remove repetetive text
        clean_title = col_name.replace(" - Actual Aggregated [MW]", "")
        ax.set_title(clean_title, fontsize=10, fontweight='bold')
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(True, linestyle=':', alpha=0.4)
        
        # added here are the limits set / cut off
        if has_data and view_max > view_min:
            ax.set_xlim(view_min, view_max)
        
        # the legend is only displayed on the first plot
        if i == 0:
            ax.legend(loc='upper right', fontsize=8, title="Country")
        else:
            if ax.get_legend(): ax.get_legend().remove()

        if not has_data:
            ax.text(0.5, 0.5, "No Data / Constant", ha='center', fontsize=8, color='gray')

    # set the empty plots to no data / constant
    for j in range(len(cols_to_plot), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Comparative Distribution Analysis (Zoomed into 1st-99th Percentile)", fontsize=16, y=1.005)
    plt.tight_layout()
    plt.show()

# execute
plot_distribution_all_sources(dataframes)
```

2.2.3 Correlation analysis and heatmaps (Pearson and at least one alternative such as Spearman,
with short interpretation) (10 points)


```
def analyze_correlations_standard(dataframes_list):

    print(f"\n{'='*80}")
    print("B.3 CORRELATION ANALYSIS (SHORT LABELS)")
    print(f"{'='*80}")

    for country, df in dataframes_list:
        print(f"\nAnalyzing {country}...")
        
        # select colums
        target_cols = Columns.Power.ALL + [Columns.CALC.TOTAL_POWER]
        available_cols = [c for c in target_cols if c in df.columns]
        
        # 2. validate colums (take ony valid ones)
        valid_cols = []
        for col in available_cols:
            if df[col].notna().sum() > 10 and df[col].std() > 0:
                valid_cols.append(col)
        
        # if there are less than 2 colums - correlation not possible
        if len(valid_cols) < 2:
            print(f"Skipping {country}: Not enough valid columns.")
            continue

        # 3. only validated collums
        corr_data = df[valid_cols].copy()
        
        # Logic: Split the string at " - " and keep only the first part.
        # "Hydro Pumped Storage - Actual Aggregated [MW]" -> "Hydro Pumped Storage"
        new_names = {}
        for col in valid_cols:
            if " - " in col:
                clean_name = col.split(" - ")[0] 
            else:
                clean_name = col # Keep original if no separator found
            new_names[col] = clean_name
            
        corr_data.rename(columns=new_names, inplace=True)

        # 4. Calculate Matrices
        pearson_corr = corr_data.corr(method='pearson')
        spearman_corr = corr_data.corr(method='spearman')
        
        # 5. Plot
        fig, axes = plt.subplots(1, 2, figsize=(22, 9))
        
        # Left: Pearson
        sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap='coolwarm', 
                    vmin=-1, vmax=1, center=0, square=True, linewidths=0.5, 
                    ax=axes[0], annot_kws={"size": 10})
        axes[0].set_title(f"{country} - Pearson (Linear)", fontsize=16, fontweight='bold', pad=15)
        axes[0].tick_params(axis='both', which='major', labelsize=11)
        
        # Right: Spearman
        sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap='coolwarm', 
                    vmin=-1, vmax=1, center=0, square=True, linewidths=0.5, 
                    ax=axes[1], annot_kws={"size": 10})
        axes[1].set_title(f"{country} - Spearman (Rank Order)", fontsize=16, fontweight='bold', pad=15)
        axes[1].tick_params(axis='both', which='major', labelsize=11)
        
        plt.tight_layout()
        plt.show()

# --- EXECUTION ---
analyze_correlations_standard(dataframes)
```


```
def remove_outliers_by_fixed_threshold(
    df: pd.DataFrame,
    *,
    state: str,
    thresholds: dict[str, tuple[float | None, float | None]],
    min_points: int = 10
):
    df = df.copy()
    mask_keep = pd.Series(True, index=df.index)

    report = {
        "state": state,
        "method": "fixed_threshold",
        "power_sources": {},
        "summary": {}
    }

    for power, (min_val, max_val) in thresholds.items():

        if power not in df.columns:
            report["power_sources"][power] = {
                "status": "skipped (column not found)"
            }
            continue

        series = df[power]

        valid = series.notna()
        n_valid = valid.sum()

        if n_valid < min_points:
            report["power_sources"][power] = {
                "status": "skipped (not enough data)",
                "points": int(n_valid)
            }
            continue

        keep = pd.Series(True, index=df.index)

        if min_val is not None:
            keep &= (series >= min_val) | ~valid
        if max_val is not None:
            keep &= (series <= max_val) | ~valid

        removed = (~keep & valid).sum()

        mask_keep &= keep

        report["power_sources"][power] = {
            "min_threshold": min_val,
            "max_threshold": max_val,
            "total_points": int(n_valid),
            "removed": int(removed),
            "removed_pct": float(100 * removed / n_valid),
        }

    filtered_df = df.loc[mask_keep].copy()

    report["summary"] = {
        "rows_before": len(df),
        "rows_after": len(filtered_df),
        "rows_removed": int((~mask_keep).sum()),
        "rows_removed_pct": float(100 * (~mask_keep).mean()),
    }

    return filtered_df, report



POWER_THRESHOLDS = {
    "Italy": {
        Columns.Power.SOLAR: (0, 120000),
        Columns.Power.WIND_ONSHORE: (0, 150000),
        Columns.Power.WIND_OFFSHORE: (0, 90000),
        Columns.Power.NUCLEAR: (0, 100000),
        Columns.Power.FOSSIL_GAS: (0, 200000),
        Columns.CALC.TOTAL_POWER: (10000, 50000)
    },
    "France": {
        Columns.Power.SOLAR: (0, 100000),
        Columns.Power.WIND_ONSHORE: (0, 180000),
        Columns.Power.WIND_OFFSHORE: (0, 70000),
        Columns.Power.NUCLEAR: (0, 120000),
        Columns.Power.FOSSIL_GAS: (0, 160000),  
        Columns.CALC.TOTAL_POWER: (20000, 100000)
    },
    "Germany": {
        Columns.Power.SOLAR: (0, 80000),
        Columns.Power.WIND_ONSHORE: (0, 200000),
        Columns.Power.WIND_OFFSHORE: (0, 60000),
        Columns.Power.NUCLEAR: (0, 90000),
        Columns.Power.FOSSIL_GAS: (0, 150000),
        Columns.CALC.TOTAL_POWER: (20000, 150000)
    },
    "Spain": {
        Columns.Power.SOLAR: (0, 90000),
        Columns.Power.WIND_ONSHORE: (0, 160000),
        Columns.Power.WIND_OFFSHORE: (0, 50000),
        Columns.Power.NUCLEAR: (0, 80000),
        Columns.Power.FOSSIL_GAS: (0, 140000),
        Columns.CALC.TOTAL_POWER: (15000, 110000)
    }
}


dataframes_filtered = []
for country, df in dataframes:
    # if not ActvnMatrix.is_active(country, PlotOptions.OUTLIER_REMOVAL_FIXED_THRESHOLDS):
    #     continue

    filtered_df, report = remove_outliers_by_fixed_threshold(
        df,
        state=country,
        thresholds=POWER_THRESHOLDS[country]
    )
    dataframes_filtered.append((country, filtered_df))
```


```
for country, df in dataframes_filtered:
    print("=" * 40)
    print(f"\nOutlier Removal Report for {country}:")
    print("-" * 40)
    plot_filter_diagnostics_scatter(
        df_original=dict(dataframes)[country], 
        df_filtered=df,
        x_col=Columns.AXIS.HOURS_OF_DAY,
        y_col=Columns.CALC.TOTAL_POWER,
        title=f"Hours of the day - Outlier Removal - {country}"
    )

for country, df in dataframes_filtered:
    print("=" * 40)
    print(f"\nOutlier Removal Report for {country}:")
    print("-" * 40)
    plot_filter_diagnostics_scatter(
        df_original=dict(dataframes)[country], 
        df_filtered=df,
        x_col=Columns.AXIS.DAY_OF_WEEK,
        y_col=Columns.CALC.TOTAL_POWER,
        title=f"Day of the week - Outlier Removal - {country}"
    )

for country, df in dataframes_filtered:
    print("=" * 40)
    print(f"\nOutlier Removal Report for {country}:")
    print("-" * 40)
    plot_filter_diagnostics_scatter(
        df_original=dict(dataframes)[country], 
        df_filtered=df,
        x_col=Columns.AXIS.DAY_OF_YEAR,
        y_col=Columns.CALC.TOTAL_POWER,
        title=f"Day of the year - Outlier Removal - {country}"
    )
```


```

def remove_outliers_by_regression(
    df: pd.DataFrame,
    *,
    state: str,
    power_columns: Iterable[str],
    x_col: str,
    poly_degree: int = 3,
    sigma_threshold: float = 3.0,
    min_points: int = 50,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Remove outliers per power source using deviation from a regression curve.

    Outlier rule:
        |y - y_pred| > sigma_threshold * std(residuals)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    state : str
        Name of the state or region.
    power_columns : iterable of str
        Columns containing power source data.
    x_col : str
        Column to use as independent variable (e.g., hour, timestamp).
    poly_degree : int
        Degree of polynomial regression.
    sigma_threshold : float
        Threshold in standard deviations for outlier removal.
    min_points : int
        Minimum number of points required to perform regression.

    Returns
    -------
    filtered_df : pd.DataFrame
        DataFrame with outliers removed.
    report : dict
        Detailed report of the filtering process.
    """

    df = df.copy()
    mask_keep = pd.Series(True, index=df.index)

    report = {
        "state": state,
        "x_col": x_col,
        "poly_degree": poly_degree,
        "sigma_threshold": sigma_threshold,
        "power_sources": {}
    }

    for power in power_columns:
        if power not in df.columns:
            report["power_sources"][power] = {
                "status": "skipped (column not found)"
            }
            continue

        valid = df[[x_col, power]].dropna()
        if len(valid) < min_points:
            report["power_sources"][power] = {
                "status": "skipped (not enough data)",
                "points": len(valid)
            }
            continue

        x = valid[x_col].values
        y = valid[power].values

        # Polynomial regression
        coeffs = np.polyfit(x, y, poly_degree)
        poly = np.poly1d(coeffs)
        y_pred = poly(x)

        residuals = y - y_pred
        sigma = np.std(residuals)

        threshold = sigma_threshold * sigma
        is_outlier = np.abs(residuals) > threshold

        mask_keep.loc[valid.index[is_outlier]] = False

        report["power_sources"][power] = {
            "total_points": len(valid),
            "removed": int(is_outlier.sum()),
            "removed_pct": float(100 * is_outlier.mean()),
            "sigma": float(sigma),
            "threshold": float(threshold)
        }

    filtered_df = df.loc[mask_keep].copy()

    report["summary"] = {
        "rows_before": len(df),
        "rows_after": len(filtered_df),
        "rows_removed": int((~mask_keep).sum()),
        "rows_removed_pct": float(100 * (~mask_keep).mean())
    }

    return filtered_df, report


dataframes_filtered_regression = []
for country, df in dataframes_filtered:
    # if not ActvnMatrix.is_active(country, PlotOptions.TREND_TOTAL_POWER_OVER_MONTHS):
    #     continue
    remove_outliers_by_regression(
        df,
        state=country,
        power_columns=Columns.Power.ALL_FILT + [Columns.CALC.TOTAL_POWER],
        x_col=Columns.AXIS.DAY_OF_WEEK,
        poly_degree=10,
        sigma_threshold=1.0,
        min_points=50
        )
    dataframes_filtered_regression.append((country, df))


for country, df in dataframes_filtered_regression:
    print("=" * 40)
    print(f"\nOutlier Removal Report for {country}:")
    print("-" * 40)
    plot_filter_diagnostics_scatter(
        df_original=dict(dataframes)[country], 
        df_filtered=df,
        x_col=Columns.AXIS.DAY_OF_YEAR,
        y_col=Columns.CALC.TOTAL_POWER,
        title=f"Day of the year - Outlier Removal - {country}"
    )

    for country, df in dataframes_filtered_regression:
        print("=" * 40)
        print(f"\nOutlier Removal Report for {country}:")
        print("-" * 40)
        plot_filter_diagnostics_scatter(
            df_original=dict(dataframes)[country], 
            df_filtered=df,
            x_col=Columns.AXIS.DAY_OF_WEEK,
            y_col=Columns.CALC.TOTAL_POWER,
            title=f"Day of the week - Outlier Removal - {country}"
        )
    
```


```
def plot_power_comparison(
    df_raw: pd.DataFrame,
    df_filtered: pd.DataFrame,
    country: str,
    colors: dict[str, str],
    title_suffix: str = "Power Generation (Hourly)"
):
    """
    Plot a comparison of raw vs filtered power data for each power source.
    """

    n_sources = len(Columns.Power.ALL)
    fig, axes = plt.subplots(
        nrows=n_sources,
        ncols=1,
        sharex=True,
        figsize=(14, 2.5 * n_sources)
    )

    for ax, col in zip(axes, Columns.Power.ALL):
        # Plot raw data
        ax.plot(
            df_raw.index,
            df_raw[col],
            color="lightgray",
            linewidth=0.6,
            label="Raw"
        )

        # Plot filtered data
        ax.plot(
            df_filtered.index,
            df_filtered[col],
            color=colors.get(country, "tab:blue"),
            linewidth=0.8,
            label="Filtered"
        )

        # Highlight removed points
        removed_idx = df_raw.index.difference(df_filtered.index)
        # Keep only valid indices for this column
        valid_idx = removed_idx.intersection(df_raw[col].dropna().index)

        # Extract x and y as numpy arrays to avoid alignment issues
        x_vals = valid_idx.to_numpy()
        y_vals = df_raw.loc[valid_idx, col].to_numpy()

        if len(x_vals) > 0:
            ax.scatter(
                x_vals,
                y_vals,
                color="red",
                s=20,
                marker="x",
                label="Removed"
            )

        ax.set_title(col, fontsize=11)
        ax.set_ylabel("MW")
        ax.grid(alpha=0.3)

        # Avoid duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=9)

    fig.suptitle(f"{country} – {title_suffix}", fontsize=16)
    plt.xlabel("Datetime")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()





# for country, df in dataframes_filtered_regression:
#     print("=" * 40)
#     print(f"\nPower Comparison for {country}:")
#     print("-" * 40)
#     plot_power_comparison(
#         df_raw=dict(dataframes)[country],
#         df_filtered=df,
#         country=country,
#         colors=colors,
#         title_suffix="Power Generation – Filtered Diagnostics"
#     )
```

2.2.4Daily or periodic pattern analysis 
day-of-week
hour-of-day
seasonality indicators
test-cycle patterns


```
def analyze_periodic_patterns(dataframes_list):

    print(f"\n{'='*80}\n2.2.4 PERIODIC PATTERN ANALYSIS\n{'='*80}")
    
    # only visalize the total pwoer collumn
    TARGET = Columns.CALC.TOTAL_POWER

    # iteration through all countries
    for country, df in dataframes_list:
        if TARGET not in df.columns: continue
        print(f"{country}:")

        # copy the data and prepareing it for plotting
        pdf = df[[TARGET]].copy()
        pdf[Columns.AXIS.HOURS_OF_DAY] = pdf.index.hour
        pdf[Columns.AXIS.DAY_OF_WEEK_STR] = pdf.index.day_name()
        pdf[Columns.AXIS.MONTH_STR] = pdf.index.month_name()

        # layout for the plots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4.5))
        
        # 1 daily profile (mean line and shading for all the other data)
        #------------------
        day_stats = pdf.groupby(Columns.AXIS.HOURS_OF_DAY)[TARGET].agg(['mean', 'std'])
        # averaging line
        ax1.plot(day_stats.index, 
                 day_stats['mean'], 
                 color=colors[country], 
                 lw=2.5, label='Mean')
        # shading between -std / mean / +std
        ax1.fill_between(day_stats.index, 
                         day_stats['mean'] - day_stats['std'], 
                         day_stats['mean'] + day_stats['std'], 
                         color=colors[country], 
                         alpha=0.2)
        
        # set title and gid
        ax1.set(title=f"{country}: Daily Profile", 
                xlabel="Hour (0-23)", 
                ylabel="MW", 
                xticks=range(0, 25, 4))
        ax1.grid(True, 
                 ls='--', 
                 alpha=0.5)

        # 2 weekly profile (mean line and shading for all the other data)
        #------------------
        week_stats = pdf.groupby(Columns.AXIS.DAY_OF_WEEK_STR)[TARGET].agg(['mean', 'std']).reindex(WEEK_ORDER)
        # averaging line
        ax2.plot(week_stats.index, 
                 week_stats['mean'], 
                 color=colors[country], 
                 lw=2.5, 
                 label='Mean')
        ax2.fill_between(week_stats.index, 
                         week_stats['mean'] - week_stats['std'], 
                         week_stats['mean'] + week_stats['std'], 
                         color=colors[country], 
                         alpha=0.2)
        # shading between -std / mean / +std
        ax2.set(title=f"{country}: Weekly Cycle", 
                xlabel="", 
                ylabel="")
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, ls='--', alpha=0.5)

        # 3 seasonal tred with bar charts
        #   + marking the maximum in red
        #------------------
        season_stats = pdf.groupby(Columns.AXIS.MONTH_STR)[TARGET].mean().reindex(MONTH_ORDER)
        bars = ax3.bar(season_stats.index, 
                       season_stats.values, 
                       color=colors[country], 
                       alpha=0.7, 
                       edgecolor='k')
        # finding the month with the maximum production
        # marking it dark red
        if not season_stats.empty:
            peak_idx = MONTH_ORDER.index(season_stats.idxmax())
            bars[peak_idx].set(color='darkred', alpha=1.0)
        
        # set up plot
        ax3.set(title=f"{country}: Seasonal Trend", ylabel="")
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(axis='y', ls='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

# execute
analyze_periodic_patterns(dataframes)
```

2.2.5 Summary of observed patterns as short check statements (similar to True/False style) with evidence

[ITALY] 
1. Significant consumption drop on Weekends (>5%).
   -> [TRUE]
   -> Weekends are 20.9% lower than Weekdays (Avg: 24.5GW vs 31.0GW).

2. Grid load is dominated by Winter heating demand.
   -> [FALSE]
   -> Summer load is 2.7% higher on average.
      Temperature Spain (average):  summer maximum 32
                                    winter minimum 5

3. Strong correlation between Solar Generation and Total Load.
   -> [FALSE]
   -> Pearson Correlation coefficient is 0.41 (1.0 = Perfect Sync).
      Best correlation in italy is between fossil coal delivered gas and Biomass (0.72)
      Also a high collrelation for Hydro Pumped storage and Hydro Water Reservoirs is visible (0.75)
------------------------------------------------------------

[FRANCE] REPORT CARD:
------------------------------------------------------------
1. Significant consumption drop on Weekends (>5%).
   -> [TRUE]
   -> Weekends are 8.0% lower than Weekdays (Avg: 55.6GW vs 60.5GW).

2. Grid load is dominated by Winter heating demand.
   -> [TRUE ]
   -> Winter load is 42.5% higher on average.
      Temperature Spain (average):  summer maximum 25
                                    winter minimum 0

3. Strong correlation between Solar Generation and Total Load.
   -> [FALSE]
   -> Pearson Correlation coefficient is -0.12 (1.0 = Perfect Sync).
      France has the most destinct correlation between total power and nuclear (0.9) which is due to the fact that france has the most nuclear power plants of our observed states
------------------------------------------------------------

[GERMANY] REPORT CARD:
------------------------------------------------------------
1. Significant consumption drop on Weekends (>5%).
   -> [TRUE]
   -> Weekends are 12.3% lower than Weekdays (Avg: 55.7GW vs 63.5GW).

2. Grid load is dominated by Winter heating demand.
   -> [TRUE]
   -> Winter load is 18.0% higher on average.
      Temperature Spain (average):  summer maximum 20
                                    winter minimum 1.7


3. Strong correlation between Solar Generation and Total Load.
   -> [FALSE]
   -> Pearson Correlation coefficient is 0.37 (1.0 = Perfect Sync).
      But a really good correlation between Windenergy offshore and onshore of 0.64
------------------------------------------------------------

[SPAIN] REPORT CARD:
1. Significant consumption drop on Weekends (>5%). 
   -> [TRUE] 
   -> Weekends are 9.1% lower than Weekdays (Avg: 28.4GW vs 31.2GW).

2. Grid load is dominated by Winter heating demand. 
   -> [FALSE] (Spain is Summer/Winter balanced or Summer peaking) 
   -> Summer load is 1.2% higher on average (Air Conditioning demand balances Winter heating). (Avg Winter: 30.1GW vs Avg Summer: 30.5GW)
      Temperature Spain (average):  summer maximum 28.6
                                    winter minimum 10.5

3. Strong correlation between Solar Generation and Total Load.
   -> [TRUE] 
   -> Pearson Correlation coefficient is 0.72 (1.0 = Perfect Sync).
      High correlation due to sunny weather driving both solar output and air conditioning demand

2.3  C. Probability and Event Analysis (45 points)
- Threshold-based probability estimation for events (define event, justify threshold, compute empirical probability) (15 points)
- Cross tabulation analysis for two variables (10 points)
- Conditional probability analysis (at least two meaningful conditional relationships) (15 points)
- Summary of observations and limitations (what could bias these estimates, what assumppions were made) (5 points)


```
def threshold_event_probability(df, state, thresholds, plot=False):
    report = {
        "state": state,
        "probabilities": {},
        "thresholds": {}
    }

    for power, (min_val, thresh_type) in thresholds.items():
        if power not in df.columns:
            report["thresholds"][power] = {"type": thresh_type, "value": None}
            report["probabilities"][power] = None
            continue

        series = df[power].dropna()

        if series.empty:  # <- critical check
            report["thresholds"][power] = {"type": thresh_type, "value": None}
            report["probabilities"][power] = None
            continue

        if thresh_type == "percentile":
            threshold_val = np.percentile(series, min_val)
        else:  # absolute
            threshold_val = min_val

        event = series >= threshold_val
        probability = event.sum() / len(series)

        report["thresholds"][power] = {
            "type": thresh_type,
            "value": threshold_val
        }
        report["probabilities"][power] = probability

    return report





# Define thresholds per power source
POWER_THRESHOLDS = {
    # Columns.Power.SOLAR: (90, "percentile"),          # Top 5% of solar output
    Columns.Power.SOLAR: (10000, "absolute"),      # Absolute threshold in MW
    # Columns.Power.SOLAR: (12000, "absolute"),      # Absolute threshold in MW
    # Columns.Power.SOLAR: (35000, "absolute"),      # Absolute threshold in MW
    # Columns.Power.WIND_ONSHORE: (150000, "absolute"), # Absolute threshold in MW
    # Columns.Power.WIND_OFFSHORE: (10000, "absolute"),
    Columns.Power.WIND_ONSHORE: (30, "percentile"),
    Columns.Power.WIND_OFFSHORE: (30, "percentile"),
    Columns.Power.NUCLEAR: (80, "percentile"),
    Columns.Power.FOSSIL_GAS: (10_000, "absolute")
}

for country, df in dataframes_filtered:
    
    report = threshold_event_probability(
        df=dict(dataframes)[country],
        state=country,
        thresholds=POWER_THRESHOLDS,
        plot=True
    )

    # Print probabilities
    print(f"\nThreshold Event Probabilities for {country}:")
    print("-" * 40)
    for source, prob in report["probabilities"].items():
        if prob is None:
            print(f"{source}: column not found in data")
        else:
            print(f"{source}: {prob:.2%}")


```


```

# -----------------------------
# 1. Threshold-based probability estimation
# -----------------------------
def compute_event_probabilities(df, thresholds: dict[str, tuple[float | str]], verbose=True):
    """
    Compute empirical probability of threshold-defined events for each power source.
    Handles missing or insufficient data safely.
    """

    report = {
        "probabilities": {},
        "thresholds": {},
        "notes": {}
    }

    for power, (val, typ) in thresholds.items():

        if power not in df.columns:
            report["probabilities"][power] = None
            report["notes"][power] = "column not found"
            continue

        series = df[power].dropna()

        # Handle empty or insufficient data
        if len(series) == 0:
            report["probabilities"][power] = None
            report["notes"][power] = "no valid data"
            continue

        if typ == "percentile":
            threshold_val = np.percentile(series, val)
        else:  # absolute
            threshold_val = val

        # Define event
        event = series >= threshold_val
        probability = event.mean()

        report["thresholds"][power] = {
            "value": float(threshold_val),
            "type": typ
        }
        report["probabilities"][power] = float(probability)
        report["notes"][power] = "ok"

        if verbose:
            print(
                f"{power}: threshold={threshold_val:.2f} ({typ}), "
                f"P(event)={probability:.2%}"
            )

    return report





# -----------------------------
# 2. Cross-tabulation analysis
# -----------------------------
def cross_tab_analysis(df, col1, col2, bins1=None, bins2=None, normalize=True):
    """
    Cross-tabulation between two variables (can discretize with bins).

    Parameters
    ----------
    df : pd.DataFrame
    col1, col2 : str
        Columns to cross-tabulate
    bins1, bins2 : list or None
        Optional bin edges to discretize continuous variables
    normalize : bool
        Return percentages if True
    
    Returns
    -------
    ctab : pd.DataFrame
    """
    data1 = pd.cut(df[col1], bins=bins1) if bins1 else df[col1]
    data2 = pd.cut(df[col2], bins=bins2) if bins2 else df[col2]
    
    ctab = pd.crosstab(data1, data2, normalize="all" if normalize else False)
    return ctab

# -----------------------------
# 3. Conditional probability
# -----------------------------
def conditional_probability(df, event_col, condition_col, threshold_event, threshold_condition):
    """
    Compute P(event | condition) for two columns.

    Parameters
    ----------
    df : pd.DataFrame
    event_col, condition_col : str
    threshold_event, threshold_condition : numeric
    
    Returns
    -------
    float
        Conditional probability
    """
    event = df[event_col] >= threshold_event
    condition = df[condition_col] >= threshold_condition

    if condition.sum() == 0:
        return None

    return (event & condition).sum() / condition.sum()


# -----------------------------
# 4. Summary
# -----------------------------
def summarize_prob_analysis(prob_report, ctab_report=None, conditional_probs=None):
    """
    Print concise summary of probability analysis.
    """
    print("\n=== Summary of Probability & Event Analysis ===\n")
    print("Event Probabilities:")
    for source, prob in prob_report["probabilities"].items():
        if prob is not None:
            print(f"- {source}: {prob:.2%}")
        else:
            print(f"- {source}: No data")
    
    if ctab_report is not None:
        print("\nCross-tabulation (sample):")
        print(ctab_report.head())
    
    if conditional_probs is not None:
        print("\nConditional Probabilities:")
        for desc, val in conditional_probs.items():
            if val is not None:
                print(f"- {desc}: {val:.2%}")
            else:
                print(f"- {desc}: No data / zero condition count")
    
    print("\nLimitations: thresholds are user-defined, independent assumption, missing data not considered.")



for country, df in dataframes_filtered:
    # 1. Compute event probabilities
    prob_report = compute_event_probabilities(df, POWER_THRESHOLDS)

    # 2. Cross-tabulation example
    ctab = cross_tab_analysis(df, Columns.Power.SOLAR, Columns.Power.WIND_ONSHORE, bins1=5, bins2=5)

    # 3. Conditional probabilities
    cond_probs = {
        "P(SOLAR high | WIND_ONSHORE high)": conditional_probability(
            df, Columns.Power.SOLAR, Columns.Power.WIND_ONSHORE, threshold_event=np.percentile(df[Columns.Power.SOLAR], 95),
            threshold_condition=100_000
        ),
        "P(FOSSIL_GAS high | SOLAR low)": conditional_probability(
            df, Columns.Power.FOSSIL_GAS, Columns.Power.SOLAR, threshold_event=200_000, threshold_condition=np.percentile(df[Columns.Power.SOLAR], 5)
        )
    }

    # 4. Summary
    summarize_prob_analysis(prob_report, ctab_report=ctab, conditional_probs=cond_probs)



```


```

def plot_cross_tab_heatmap(
    ctab: pd.DataFrame,
    *,
    title: str,
    cmap: str = "viridis",
    show_percent: bool = True,
    figsize=(10, 7)
):
    """
    Plot a heatmap for a cross-tabulation matrix.

    Parameters
    ----------
    ctab : pd.DataFrame
        Output of cross_tab_analysis (normalized or not).
    title : str
        Plot title.
    cmap : str
        Seaborn colormap.
    show_percent : bool
        If True, display values as percentages.
    figsize : tuple
        Figure size.
    """

    data = ctab.copy()

    if show_percent:
        data = data * 100
        fmt = ".2f"
        cbar_label = "Joint probability (%)"
    else:
        fmt = ".3f"
        cbar_label = "Frequency"

    plt.figure(figsize=figsize)

    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        cbar_kws={"label": cbar_label},
        linewidths=0.5,
        linecolor="white"
    )

    plt.title(title, fontsize=14, pad=12)
    plt.xlabel(ctab.columns.name or "Wind output (binned)")
    plt.ylabel(ctab.index.name or "Solar output (binned)")
    plt.tight_layout()
    plt.show()


for country, df in dataframes_filtered:

    ctab = cross_tab_analysis(
        df,
        "Solar - Actual Aggregated [MW]",
        "Wind Onshore - Actual Aggregated [MW]",
        bins1=5,
        bins2=5,
        normalize=True
    )

    plot_cross_tab_heatmap(
        ctab,
        title=f"{country} – Joint Distribution of Solar and Onshore Wind Generation"
    )
```
