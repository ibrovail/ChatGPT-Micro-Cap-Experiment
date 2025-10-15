"""
Plot portfolio performance vs. S&P 500 with a configurable starting equity.

- Normalizes BOTH series (portfolio and S&P) to the same starting equity.
- Aligns S&P data to the portfolio dates with forward-fill.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List

import matplotlib.pyplot as plt # type: ignore
import matplotlib.dates as mdates # type: ignore
import matplotlib.ticker as mticker  # type: ignore # <-- added
import pandas as pd # type: ignore
import yfinance as yf # type: ignore

DATA_DIR = Path(__file__).resolve().parent
PORTFOLIO_CSV = DATA_DIR / "chatgpt_portfolio_update.csv"


# ---------- Helpers ----------

def _date_only_series(x: pd.Series) -> pd.Series:
    """tz-naive, normalized to midnight (YYYY-MM-DD 00:00)."""
    dt = pd.to_datetime(x, errors="coerce")
    if hasattr(dt, "dt"):
        try:
            dt = dt.dt.tz_localize(None)
        except (TypeError, AttributeError):
            pass
        return dt.dt.normalize()
    if getattr(dt, "tzinfo", None) is not None:
        try:
            dt = dt.tz_convert(None)
        except Exception:
            dt = dt.tz_localize(None)
    return pd.to_datetime(dt).normalize()


def parse_date(date_str: str, label: str) -> pd.Timestamp:
    try:
        dt = pd.to_datetime(date_str)
        if getattr(dt, "tzinfo", None) is not None:
            try:
                dt = dt.tz_convert(None)
            except Exception:
                dt = dt.tz_localize(None)
        return pd.to_datetime(dt).normalize()
    except Exception as exc:
        raise SystemExit(f"Invalid {label} '{date_str}'. Use YYYY-MM-DD.") from exc


def _normalize_to_start(series, starting_equity):
    """Normalize a series to start at starting_equity."""
    s = pd.to_numeric(series.iloc[:, 0], errors="coerce") if isinstance(series, pd.DataFrame) \
        else pd.to_numeric(series, errors="coerce")
    if s.empty:
        return pd.Series(dtype=float)
    start_value = s.iloc[0]
    if start_value == 0:
        return s * 0
    return (s / start_value) * starting_equity


def _align_to_dates(sp500_data: pd.DataFrame, portfolio_dates: pd.Series) -> pd.Series:
    """Align S&P 500 data to portfolio dates using forward fill."""
    aligned_df = pd.DataFrame({"Date": _date_only_series(portfolio_dates)})
    merged = aligned_df.merge(sp500_data, on="Date", how="left")
    merged["Value"] = merged["Value"].ffill()
    return merged["Value"]


# ---------- Data loaders ----------

def load_portfolio_details(
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    portfolio_csv: Path = PORTFOLIO_CSV,
) -> pd.DataFrame:
    """Return TOTAL rows (Date, Total Equity) filtered to [start_date, end_date]."""
    if not portfolio_csv.exists():
        raise SystemExit(f"Portfolio file '{portfolio_csv}' not found.")

    df = pd.read_csv(portfolio_csv)
    totals = df[df["Ticker"] == "TOTAL"].copy()
    if totals.empty:
        raise SystemExit("Portfolio CSV contains no TOTAL rows.")

    totals["Date"] = _date_only_series(totals["Date"])
    totals["Total Equity"] = pd.to_numeric(totals["Total Equity"], errors="coerce")
    totals = totals.dropna(subset=["Date", "Total Equity"]).sort_values("Date")

    min_date = totals["Date"].min()
    max_date = totals["Date"].max()
    if start_date is None or start_date < min_date:
        start_date = min_date
    if end_date is None or end_date > max_date:
        end_date = max_date
    if start_date > end_date:
        raise SystemExit("Start date must be on or before end date.")

    mask = (totals["Date"] >= start_date) & (totals["Date"] <= end_date)
    return totals.loc[mask, ["Date", "Total Equity"]].reset_index(drop=True)


def _flatten_columns(cols) -> List[str]:
    """Flatten possible MultiIndex columns from yfinance."""
    if isinstance(cols, pd.MultiIndex):
        flat = []
        for tup in cols:
            parts = [str(p) for p in tup if p is not None and str(p) != ""]
            flat.append("_".join(parts))
        return flat
    return list(map(str, cols))


def download_sp500(dates: pd.Series, starting_equity: float) -> pd.DataFrame:
    """Download S&P 500 data and normalize to starting equity, aligned to the given dates."""
    dates = _date_only_series(dates)
    if dates.empty:
        return pd.DataFrame(columns=["Date", "SPX Value"])

    start_date = dates.min()
    end_date = dates.max()

    try:
        sp500 = yf.download("^GSPC",
                            start=start_date, end=end_date + pd.Timedelta(days=1),
                            progress=False, auto_adjust=False, group_by="column")
    except Exception as e:
        print(f"Error downloading S&P 500 data: {e}")
        return pd.DataFrame(columns=["Date", "SPX Value"])

    if sp500 is None or sp500.empty:
        return pd.DataFrame(columns=["Date", "SPX Value"])

    sp500 = sp500.reset_index()
    sp500.columns = _flatten_columns(sp500.columns)
    sp500["Date"] = _date_only_series(sp500["Date"])

    close_candidates = [c for c in sp500.columns if c.lower().startswith("close")]
    if not close_candidates:
        close_candidates = [c for c in sp500.columns if c.lower().startswith("adj close")]
    if not close_candidates:
        print("Could not find a Close column in yfinance data.")
        return pd.DataFrame(columns=["Date", "SPX Value"])

    close_col = close_candidates[0]
    sp500_close = sp500[["Date", close_col]].rename(columns={close_col: "Value"})

    aligned_values = _align_to_dates(sp500_close, dates)
    norm = _normalize_to_start(aligned_values, starting_equity)
    return pd.DataFrame({"Date": dates, "SPX Value": norm.values})


# ---------- Plotting ----------

def plot_comparison(
    portfolio: pd.DataFrame,
    spx: pd.DataFrame,
    starting_equity: float,
    title: str = "Portfolio vs. S&P 500 (Indexed)",
) -> None:
    """Plot the two normalized lines with strict day ticks, YYYY-MM-DD labels."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    p_dates = _date_only_series(portfolio["Date"])
    ax.plot(p_dates, portfolio["Total Equity"], label=f"Portfolio (start = ${starting_equity:g})", marker="o")

    if not spx.empty:
        s_dates = _date_only_series(spx["Date"])
        ax.plot(s_dates, spx["SPX Value"], label="S&P 500", marker="o", linestyle="--")

    p_last = float(portfolio["Total Equity"].iloc[-1])
    ax.text(p_dates.iloc[-1], p_last * 1.01, f"{(p_last/starting_equity - 1)*100:+.1f}%", fontsize=9)
    if not spx.empty:
        s_last = float(spx["SPX Value"].iloc[-1])
        ax.text(s_dates.iloc[-1], s_last * 1.01, f"{(s_last/starting_equity - 1)*100:+.1f}%", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Index (start = {starting_equity:g})")
    ax.legend()
    ax.grid(True)

    # Strict daily ticks & YYYY-MM-DD labels
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3)) # <-- changed to 3-day interval
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_minor_locator(mticker.NullLocator())  # <-- fixed

    # Add 1-day padding on each side
    pad = pd.Timedelta(days=0.75)
    ax.set_xlim(p_dates.min() - pad, p_dates.max() + pad)

    fig.autofmt_xdate(rotation=30, ha="right")
    plt.tight_layout()


# ---------- Main ----------

def main(
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    starting_equity: float,
    output: Optional[Path],
    portfolio_csv: Path = PORTFOLIO_CSV,
) -> None:
    totals = load_portfolio_details(start_date, end_date, portfolio_csv=portfolio_csv)

    norm_port = totals.copy()
    norm_port["Total Equity"] = _normalize_to_start(norm_port["Total Equity"], starting_equity)

    spx = download_sp500(norm_port["Date"], starting_equity)

    plot_comparison(norm_port, spx, starting_equity, title="ChatGPT Portfolio vs. S&P 500 (Indexed)")

    if output:
        output = output if output.is_absolute() else DATA_DIR / output
        plt.savefig(output, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot portfolio performance vs S&P 500")
    parser.add_argument("--start-date", type=str, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, help="YYYY-MM-DD")
    parser.add_argument("--start-equity", type=float, default=100.0, help="Baseline to index both series (default 100)")
    parser.add_argument("--baseline-file", type=str, help="Path to a text file containing a single number for baseline")
    parser.add_argument("--output", type=str, help="Optional path to save the chart (.png/.jpg/.pdf)")

    args = parser.parse_args()
    start = parse_date(args.start_date, "start date") if args.start_date else None
    end = parse_date(args.end_date, "end date") if args.end_date else None

    baseline = args.start_equity
    if args.baseline_file:
        p = Path(args.baseline_file)
        if not p.exists():
            raise SystemExit(f"Baseline file not found: {p}")
        try:
            baseline = float(p.read_text().strip())
        except Exception as exc:
            raise SystemExit(f"Could not parse baseline from {p}") from exc

    out_path = Path(args.output) if args.output else None
    main(start, end, baseline, out_path)
