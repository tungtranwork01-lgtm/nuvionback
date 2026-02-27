"""
btcjpy_cron.py  —  Real-time updater for BTCJPY + MACD on Supabase
=======================================================================
Every 60 s:
  1. Read the latest open_time already in Supabase
  2. Pull ~8 000 historical close prices (for EMA warm-up)
  3. Fetch new 5 m candles from Binance  (from latest onward)
  4. Compute MACD  5 m / 1 h / 4 h / 1 d
  5. UPSERT rows (update if open_time exists, insert if new)

Prerequisites
-------------
  pip install requests pandas supabase python-dotenv

  Create  .env  in the same folder:
      SUPABASE_URL=https://xxxxx.supabase.co
      SUPABASE_KEY=your-service-role-key

Recommended (run once in Supabase SQL editor):
  ALTER TABLE public."BTCJPY" ADD PRIMARY KEY (open_time);
"""

import os
import time
import logging
from typing import Optional

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from supabase import create_client, Client

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Config ────────────────────────────────────────────────────────────────────
SUPABASE_URL: str = os.environ["SUPABASE_URL"]
SUPABASE_KEY: str = os.environ["SUPABASE_KEY"]

BINANCE_BASE = "https://api.binance.com"
SYMBOL = "BTCJPY"
TABLE = "BTCJPY"
INTERVAL = "5m"
INTERVAL_MS = 5 * 60 * 1000

# ~28 days of 5m candles — covers the 26-period slow EMA on the 1D timeframe
WARMUP_CANDLES = 8000
REFRESH_SEC = 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("btcjpy_cron")


# ── Binance ───────────────────────────────────────────────────────────────────

def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    url = f"{BINANCE_BASE}/api/v3/klines"
    all_klines: list = []
    cur = start_ms

    while cur <= end_ms:
        resp = requests.get(
            url,
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": cur,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_klines.extend(data)

        nxt = data[-1][0] + INTERVAL_MS
        if nxt <= cur:
            break
        cur = nxt
        if cur > end_ms:
            break
        time.sleep(0.15)

    return all_klines


def klines_to_df(klines: list) -> pd.DataFrame:
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(klines, columns=cols)

    num_cols = cols[1:]  # everything except open_time
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.set_index("open_time").sort_index()


# ── MACD ──────────────────────────────────────────────────────────────────────

def compute_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9,
) -> pd.DataFrame:
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": macd, "signal": sig, "hist": macd - sig})


def macd_all_timeframes(
    close_5m: pd.Series, target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute MACD 5m/1h/4h/1d on *close_5m* and return rows for *target_index*."""

    # 5m MACD
    m5 = compute_macd(close_5m)
    m5.columns = ["macd_5m", "signal_5m", "hist_5m"]
    result = m5.reindex(target_index)

    # Higher-timeframe MACDs (resample -> MACD -> ffill back to 5m)
    for rule, label in [("1h", "1h"), ("4h", "4h"), ("1D", "1d")]:
        resampled = close_5m.resample(rule).last().dropna()
        m = compute_macd(resampled)
        m.columns = [f"macd_{label}", f"signal_{label}", f"hist_{label}"]
        aligned = m.reindex(close_5m.index, method="ffill")
        result = result.join(aligned.reindex(target_index))

    return result


# ── Supabase helpers ──────────────────────────────────────────────────────────

def new_sb() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def get_latest_open_time(sb: Client) -> Optional[pd.Timestamp]:
    res = (
        sb.table(TABLE)
        .select("open_time")
        .order("open_time", desc=True)
        .limit(1)
        .execute()
    )
    if res.data and res.data[0].get("open_time"):
        ts = pd.Timestamp(res.data[0]["open_time"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts
    return None


def fetch_warmup_closes(
    sb: Client, before: pd.Timestamp, n: int = WARMUP_CANDLES,
) -> pd.Series:
    """Pull last *n* close prices BEFORE *before* from Supabase (paginated)."""
    rows: list = []
    chunk = 1000
    offset = 0

    while offset < n:
        size = min(chunk, n - offset)
        res = (
            sb.table(TABLE)
            .select("open_time,close")
            .lt("open_time", before.isoformat())
            .order("open_time", desc=True)
            .range(offset, offset + size - 1)
            .execute()
        )
        if not res.data:
            break
        rows.extend(res.data)
        if len(res.data) < size:
            break
        offset += len(res.data)

    if not rows:
        return pd.Series(dtype=float, name="close")

    df = pd.DataFrame(rows)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df["close"] = pd.to_numeric(df["close"])
    return df.set_index("open_time").sort_index()["close"]


def upsert_rows(sb: Client, records: list, batch_size: int = 500):
    """UPSERT rows into Supabase — requires PRIMARY KEY on open_time."""
    for i in range(0, len(records), batch_size):
        sb.table(TABLE).upsert(records[i : i + batch_size]).execute()


def df_to_records(df: pd.DataFrame) -> list:
    """DataFrame -> list[dict] safe for Supabase JSON insert."""
    recs = []
    for ts, row in df.iterrows():
        d = {"open_time": ts.isoformat()}
        for col in df.columns:
            v = row[col]
            d[col] = None if pd.isna(v) else float(v)
        recs.append(d)
    return recs


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_once():
    sb = new_sb()

    # 1) Latest candle already in DB
    latest = get_latest_open_time(sb)
    if latest is None:
        log.warning("Table is empty. Import CSV first.")
        return
    log.info("Latest in DB : %s", latest)

    # 2) Historical close prices for EMA warm-up
    warmup = fetch_warmup_closes(sb, latest, WARMUP_CANDLES)
    log.info("Warmup rows  : %s", len(warmup))

    # 3) New candles from Binance  (from latest, inclusive — refreshes last candle)
    start_ms = int(latest.timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    klines = fetch_klines(SYMBOL, INTERVAL, start_ms, end_ms)

    if not klines:
        log.info("No new candles from Binance.")
        return

    new_df = klines_to_df(klines)
    log.info(
        "Binance rows : %s  (%s -> %s)",
        len(new_df), new_df.index[0], new_df.index[-1],
    )

    # 4) Combine warm-up + new, compute MACD
    combined_close = pd.concat([warmup, new_df["close"]])
    combined_close = combined_close[~combined_close.index.duplicated(keep="last")].sort_index()

    macd_df = macd_all_timeframes(combined_close, target_index=new_df.index)
    new_df = new_df.join(macd_df)

    # 5) Write to Supabase via UPSERT (requires PK on open_time)
    records = df_to_records(new_df)
    upsert_rows(sb, records)
    log.info("Upserted %s rows to Supabase.", len(records))


def main():
    log.info("=== BTCJPY cron started  (refresh every %ss) ===", REFRESH_SEC)
    while True:
        try:
            run_once()
        except KeyboardInterrupt:
            log.info("Stopped by user.")
            break
        except Exception:
            log.exception("Error in run_once")
        time.sleep(REFRESH_SEC)


if __name__ == "__main__":
    main()
