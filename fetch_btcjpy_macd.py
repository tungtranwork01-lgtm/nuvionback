import time
import math
import requests
import pandas as pd

from datetime import datetime, timezone


BINANCE_BASE_URL = "https://api.binance.com"


def to_milliseconds(dt: datetime) -> int:
    """
    Chuyển datetime (UTC) sang milliseconds kể từ epoch.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_klines(
    symbol: str,
    interval: str,
    start_time_ms: int,
    end_time_ms: int,
    limit: int = 1000,
) -> list:
    """
    Lấy toàn bộ dữ liệu kline giữa start_time_ms và end_time_ms
    bằng cách phân trang (Binance giới hạn 1000 dòng mỗi request).
    """
    url = f"{BINANCE_BASE_URL}/api/v3/klines"

    # Thời gian mỗi nến theo ms cho 1 số interval cơ bản
    interval_ms_map = {
        "1m": 60 * 1000,
        "3m": 3 * 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }

    if interval not in interval_ms_map:
        raise ValueError(f"Interval không được hỗ trợ: {interval}")

    step = interval_ms_map[interval]

    all_klines = []
    current_start = start_time_ms

    while current_start <= end_time_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time_ms,
            "limit": limit,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()

        klines = resp.json()
        if not klines:
            break

        all_klines.extend(klines)

        # Thời gian mở của nến cuối cùng
        last_open_time = klines[-1][0]
        next_start = last_open_time + step

        if next_start <= current_start:
            # Tránh loop vô hạn nếu vì lý do gì đó dữ liệu không tiến
            break

        current_start = next_start

        # Nghỉ nhẹ để tránh rate-limit
        time.sleep(0.2)

        # Nếu đã vượt quá end_time thì dừng
        if current_start > end_time_ms:
            break

    return all_klines


def klines_to_dataframe(klines: list) -> pd.DataFrame:
    """
    Chuyển list klines Binance thành DataFrame với index là open_time (UTC).
    Mỗi phần tử kline có 12 trường:
    [0] open time
    [1] open
    [2] high
    [3] low
    [4] close
    [5] volume
    [6] close time
    [7] quote asset volume
    [8] number of trades
    [9] taker buy base asset volume
    [10] taker buy quote asset volume
    [11] ignore
    """
    if not klines:
        raise ValueError("Không có dữ liệu kline để chuyển thành DataFrame.")

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]

    df = pd.DataFrame(klines, columns=columns)

    # Ép kiểu số
    float_cols = ["open", "high", "low", "close", "volume"]
    for col in float_cols:
        df[col] = df[col].astype(float)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()

    return df


def compute_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Tính MACD (EMA) cho một chuỗi giá (thường là close).
    Trả về DataFrame gồm: macd, signal, hist.
    """
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    hist = macd - signal

    return pd.DataFrame(
        {
            "macd": macd,
            "signal": signal,
            "hist": hist,
        }
    )


def resample_and_macd(
    df_5m: pd.DataFrame,
    resample_rule: str,
    label: str,
) -> pd.DataFrame:
    """
    Resample từ 5m sang khung lớn hơn (1H, 4H, 1D, ...) dựa trên cột close,
    sau đó tính MACD trên chuỗi resampled.

    Kết quả được align ngược lại về index 5m bằng forward-fill, với
    các cột macd_<label>, signal_<label>, hist_<label>.
    """
    # Resample chuỗi close sang khung lớn hơn
    close_resampled = df_5m["close"].resample(resample_rule).last().dropna()

    macd_df = compute_macd(close_resampled)
    macd_df.columns = [f"macd_{label}", f"signal_{label}", f"hist_{label}"]

    # Đưa MACD khung lớn về index 5m
    macd_5m_aligned = macd_df.reindex(df_5m.index, method="ffill")

    return macd_5m_aligned


def main():
    # Cấu hình
    symbol = "BTCJPY"
    interval = "5m"

    # Ngày bắt đầu: 01/04/2024 (giả sử theo giờ UTC)
    start_dt = datetime(2024, 4, 1, 0, 0, 0, tzinfo=timezone.utc)
    start_time_ms = to_milliseconds(start_dt)

    # Đến thời điểm hiện tại (UTC)
    now_utc = datetime.now(timezone.utc)
    end_time_ms = to_milliseconds(now_utc)

    print(f"Lấy dữ liệu {symbol} khung {interval} từ {start_dt} đến {now_utc} (UTC)...")
    klines = fetch_klines(
        symbol=symbol,
        interval=interval,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
    )
    print(f"Số lượng nến 5m nhận được: {len(klines)}")

    if not klines:
        print("Không lấy được dữ liệu nào từ Binance. Thoát.")
        return

    df_5m = klines_to_dataframe(klines)

    # MACD cho khung 5m
    macd_5m = compute_macd(df_5m["close"])
    macd_5m.columns = ["macd_5m", "signal_5m", "hist_5m"]
    df_5m = df_5m.join(macd_5m)

    # MACD cho khung 1h, 4h, 1d (dựa trên khung 5m, thông qua resample)
    macd_1h = resample_and_macd(df_5m, "1H", "1h")
    macd_4h = resample_and_macd(df_5m, "4H", "4h")
    macd_1d = resample_and_macd(df_5m, "1D", "1d")

    df_all = df_5m.join([macd_1h, macd_4h, macd_1d])

    # Xuất ra CSV
    output_file = "btcjpy_macd_5m_1h_4h_1d_from_2024-04-01.csv"
    df_all.to_csv(output_file, index_label="open_time")

    print(f"Đã lưu dữ liệu vào file: {output_file}")


if __name__ == "__main__":
    main()

