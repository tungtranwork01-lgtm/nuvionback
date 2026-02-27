"""
Dashboard BTC/JPY  --  Dash app
  - Load full CSV into memory
  - Display max 2000 candles at a time
  - Date range picker + quick-select buttons (1D, 3D, 1W, 1M, 3M, 6M, 1Y, ALL)
  - Candlestick, Volume, MACD Histogram 5m / 1h / 4h / 1d

Run:  python build_dashboard.py
Open: http://127.0.0.1:8050
"""

import pandas as pd
import dash
from dash import dcc, html, Input, Output, callback_context, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CSV_PATH = "btcjpy_macd_5m_1h_4h_1d_from_2024-04-01.csv"
MAX_CANDLES = 2000

# ---------------------------------------------------------------------------
# Load data once
# ---------------------------------------------------------------------------
print(f"Loading {CSV_PATH} ...")
DF = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
print(f"Total candles: {len(DF):,}")

FIRST_DATE = DF.index[0]
LAST_DATE = DF.index[-1]

# Quick-button -> timedelta mapping
DELTAS = {
    "btn-1d": timedelta(days=1),
    "btn-3d": timedelta(days=3),
    "btn-1w": timedelta(weeks=1),
    "btn-1m": timedelta(days=30),
    "btn-3m": timedelta(days=90),
    "btn-6m": timedelta(days=180),
    "btn-1y": timedelta(days=365),
}

# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------
app = dash.Dash(__name__, title="BTC/JPY Dashboard")

BTN_STYLE = {
    "padding": "6px 14px",
    "border": "1px solid #444",
    "borderRadius": "4px",
    "backgroundColor": "#2a2a3d",
    "color": "#ccc",
    "cursor": "pointer",
    "fontSize": "13px",
}

app.layout = html.Div(
    style={
        "fontFamily": "'Segoe UI', Arial, sans-serif",
        "backgroundColor": "#131722",
        "color": "#d1d4dc",
        "minHeight": "100vh",
        "padding": "8px 12px",
    },
    children=[
        html.H2(
            "BTC / JPY  Dashboard",
            style={"textAlign": "center", "margin": "6px 0 10px", "letterSpacing": "1px"},
        ),
        # --- Controls row ---
        html.Div(
            style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "gap": "8px",
                "flexWrap": "wrap",
                "marginBottom": "6px",
            },
            children=[
                html.Label("From:", style={"fontSize": "13px"}),
                dcc.DatePickerSingle(
                    id="pick-start",
                    date=(LAST_DATE - timedelta(days=7)).date(),
                    min_date_allowed=FIRST_DATE.date(),
                    max_date_allowed=LAST_DATE.date(),
                    display_format="YYYY-MM-DD",
                    style={"backgroundColor": "#2a2a3d"},
                ),
                html.Label("To:", style={"fontSize": "13px"}),
                dcc.DatePickerSingle(
                    id="pick-end",
                    date=LAST_DATE.date(),
                    min_date_allowed=FIRST_DATE.date(),
                    max_date_allowed=LAST_DATE.date(),
                    display_format="YYYY-MM-DD",
                    style={"backgroundColor": "#2a2a3d"},
                ),
                html.Button("Go", id="btn-go", n_clicks=0, style={**BTN_STYLE, "backgroundColor": "#26a69a", "color": "#fff", "fontWeight": "bold"}),
                html.Span("|", style={"color": "#555"}),
                *[
                    html.Button(label, id=btn_id, n_clicks=0, style=BTN_STYLE)
                    for btn_id, label in [
                        ("btn-1d", "1D"),
                        ("btn-3d", "3D"),
                        ("btn-1w", "1W"),
                        ("btn-1m", "1M"),
                        ("btn-3m", "3M"),
                        ("btn-6m", "6M"),
                        ("btn-1y", "1Y"),
                        ("btn-all", "ALL"),
                    ]
                ],
            ],
        ),
        # --- Info text ---
        html.Div(
            id="info-text",
            style={"textAlign": "center", "fontSize": "12px", "color": "#888", "marginBottom": "4px"},
        ),
        # --- Chart ---
        dcc.Graph(
            id="main-chart",
            style={"height": "calc(100vh - 120px)"},
            config={"scrollZoom": True, "displayModeBar": True, "displaylogo": False},
        ),
    ],
)


# ---------------------------------------------------------------------------
# Build Plotly figure from a slice of DF
# ---------------------------------------------------------------------------
def build_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.022,
        row_heights=[0.38, 0.12, 0.12, 0.12, 0.13, 0.13],
        subplot_titles=(
            "BTC/JPY  Price (OHLC)",
            "Volume",
            "MACD Hist  5 m",
            "MACD Hist  1 h",
            "MACD Hist  4 h",
            "MACD Hist  1 d",
        ),
    )

    # 1) Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="BTC/JPY",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1,
        col=1,
    )

    # 2) Volume
    vol_colors = [
        "#26a69a" if c >= o else "#ef5350"
        for o, c in zip(df["open"], df["close"])
    ]
    fig.add_trace(
        go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color=vol_colors),
        row=2,
        col=1,
    )

    # 3-6) MACD Histograms
    hist_configs = [
        ("hist_5m", "5m", 3),
        ("hist_1h", "1h", 4),
        ("hist_4h", "4h", 5),
        ("hist_1d", "1d", 6),
    ]
    for col_name, label, row_idx in hist_configs:
        colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df[col_name]]
        fig.add_trace(
            go.Bar(x=df.index, y=df[col_name], name=f"Hist {label}", marker_color=colors),
            row=row_idx,
            col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        margin=dict(l=55, r=15, t=35, b=25),
        xaxis_rangeslider_visible=False,
        showlegend=False,
        font=dict(size=11, color="#d1d4dc"),
    )

    y_labels = [("Price", 1), ("Vol", 2), ("5m", 3), ("1h", 4), ("4h", 5), ("1d", 6)]
    for text, row in y_labels:
        fig.update_yaxes(title_text=text, row=row, col=1, gridcolor="#1e222d")

    for i in range(1, 7):
        fig.update_xaxes(gridcolor="#1e222d", row=i, col=1)

    return fig


# ---------------------------------------------------------------------------
# Main callback
# ---------------------------------------------------------------------------
@app.callback(
    Output("main-chart", "figure"),
    Output("info-text", "children"),
    Output("pick-start", "date"),
    Output("pick-end", "date"),
    # --- Inputs ---
    Input("btn-go", "n_clicks"),
    Input("btn-1d", "n_clicks"),
    Input("btn-3d", "n_clicks"),
    Input("btn-1w", "n_clicks"),
    Input("btn-1m", "n_clicks"),
    Input("btn-3m", "n_clicks"),
    Input("btn-6m", "n_clicks"),
    Input("btn-1y", "n_clicks"),
    Input("btn-all", "n_clicks"),
    # Date pickers as State so they don't auto-trigger on every change
    dash.dependencies.State("pick-start", "date"),
    dash.dependencies.State("pick-end", "date"),
    prevent_initial_call=False,
)
def update_chart(*args):
    # Unpack: first 9 are n_clicks, last 2 are dates from state
    start_date_str = args[-2]
    end_date_str = args[-1]

    trigger = callback_context.triggered_id or "btn-go"

    if trigger == "btn-all":
        s = FIRST_DATE
        e = LAST_DATE
    elif trigger in DELTAS:
        e = LAST_DATE
        s = e - DELTAS[trigger]
    else:
        s = pd.Timestamp(start_date_str, tz="UTC") if start_date_str else LAST_DATE - timedelta(days=7)
        e = pd.Timestamp(end_date_str, tz="UTC") + timedelta(hours=23, minutes=59) if end_date_str else LAST_DATE

    df_slice = DF.loc[s:e]

    truncated = len(df_slice) > MAX_CANDLES
    if truncated:
        df_slice = df_slice.tail(MAX_CANDLES)

    if len(df_slice) == 0:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#131722",
            plot_bgcolor="#131722",
            annotations=[dict(text="No data in selected range", showarrow=False, font=dict(size=20, color="#ef5350"))],
        )
        return fig, "No data in selected range", no_update, no_update

    trunc_note = f"  (capped at {MAX_CANDLES})" if truncated else ""
    info = (
        f"{len(df_slice):,} candles{trunc_note}   |   "
        f"{df_slice.index[0].strftime('%Y-%m-%d %H:%M')}  ->  {df_slice.index[-1].strftime('%Y-%m-%d %H:%M')} UTC"
    )

    fig = build_chart(df_slice)

    new_start = df_slice.index[0].date().isoformat()
    new_end = df_slice.index[-1].date().isoformat()

    return fig, info, new_start, new_end


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import webbrowser
    from threading import Timer

    port = 8050
    url = f"http://127.0.0.1:{port}"
    Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"Starting server at {url} ...")
    app.run(debug=False, port=port)
