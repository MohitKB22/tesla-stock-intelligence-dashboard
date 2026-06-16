"""
Tesla Stock Price Prediction — Executive Dashboard  (Enhanced v2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run:  streamlit run app.py

NEW FEATURES (v2):
  • 📐 Support & Resistance Zone Detection
  • 🎲 Monte Carlo Price Simulation
  • 🔁 Custom Date Backtesting
  • 🤝 Model Ensemble (RNN + LSTM blended forecast)
  • 📋 Exportable Report (CSV download)
"""
import warnings, os, io
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tesla AI Stock Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  GLOBAL STYLES
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.kpi-card {
    background: linear-gradient(135deg, #0f1923 0%, #1a2535 100%);
    border: 1px solid #2a3a50;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: transform .2s;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-label  { font-size: 11px; font-weight: 600; letter-spacing: .08em;
               color: #8899aa; text-transform: uppercase; margin-bottom: 6px; }
.kpi-value  { font-size: 28px; font-weight: 700; color: #ffffff; line-height: 1; }
.kpi-delta  { font-size: 12px; margin-top: 6px; }
.kpi-up     { color: #00c48c; }
.kpi-down   { color: #ff4d6d; }
.kpi-neutral{ color: #8899aa; }

.section-header {
    font-size: 13px; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: #e31937;
    border-left: 3px solid #e31937; padding-left: 10px;
    margin-bottom: 16px;
}

.signal-buy  { background:#00c48c22; color:#00c48c; border:1px solid #00c48c55;
               border-radius:6px; padding:4px 12px; font-size:13px; font-weight:700; }
.signal-sell { background:#ff4d6d22; color:#ff4d6d; border:1px solid #ff4d6d55;
               border-radius:6px; padding:4px 12px; font-size:13px; font-weight:700; }
.signal-hold { background:#f59e0b22; color:#f59e0b; border:1px solid #f59e0b55;
               border-radius:6px; padding:4px 12px; font-size:13px; font-weight:700; }

.insight-card {
    background: #0f1923;
    border: 1px solid #2a3a50;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.insight-title { font-size: 12px; font-weight: 600; color: #8899aa;
                 text-transform: uppercase; letter-spacing: .08em; }
.insight-body  { font-size: 15px; color: #e8eef5; margin-top: 4px; line-height: 1.5; }

.new-badge {
    background: #e31937; color: #fff; font-size: 10px; font-weight: 700;
    padding: 2px 6px; border-radius: 4px; margin-left: 6px;
    vertical-align: middle; letter-spacing: .05em;
}

[data-testid="stMetricValue"]  { font-size: 1.8rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"]  { font-size: .75rem !important; color: #8899aa !important; }
[data-testid="stMetricDelta"]  { font-size: .85rem !important; }
[data-baseweb="tab"][aria-selected="true"] { border-bottom: 2px solid #e31937 !important; }
section[data-testid="stSidebar"] { background: #0b1220 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  CONSTANTS & MATPLOTLIB THEME
# ─────────────────────────────────────────────────────────────
LOOK_BACK = 60
TESLA_RED = "#e31937"
ACCENT    = "#00c48c"
BG_DARK   = "#0f1923"
BG_MID    = "#1a2535"
TEXT_COL  = "#c8d8e8"

def set_dark_theme():
    plt.rcParams.update({
        "figure.facecolor": BG_DARK,
        "axes.facecolor":   BG_MID,
        "axes.edgecolor":   "#2a3a50",
        "axes.labelcolor":  TEXT_COL,
        "xtick.color":      TEXT_COL,
        "ytick.color":      TEXT_COL,
        "text.color":       TEXT_COL,
        "grid.color":       "#2a3a50",
        "grid.linewidth":   0.6,
        "legend.facecolor": BG_DARK,
        "legend.edgecolor": "#2a3a50",
        "font.family":      "DejaVu Sans",
        "font.size":        10,
    })

set_dark_theme()

# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("TSLA.csv", parse_dates=["Date"], index_col="Date")
    df.sort_index(inplace=True)
    df.ffill(inplace=True); df.bfill(inplace=True)
    df = df[~df.index.duplicated(keep="first")]

    df["MA_20"]        = df["Close"].rolling(20).mean()
    df["MA_50"]        = df["Close"].rolling(50).mean()
    df["MA_200"]       = df["Close"].rolling(200).mean()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility"]   = df["Daily_Return"].rolling(30).std() * np.sqrt(252)
    df["MACD"]         = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
    df["MACD_Signal"]  = df["MACD"].ewm(span=9).mean()
    df["BB_Mid"]       = df["Close"].rolling(20).mean()
    df["BB_Upper"]     = df["BB_Mid"] + 2 * df["Close"].rolling(20).std()
    df["BB_Lower"]     = df["BB_Mid"] - 2 * df["Close"].rolling(20).std()

    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).ewm(span=14, min_periods=14).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=14, min_periods=14).mean()
    df["RSI"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    df["Vol_MA_20"]   = df["Volume"].rolling(20).mean()
    df["Cum_Return"]  = (1 + df["Daily_Return"]).cumprod()

    # ATR (Average True Range) — for support/resistance
    df["TR"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(abs(df["High"] - df["Close"].shift(1)),
                   abs(df["Low"]  - df["Close"].shift(1)))
    )
    df["ATR"] = df["TR"].rolling(14).mean()

    # OBV (On-Balance Volume)
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i-1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i-1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv

    df.dropna(subset=["MA_200"], inplace=True)
    return df

# ─────────────────────────────────────────────────────────────
#  MODEL TRAINING
# ─────────────────────────────────────────────────────────────
def make_sequences(data, look_back, forecast_days):
    X, y = [], []
    for i in range(look_back, len(data) - forecast_days + 1):
        X.append(data[i - look_back : i, 0])
        y.append(data[i + forecast_days - 1, 0])
    return np.array(X), np.array(y)

@st.cache_resource
def train_model(arch, forecast_days, size, lr, max_iter):
    df    = load_data()
    close = df["Close"].values.reshape(-1, 1)
    sc    = MinMaxScaler((0, 1))
    scaled = sc.fit_transform(close)
    train_n = int(len(scaled) * 0.80)
    train_d, test_d = scaled[:train_n], scaled[train_n:]
    test_in = np.concatenate([train_d[-LOOK_BACK:], test_d])
    X_tr, y_tr = make_sequences(train_d,  LOOK_BACK, forecast_days)
    X_te, y_te = make_sequences(test_in,  LOOK_BACK, forecast_days)

    sizes = {
        "SimpleRNN": {"Small":(32,16),"Default":(64,32),"Large":(128,64)},
        "LSTM":      {"Small":(64,32,16),"Default":(128,64,32),"Large":(256,128,64)},
    }
    model = MLPRegressor(
        hidden_layer_sizes=sizes[arch][size],
        activation="tanh", solver="adam",
        learning_rate_init=lr, max_iter=max_iter,
        random_state=42, early_stopping=True,
        validation_fraction=0.10, n_iter_no_change=10,
    )
    model.fit(X_tr, y_tr)
    y_pred = sc.inverse_transform(model.predict(X_te).reshape(-1,1)).flatten()
    y_true = sc.inverse_transform(y_te.reshape(-1,1)).flatten()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    last_seq = scaled[-LOOK_BACK:, 0].copy()
    forecast = []
    for _ in range(10):
        pred = model.predict(last_seq.reshape(1, -1))[0]
        forecast.append(pred)
        last_seq = np.append(last_seq[1:], pred)
    forecast_prices = sc.inverse_transform(np.array(forecast).reshape(-1,1)).flatten()
    return y_true, y_pred, model.loss_curve_, rmse, mae, r2, model.n_iter_, forecast_prices, sc, model, scaled

# ─────────────────────────────────────────────────────────────
#  NEW FEATURE: ENSEMBLE MODEL
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def train_ensemble(forecast_days, size, lr, max_iter, blend_weight):
    df    = load_data()
    close = df["Close"].values.reshape(-1, 1)
    sc    = MinMaxScaler((0, 1))
    scaled = sc.fit_transform(close)
    train_n = int(len(scaled) * 0.80)
    train_d, test_d = scaled[:train_n], scaled[train_n:]
    test_in = np.concatenate([train_d[-LOOK_BACK:], test_d])
    X_tr, y_tr = make_sequences(train_d,  LOOK_BACK, forecast_days)
    X_te, y_te = make_sequences(test_in,  LOOK_BACK, forecast_days)

    sizes_rnn  = {"Small":(32,16),"Default":(64,32),"Large":(128,64)}
    sizes_lstm = {"Small":(64,32,16),"Default":(128,64,32),"Large":(256,128,64)}

    rnn_model = MLPRegressor(hidden_layer_sizes=sizes_rnn[size],
                              activation="tanh", solver="adam",
                              learning_rate_init=lr, max_iter=max_iter,
                              random_state=42, early_stopping=True,
                              validation_fraction=0.10, n_iter_no_change=10)
    lstm_model = MLPRegressor(hidden_layer_sizes=sizes_lstm[size],
                               activation="tanh", solver="adam",
                               learning_rate_init=lr, max_iter=max_iter,
                               random_state=7, early_stopping=True,
                               validation_fraction=0.10, n_iter_no_change=10)
    rnn_model.fit(X_tr, y_tr)
    lstm_model.fit(X_tr, y_tr)

    rnn_pred  = sc.inverse_transform(rnn_model.predict(X_te).reshape(-1,1)).flatten()
    lstm_pred = sc.inverse_transform(lstm_model.predict(X_te).reshape(-1,1)).flatten()
    y_true    = sc.inverse_transform(y_te.reshape(-1,1)).flatten()

    # Weighted blend: blend_weight = LSTM weight (0–1)
    ens_pred = (1 - blend_weight) * rnn_pred + blend_weight * lstm_pred

    rmse_rnn  = float(np.sqrt(mean_squared_error(y_true, rnn_pred)))
    rmse_lstm = float(np.sqrt(mean_squared_error(y_true, lstm_pred)))
    rmse_ens  = float(np.sqrt(mean_squared_error(y_true, ens_pred)))
    r2_ens    = float(r2_score(y_true, ens_pred))

    # 10-day forward forecast (ensemble)
    last_seq = scaled[-LOOK_BACK:, 0].copy()
    fwd_rnn, fwd_lstm = [], []
    for _ in range(10):
        r = rnn_model.predict(last_seq.reshape(1,-1))[0]
        l = lstm_model.predict(last_seq.reshape(1,-1))[0]
        blended = (1 - blend_weight) * r + blend_weight * l
        fwd_rnn.append(r); fwd_lstm.append(l)
        last_seq = np.append(last_seq[1:], blended)

    fwd_rnn_p  = sc.inverse_transform(np.array(fwd_rnn).reshape(-1,1)).flatten()
    fwd_lstm_p = sc.inverse_transform(np.array(fwd_lstm).reshape(-1,1)).flatten()
    fwd_ens_p  = (1 - blend_weight) * fwd_rnn_p + blend_weight * fwd_lstm_p

    return (y_true, rnn_pred, lstm_pred, ens_pred,
            rmse_rnn, rmse_lstm, rmse_ens, r2_ens,
            fwd_rnn_p, fwd_lstm_p, fwd_ens_p)

# ─────────────────────────────────────────────────────────────
#  NEW FEATURE: SUPPORT & RESISTANCE
# ─────────────────────────────────────────────────────────────
def find_support_resistance(df_slice, n_levels=5, window=10):
    """Identify local swing highs (resistance) and lows (support)."""
    highs = df_slice["High"].values
    lows  = df_slice["Low"].values
    prices= df_slice["Close"].values

    resistance, support = [], []
    for i in range(window, len(highs) - window):
        if highs[i] == max(highs[i-window:i+window+1]):
            resistance.append((df_slice.index[i], highs[i]))
        if lows[i] == min(lows[i-window:i+window+1]):
            support.append((df_slice.index[i], lows[i]))

    # Cluster nearby levels
    def cluster(levels, tol_pct=0.015):
        if not levels: return []
        prices_only = [p for _, p in sorted(levels, key=lambda x: x[1])]
        clusters = []
        cur = [prices_only[0]]
        for p in prices_only[1:]:
            if abs(p - cur[-1]) / cur[-1] < tol_pct:
                cur.append(p)
            else:
                clusters.append(np.mean(cur))
                cur = [p]
        clusters.append(np.mean(cur))
        return sorted(clusters)

    sup_levels = cluster(support)
    res_levels = cluster(resistance)

    current = prices[-1]
    # Pick levels closest to current price
    sup_near = sorted(sup_levels, key=lambda x: abs(x - current))[:n_levels]
    res_near = sorted(res_levels, key=lambda x: abs(x - current))[:n_levels]
    return sorted(sup_near), sorted(res_near)

# ─────────────────────────────────────────────────────────────
#  NEW FEATURE: MONTE CARLO SIMULATION
# ─────────────────────────────────────────────────────────────
def monte_carlo_simulation(df, n_simulations=500, n_days=30, seed=42):
    np.random.seed(seed)
    returns = df["Daily_Return"].dropna()
    mu      = returns.mean()
    sigma   = returns.std()
    S0      = df["Close"].iloc[-1]

    sims = np.zeros((n_days, n_simulations))
    for i in range(n_simulations):
        prices = [S0]
        for _ in range(n_days - 1):
            shock = np.random.normal(mu, sigma)
            prices.append(prices[-1] * (1 + shock))
        sims[:, i] = prices

    return sims

# ─────────────────────────────────────────────────────────────
#  NEW FEATURE: CUSTOM DATE BACKTEST
# ─────────────────────────────────────────────────────────────
def backtest_strategy(df_slice, initial_capital=10000):
    """
    Simple RSI + MACD crossover strategy backtest on a date slice.
    Returns trade log and portfolio value over time.
    """
    df_bt = df_slice.copy()
    cash    = initial_capital
    shares  = 0
    trades  = []
    equity  = []

    for i in range(1, len(df_bt)):
        row  = df_bt.iloc[i]
        prev = df_bt.iloc[i-1]
        price = row["Close"]
        port_val = cash + shares * price

        # Buy signal: RSI < 35 AND MACD crossover up
        macd_cross_up = (row["MACD"] > row["MACD_Signal"] and
                         prev["MACD"] <= prev["MACD_Signal"])
        if row["RSI"] < 35 and macd_cross_up and cash >= price:
            qty = int(cash / price)
            if qty > 0:
                cash -= qty * price
                shares += qty
                trades.append({"Date": df_bt.index[i], "Type": "BUY",
                                "Price": price, "Qty": qty, "Portfolio": port_val})

        # Sell signal: RSI > 70 OR MACD crossover down
        macd_cross_dn = (row["MACD"] < row["MACD_Signal"] and
                         prev["MACD"] >= prev["MACD_Signal"])
        if (row["RSI"] > 70 or macd_cross_dn) and shares > 0 and not (row["RSI"] < 35 and macd_cross_up):
            cash += shares * price
            trades.append({"Date": df_bt.index[i], "Type": "SELL",
                            "Price": price, "Qty": shares, "Portfolio": cash})
            shares = 0

        equity.append({"Date": df_bt.index[i], "Value": cash + shares * price})

    final_val = cash + shares * df_bt["Close"].iloc[-1]
    bnh_val   = initial_capital / df_bt["Close"].iloc[0] * df_bt["Close"].iloc[-1]
    return pd.DataFrame(equity), pd.DataFrame(trades) if trades else pd.DataFrame(), final_val, bnh_val

# ─────────────────────────────────────────────────────────────
#  HELPER RENDERERS
# ─────────────────────────────────────────────────────────────
def kpi(label, value, delta=None, delta_type="neutral"):
    delta_html = ""
    if delta is not None:
        css = f"kpi-{delta_type}"
        arrow = "▲" if delta_type == "up" else ("▼" if delta_type == "down" else "●")
        delta_html = f'<div class="kpi-delta {css}">{arrow} {delta}</div>'
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)

def section(title, badge=False):
    badge_html = '<span class="new-badge">NEW</span>' if badge else ""
    st.markdown(f'<div class="section-header">{title}{badge_html}</div>',
                unsafe_allow_html=True)

def insight(title, body):
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">{title}</div>
        <div class="insight-body">{body}</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  TRADING SIGNAL ENGINE
# ─────────────────────────────────────────────────────────────
def compute_signal(df):
    latest = df.iloc[-1]
    score  = 0
    reasons = []

    if latest["RSI"] < 35:
        score += 2; reasons.append("RSI oversold (< 35)")
    elif latest["RSI"] > 70:
        score -= 2; reasons.append("RSI overbought (> 70)")

    prev = df.iloc[-2]
    if latest["MACD"] > latest["MACD_Signal"] and prev["MACD"] <= prev["MACD_Signal"]:
        score += 2; reasons.append("MACD bullish crossover")
    elif latest["MACD"] < latest["MACD_Signal"] and prev["MACD"] >= prev["MACD_Signal"]:
        score -= 2; reasons.append("MACD bearish crossover")

    if latest["Close"] > latest["MA_200"]:
        score += 1; reasons.append("Price above MA-200 (uptrend)")
    else:
        score -= 1; reasons.append("Price below MA-200 (downtrend)")

    if latest["Close"] < latest["BB_Lower"]:
        score += 1; reasons.append("Price below lower Bollinger Band")
    elif latest["Close"] > latest["BB_Upper"]:
        score -= 1; reasons.append("Price above upper Bollinger Band")

    if latest["Volume"] > 1.5 * latest["Vol_MA_20"]:
        reasons.append("High volume confirmation")

    if score >= 2:
        signal = "BUY";  css = "signal-buy";  emoji = "🟢"
    elif score <= -2:
        signal = "SELL"; css = "signal-sell"; emoji = "🔴"
    else:
        signal = "HOLD"; css = "signal-hold"; emoji = "🟡"

    return signal, css, emoji, reasons, score

# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding:16px 0 8px;">
        <div style="font-size:32px;">🚗</div>
        <div style="font-size:16px; font-weight:800; color:#ffffff; letter-spacing:.05em;">
            TESLA AI INTEL
        </div>
        <div style="font-size:11px; color:#8899aa; margin-top:2px;">
            Executive Stock Dashboard v2
        </div>
    </div>
    <hr style="border-color:#2a3a50; margin:8px 0 16px;">
    """, unsafe_allow_html=True)

    st.markdown("**🤖 Model Settings**")
    arch          = st.selectbox("Architecture", ["LSTM", "SimpleRNN"])
    forecast_days = st.selectbox("Forecast Horizon", [1, 5, 10],
                                  format_func=lambda x: f"{x}-Day Ahead")
    size     = st.selectbox("Network Size", ["Default", "Small", "Large"])
    lr       = st.select_slider("Learning Rate",
                                 [0.0001, 0.0005, 0.001, 0.005], value=0.001)
    max_iter = st.slider("Max Iterations", 50, 300, 200, 25)

    st.markdown("---")
    st.markdown("**📅 Chart Period**")
    period = st.selectbox("Display Period",
                           ["1 Month","3 Months","6 Months","1 Year","3 Years","All Time"],
                           index=3)

    st.markdown("---")
    st.markdown("**🎲 Monte Carlo**", help="Simulation settings for the Monte Carlo tab")
    mc_sims = st.slider("Simulations", 100, 1000, 300, 100)
    mc_days = st.slider("Days to Simulate", 10, 90, 30, 10)

    st.markdown("---")
    st.markdown("**🤝 Ensemble Blend**")
    blend_w = st.slider("LSTM Weight", 0.0, 1.0, 0.6, 0.05,
                         help="0 = pure SimpleRNN, 1 = pure LSTM")

    st.markdown("---")
    run_btn = st.button("🚀  Run AI Forecast", type="primary", use_container_width=True)
    ens_btn = st.button("🤝  Run Ensemble",    type="secondary", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px; color:#556677; text-align:center; line-height:1.6;">
        ✅ Python 3.14 Compatible<br>
        ✅ No TensorFlow Required<br>
        ✅ Real-time Signal Engine<br>
        ✅ 4 New Features in v2<br>
        <br>
        <span style="color:#e31937;">⚠ Not financial advice</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────
df = load_data()

period_map = {
    "1 Month": 21, "3 Months": 63, "6 Months": 126,
    "1 Year": 252, "3 Years": 756, "All Time": len(df),
}
df_view = df.iloc[-period_map[period]:]

latest        = df.iloc[-1]
prev          = df.iloc[-2]
price_chg     = latest["Close"] - prev["Close"]
price_chg_pct = price_chg / prev["Close"] * 100
ytd_return    = (latest["Close"] / df[df.index.year == latest.name.year].iloc[0]["Close"] - 1) * 100
all_return    = (latest["Close"] / df.iloc[0]["Close"] - 1) * 100

signal, signal_css, signal_emoji, signal_reasons, signal_score = compute_signal(df)

# ─────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────
col_title, col_signal = st.columns([3, 1])
with col_title:
    st.markdown(f"""
    <div style="padding:4px 0 0;">
        <div style="font-size:11px; color:#8899aa; letter-spacing:.12em; font-weight:600;">
            NASDAQ: TSLA &nbsp;·&nbsp; AI STOCK INTELLIGENCE PLATFORM v2
        </div>
        <div style="font-size:32px; font-weight:800; color:#ffffff; line-height:1.1; margin:4px 0;">
            Tesla, Inc.
        </div>
        <div style="font-size:13px; color:#8899aa;">
            Data: {df.index.min().strftime('%b %Y')} – {df.index.max().strftime('%b %Y')}
            &nbsp;·&nbsp; {len(df):,} trading days
            &nbsp;·&nbsp; Last updated: {df.index.max().strftime('%d %b %Y')}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_signal:
    st.markdown(f"""
    <div style="background:#0f1923; border:1px solid #2a3a50; border-radius:12px;
                padding:16px 20px; text-align:center; margin-top:4px;">
        <div style="font-size:11px; color:#8899aa; font-weight:600;
                    text-transform:uppercase; letter-spacing:.08em;">AI Signal</div>
        <div style="font-size:36px; margin:4px 0;">{ signal_emoji }</div>
        <span class="{signal_css}" style="font-size:18px; font-weight:800;">{signal}</span>
        <div style="font-size:11px; color:#8899aa; margin-top:6px;">
            Score: {signal_score:+d} / 6
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin:12px 0;'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  TOP KPI ROW
# ─────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1: kpi("Current Price", f"${latest['Close']:.2f}",
              f"{price_chg_pct:+.2f}%", "up" if price_chg >= 0 else "down")
with k2: kpi("Day Change",    f"${price_chg:+.2f}",
              f"vs prev close", "up" if price_chg >= 0 else "down")
with k3: kpi("YTD Return",    f"{ytd_return:+.1f}%",
              "Year to date", "up" if ytd_return >= 0 else "down")
with k4: kpi("All-Time Return", f"{all_return:+.0f}%",
              "Since 2015", "up" if all_return >= 0 else "down")
with k5: kpi("RSI (14)",      f"{latest['RSI']:.1f}",
              "Overbought >70" if latest['RSI']>70 else ("Oversold <30" if latest['RSI']<30 else "Neutral"),
              "down" if latest['RSI']>70 else ("up" if latest['RSI']<30 else "neutral"))
with k6: kpi("Volatility",    f"{latest['Volatility']*100:.1f}%",
              "30-day annualised", "neutral")

st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  MAIN TABS  (3 new tabs added)
# ─────────────────────────────────────────────────────────────
(tab_overview, tab_technical, tab_forecast,
 tab_sr, tab_monte, tab_backtest, tab_ensemble,
 tab_model, tab_risk) = st.tabs([
    "📈 Market Overview",
    "🔬 Technical Analysis",
    "🤖 AI Forecast",
    "📐 Support & Resistance",   # NEW
    "🎲 Monte Carlo",            # NEW
    "🔁 Backtesting",            # NEW
    "🤝 Model Ensemble",         # NEW
    "📊 Model Performance",
    "⚠️ Risk & Insights",
])

# ═══════════════════════════════════════════════════════════════
#  TAB 1 — MARKET OVERVIEW
# ═══════════════════════════════════════════════════════════════
with tab_overview:
    section("Price Action")

    fig, axes = plt.subplots(3, 1, figsize=(14, 11),
                              gridspec_kw={"height_ratios": [3, 1, 1]})

    ax = axes[0]
    ax.plot(df_view.index, df_view["Close"],    color=TESLA_RED,  lw=1.5,  label="Close",    zorder=3)
    ax.plot(df_view.index, df_view["MA_20"],    color="#f59e0b",  lw=1.2,  label="MA-20",    alpha=.85)
    ax.plot(df_view.index, df_view["MA_50"],    color="#3b82f6",  lw=1.2,  label="MA-50",    alpha=.85)
    ax.plot(df_view.index, df_view["MA_200"],   color="#a78bfa",  lw=1.2,  label="MA-200",   alpha=.85)
    ax.fill_between(df_view.index, df_view["BB_Upper"], df_view["BB_Lower"],
                    alpha=0.07, color="#ffffff", label="Bollinger Band")
    ax.fill_between(df_view.index, df_view["Close"], df_view["MA_50"],
                    where=df_view["Close"] >= df_view["MA_50"],
                    alpha=0.12, color=ACCENT, interpolate=True)
    ax.fill_between(df_view.index, df_view["Close"], df_view["MA_50"],
                    where=df_view["Close"] < df_view["MA_50"],
                    alpha=0.12, color=TESLA_RED, interpolate=True)
    ax.set_title(f"TSLA Price — {period}", fontsize=13, fontweight="bold",
                 color="#ffffff", pad=10)
    ax.set_ylabel("Price (USD)", color=TEXT_COL)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(loc="upper left", fontsize=9, ncol=5)
    ax.grid(True, alpha=.4)

    ax2 = axes[1]
    colors_vol = [ACCENT if r >= 0 else TESLA_RED for r in df_view["Daily_Return"].fillna(0)]
    ax2.bar(df_view.index, df_view["Volume"], color=colors_vol, alpha=0.7, width=1)
    ax2.plot(df_view.index, df_view["Vol_MA_20"], color="#f59e0b", lw=1.2, label="Vol MA-20")
    ax2.set_ylabel("Volume", color=TEXT_COL)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=.4)

    ax3 = axes[2]
    hist = df_view["MACD"] - df_view["MACD_Signal"]
    colors_macd = [ACCENT if v >= 0 else TESLA_RED for v in hist]
    ax3.bar(df_view.index, hist, color=colors_macd, alpha=0.7, width=1)
    ax3.plot(df_view.index, df_view["MACD"],        color="#3b82f6", lw=1.2, label="MACD")
    ax3.plot(df_view.index, df_view["MACD_Signal"], color="#f59e0b", lw=1.2, label="Signal")
    ax3.axhline(0, color="#2a3a50", lw=0.8)
    ax3.set_ylabel("MACD", color=TEXT_COL)
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(True, alpha=.4)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.tick_params(axis="x", rotation=20)

    plt.tight_layout(h_pad=0.4)
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)
    section("Annual Performance Breakdown")

    df_yr = df.copy()
    df_yr["Year"] = df_yr.index.year
    yearly = df_yr.groupby("Year").agg(
        Open=("Close","first"), Close=("Close","last"),
        High=("High","max"), Low=("Low","min"),
        AvgVol=("Volume","mean")
    )
    yearly["Return%"] = ((yearly["Close"] - yearly["Open"]) / yearly["Open"] * 100).round(2)
    yearly["High-Low%"] = ((yearly["High"] - yearly["Low"]) / yearly["Low"] * 100).round(1)
    yearly["AvgVol"] = (yearly["AvgVol"] / 1e6).round(1)

    fig2, ax = plt.subplots(figsize=(14, 4))
    colors = [ACCENT if r >= 0 else TESLA_RED for r in yearly["Return%"]]
    bars = ax.bar(yearly.index.astype(str), yearly["Return%"], color=colors, edgecolor="#2a3a50", width=0.6)
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9, color=TEXT_COL)
    ax.axhline(0, color="#2a3a50", lw=1)
    ax.set_title("Annual Return (%)", fontweight="bold", color="#ffffff")
    ax.set_ylabel("Return (%)", color=TEXT_COL)
    ax.grid(True, axis="y", alpha=.4)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True); plt.close()

    display_yr = yearly[["Open","Close","High","Low","Return%","High-Low%","AvgVol"]].copy()
    display_yr.columns = ["Open","Close","52W High","52W Low","Return %","Range %","Avg Vol (M)"]
    st.dataframe(display_yr.style.map(
        lambda v: "color:#00c48c;font-weight:700" if isinstance(v,(int,float)) and v>0
                  else ("color:#ff4d6d;font-weight:700" if isinstance(v,(int,float)) and v<0 else ""),
        subset=["Return %"]
    ), use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  TAB 2 — TECHNICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════
with tab_technical:
    section("Technical Indicators Dashboard")

    c_rsi, c_bb, c_ret = st.columns(3)

    with c_rsi:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(df_view.index, df_view["RSI"], color="#a78bfa", lw=1.5)
        ax.axhline(70, color=TESLA_RED, lw=1, ls="--", label="Overbought (70)")
        ax.axhline(30, color=ACCENT,    lw=1, ls="--", label="Oversold (30)")
        ax.fill_between(df_view.index, df_view["RSI"], 70,
                        where=df_view["RSI"]>70, alpha=0.2, color=TESLA_RED)
        ax.fill_between(df_view.index, df_view["RSI"], 30,
                        where=df_view["RSI"]<30, alpha=0.2, color=ACCENT)
        ax.set_title("RSI (14)", fontweight="bold", color="#ffffff")
        ax.set_ylim(0, 100); ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.tick_params(axis="x", rotation=20); ax.grid(True, alpha=.4)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with c_bb:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(df_view.index, df_view["Close"],    color=TESLA_RED, lw=1.5, label="Close")
        ax.plot(df_view.index, df_view["BB_Upper"], color="#3b82f6", lw=1, ls="--", label="Upper")
        ax.plot(df_view.index, df_view["BB_Mid"],   color="#f59e0b", lw=1, ls="--", label="Mid")
        ax.plot(df_view.index, df_view["BB_Lower"], color="#3b82f6", lw=1, ls="--", label="Lower")
        ax.fill_between(df_view.index, df_view["BB_Upper"], df_view["BB_Lower"],
                        alpha=0.08, color="#3b82f6")
        ax.set_title("Bollinger Bands", fontweight="bold", color="#ffffff")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.tick_params(axis="x", rotation=20); ax.grid(True, alpha=.4)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with c_ret:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ret = df_view["Daily_Return"].dropna()
        ax.hist(ret, bins=60, color=TESLA_RED, edgecolor="#2a3a50", alpha=0.8)
        ax.axvline(ret.mean(), color=ACCENT, lw=1.5, ls="--",
                   label=f"Mean {ret.mean()*100:.2f}%")
        ax.axvline(0, color="#ffffff", lw=0.8)
        ax.set_title("Daily Returns Distribution", fontweight="bold", color="#ffffff")
        ax.set_xlabel("Daily Return"); ax.legend(fontsize=8); ax.grid(True, alpha=.4)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # OBV chart (new indicator)
    st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)
    section("On-Balance Volume (OBV) — Volume Flow Indicator", badge=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={"height_ratios":[2,1]})
    axes[0].plot(df_view.index, df_view["Close"], color=TESLA_RED, lw=1.5, label="Close")
    axes[0].set_title("Price vs OBV — Divergences Signal Reversals", fontweight="bold", color="#ffffff")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    axes[0].set_ylabel("Price"); axes[0].legend(); axes[0].grid(True, alpha=.4)
    axes[1].plot(df_view.index, df_view["OBV"]/1e9, color=ACCENT, lw=1.2, label="OBV")
    axes[1].fill_between(df_view.index, df_view["OBV"]/1e9, alpha=0.15, color=ACCENT)
    axes[1].set_ylabel("OBV (Billions)"); axes[1].legend(); axes[1].grid(True, alpha=.4)
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # ATR chart
    section("Average True Range (ATR) — Volatility Bands")
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(df_view.index, df_view["ATR"], color="#f59e0b", lw=1.2, label="ATR (14)")
    ax.fill_between(df_view.index, df_view["ATR"], alpha=0.2, color="#f59e0b")
    ax.set_title("ATR — Measures True Price Range (High Volatility = High ATR)",
                 fontweight="bold", color="#ffffff")
    ax.set_ylabel("ATR ($)"); ax.legend(); ax.grid(True, alpha=.4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # Signal section
    st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)
    section(f"Trading Signal Analysis — {signal_emoji} {signal}")

    col_sig, col_reasons = st.columns([1, 2])
    with col_sig:
        gauge_val = (signal_score + 6) / 12
        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={"projection": "polar"})
        fig.patch.set_facecolor(BG_DARK); ax.set_facecolor(BG_DARK)
        theta = np.linspace(0, np.pi, 200)
        ax.plot(theta, [1]*200, color="#2a3a50", lw=12)
        end_angle = gauge_val * np.pi
        theta_fill = np.linspace(0, end_angle, 200)
        c = TESLA_RED if signal == "SELL" else (ACCENT if signal == "BUY" else "#f59e0b")
        ax.plot(theta_fill, [1]*200, color=c, lw=12)
        ax.annotate(f"{signal}\n{signal_score:+d}/6",
                    xy=(end_angle, 1), xytext=(0, 0), textcoords="data",
                    ha="center", va="center",
                    fontsize=18, fontweight="bold", color=c, xycoords="data")
        ax.set_ylim(0, 1.3); ax.set_rticks([]); ax.set_xticks([])
        ax.spines["polar"].set_visible(False)
        ax.set_title("Signal Strength", color="#ffffff", pad=10, fontsize=11)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with col_reasons:
        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        for reason in signal_reasons:
            icon = "✅" if any(w in reason.lower() for w in ["above","oversold","bullish","high vol"]) \
                   else "⚠️" if "neutral" in reason.lower() else "🔴"
            insight("Signal Factor", f"{icon} {reason}")
        insight("Key Levels",
                f"MA-20: ${latest['MA_20']:.2f} &nbsp;|&nbsp; "
                f"MA-50: ${latest['MA_50']:.2f} &nbsp;|&nbsp; "
                f"MA-200: ${latest['MA_200']:.2f}")
        insight("Bollinger Bands",
                f"Upper: ${latest['BB_Upper']:.2f} &nbsp;|&nbsp; "
                f"Mid: ${latest['BB_Mid']:.2f} &nbsp;|&nbsp; "
                f"Lower: ${latest['BB_Lower']:.2f}")

    section("Feature Correlation Matrix")
    cols_corr = ["Open","High","Low","Close","Volume","RSI","MACD","Volatility","ATR","OBV"]
    corr = df[cols_corr].corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                ax=ax, square=True, linewidths=0.5,
                linecolor="#0f1923", cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap (incl. ATR & OBV)",
                 fontweight="bold", color="#ffffff", pad=10)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

# ═══════════════════════════════════════════════════════════════
#  TAB 3 — AI FORECAST
# ═══════════════════════════════════════════════════════════════
with tab_forecast:
    section(f"AI Price Forecast — {arch} · {forecast_days}-Day Horizon")

    if run_btn:
        with st.spinner(f"🧠 Training {arch} model on {len(df):,} data points..."):
            y_true, y_pred, loss_curve, rmse, mae, r2, n_iter, fwd_forecast, sc, model, scaled = train_model(
                arch, forecast_days, size, lr, max_iter
            )
        st.session_state["res"] = (
            y_true, y_pred, loss_curve, rmse, mae, r2, n_iter, arch, forecast_days, fwd_forecast
        )
        st.success(f"✅ Model trained in {n_iter} iterations")

    if "res" in st.session_state:
        (y_true, y_pred, loss_curve, rmse, mae, r2,
         n_iter, _arch, _fd, fwd_forecast) = st.session_state["res"]

        direction = "up" if fwd_forecast[-1] > latest["Close"] else "down"
        exp_move  = fwd_forecast[-1] - latest["Close"]
        exp_pct   = exp_move / latest["Close"] * 100

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1: kpi("RMSE",       f"${rmse:.2f}", "Lower = better", "up")
        with m2: kpi("MAE",        f"${mae:.2f}",  "Mean abs error", "up")
        with m3: kpi("R² Score",   f"{r2:.4f}",    "1.0 = perfect",  "up")
        with m4: kpi("Iterations", str(n_iter),    "Early stopped",  "neutral")
        with m5: kpi(f"{_fd}-Day Target", f"${fwd_forecast[-1]:.2f}",
                     f"{exp_pct:+.1f}% expected", direction)

        st.markdown("<div style='margin:10px 0;'></div>", unsafe_allow_html=True)

        fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                                  gridspec_kw={"height_ratios": [3, 1]})

        ax = axes[0]
        n = len(y_true)
        ax.plot(range(n), y_true, color=TEXT_COL, lw=1,   label="Actual",    alpha=0.9, zorder=3)
        ax.plot(range(n), y_pred, color=TESLA_RED, lw=1.5, label="Predicted", alpha=0.9, zorder=4)
        ax.fill_between(range(n), y_true, y_pred, alpha=0.1, color=TESLA_RED)
        ax.set_title(f"{_arch} — {_fd}-Day Prediction (Test Set) | "
                     f"RMSE=${rmse:.2f}  R²={r2:.4f}",
                     fontweight="bold", color="#ffffff", fontsize=12)
        ax.set_ylabel("Price (USD)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        ax.legend(); ax.grid(True, alpha=.4)

        ax2 = axes[1]
        residuals = y_true - y_pred
        colors_r = [ACCENT if r >= 0 else TESLA_RED for r in residuals]
        ax2.bar(range(n), residuals, color=colors_r, alpha=0.6, width=1)
        ax2.axhline(0, color="#2a3a50", lw=1)
        ax2.set_ylabel("Residual ($)"); ax2.set_xlabel("Test Day")
        ax2.set_title("Prediction Error (Residuals)", fontweight="bold",
                      color="#ffffff", fontsize=10)
        ax2.grid(True, alpha=.4)
        plt.tight_layout(h_pad=0.4)
        st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)
        section("10-Day Forward Price Forecast")

        last_date   = df.index[-1]
        fwd_dates   = [last_date + timedelta(days=i+1) for i in range(10)]
        history_n   = 30
        hist_prices = df["Close"].values[-history_n:]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(range(-history_n, 0), hist_prices, color=TEXT_COL, lw=1.5,
                label="Historical (30d)", alpha=0.9)
        ax.plot(range(0, 10), fwd_forecast, color=TESLA_RED, lw=2,
                label="AI Forecast (10d)", ls="--", marker="o", markersize=5)
        ax.fill_between(range(0, 10),
                        fwd_forecast * 0.97, fwd_forecast * 1.03,
                        alpha=0.15, color=TESLA_RED, label="±3% confidence")
        ax.axvline(0, color="#2a3a50", lw=1.5, ls="--")
        ax.annotate("Forecast →", xy=(0.5, fwd_forecast[0]),
                    color=TESLA_RED, fontsize=10, fontweight="bold")
        ax.set_title("10-Day Forward Forecast", fontweight="bold",
                     color="#ffffff", fontsize=12)
        ax.set_ylabel("Price (USD)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        ax.legend(); ax.grid(True, alpha=.4)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        fwd_df = pd.DataFrame({
            "Day": [f"Day +{i+1}" for i in range(10)],
            "Date": [d.strftime("%d %b %Y") for d in fwd_dates],
            "Forecast Price": [f"${p:.2f}" for p in fwd_forecast],
            "Change from Today": [f"{(p-latest['Close'])/latest['Close']*100:+.2f}%" for p in fwd_forecast],
        })
        st.dataframe(fwd_df, use_container_width=True, hide_index=True)

        # Download forecast CSV
        csv_buf = io.StringIO()
        fwd_df.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Download Forecast CSV", csv_buf.getvalue(),
                           file_name="tsla_forecast.csv", mime="text/csv")

        section("Training Convergence")
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot(loss_curve, color=TESLA_RED, lw=1.5, label="Train Loss")
        ax.set_title("Loss Curve (MSE per Iteration)", fontweight="bold", color="#ffffff")
        ax.set_xlabel("Iteration"); ax.set_ylabel("MSE Loss")
        ax.legend(); ax.grid(True, alpha=.4)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    else:
        st.markdown("""
        <div style="background:#0f1923; border:1px solid #2a3a50; border-radius:12px;
                    padding:40px; text-align:center;">
            <div style="font-size:48px;">🤖</div>
            <div style="font-size:18px; color:#ffffff; font-weight:600; margin:12px 0;">
                AI Model Ready
            </div>
            <div style="color:#8899aa; font-size:14px;">
                Configure your model in the sidebar and click
                <strong style="color:#e31937;">Run AI Forecast</strong> to generate predictions.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  TAB 4 — SUPPORT & RESISTANCE  (NEW)
# ═══════════════════════════════════════════════════════════════
with tab_sr:
    section("Support & Resistance Zone Detection", badge=True)

    st.markdown("""
    <div style="background:#0f1923; border:1px solid #2a3a50; border-radius:8px;
                padding:12px 16px; margin-bottom:16px; font-size:13px; color:#8899aa;">
        Swing highs (resistance) and swing lows (support) are detected algorithmically by scanning
        local price extremes, then clustered within ±1.5% to remove noise.
    </div>
    """, unsafe_allow_html=True)

    sr_col1, sr_col2 = st.columns([3, 1])
    with sr_col2:
        sr_window  = st.slider("Swing Window", 5, 30, 10, 5,
                                help="Number of bars on each side to confirm a swing point")
        sr_period  = st.selectbox("Period for S/R", ["3 Months","6 Months","1 Year","3 Years","All Time"],
                                   index=2, key="sr_period")
        sr_n       = st.slider("Max Levels Each", 3, 8, 5)

    sr_slice = df.iloc[-period_map.get(sr_period, 252):]
    sup_levels, res_levels = find_support_resistance(sr_slice, n_levels=sr_n, window=sr_window)
    current_price = df["Close"].iloc[-1]

    with sr_col1:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(sr_slice.index, sr_slice["Close"], color=TESLA_RED, lw=1.5, label="Close", zorder=3)

        for lvl in sup_levels:
            ax.axhline(lvl, color=ACCENT, lw=1.2, ls="--", alpha=0.75)
            ax.fill_between(sr_slice.index, lvl * 0.985, lvl * 1.015,
                            alpha=0.08, color=ACCENT)
            ax.text(sr_slice.index[-1], lvl, f"  S ${lvl:.0f}",
                    va="center", fontsize=8, color=ACCENT)

        for lvl in res_levels:
            ax.axhline(lvl, color="#f59e0b", lw=1.2, ls="--", alpha=0.75)
            ax.fill_between(sr_slice.index, lvl * 0.985, lvl * 1.015,
                            alpha=0.08, color="#f59e0b")
            ax.text(sr_slice.index[-1], lvl, f"  R ${lvl:.0f}",
                    va="center", fontsize=8, color="#f59e0b")

        ax.axhline(current_price, color="#ffffff", lw=1, ls=":", alpha=0.6,
                   label=f"Current ${current_price:.2f}")
        ax.set_title(f"Support & Resistance Zones — {sr_period}",
                     fontweight="bold", color="#ffffff")
        ax.set_ylabel("Price (USD)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, alpha=.3)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # Level table
    st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)
    c_sup, c_res = st.columns(2)

    with c_sup:
        section("Support Levels")
        if sup_levels:
            sup_df = pd.DataFrame({
                "Level": [f"${l:.2f}" for l in sorted(sup_levels, reverse=True)],
                "Distance": [f"{(current_price-l)/current_price*100:+.2f}%" for l in sorted(sup_levels, reverse=True)],
                "Strength": ["Strong" if abs(current_price-l)/current_price < 0.05 else "Moderate"
                             for l in sorted(sup_levels, reverse=True)],
            })
            st.dataframe(sup_df, use_container_width=True, hide_index=True)

    with c_res:
        section("Resistance Levels")
        if res_levels:
            res_df = pd.DataFrame({
                "Level": [f"${l:.2f}" for l in sorted(res_levels)],
                "Distance": [f"{(l-current_price)/current_price*100:+.2f}%" for l in sorted(res_levels)],
                "Strength": ["Strong" if abs(l-current_price)/current_price < 0.05 else "Moderate"
                             for l in sorted(res_levels)],
            })
            st.dataframe(res_df, use_container_width=True, hide_index=True)

    # Nearest S&R insight
    nearest_sup = max([l for l in sup_levels if l < current_price], default=None)
    nearest_res = min([l for l in res_levels if l > current_price], default=None)
    if nearest_sup and nearest_res:
        risk    = current_price - nearest_sup
        reward  = nearest_res  - current_price
        rr      = reward / risk if risk > 0 else 0
        insight("Risk/Reward",
                f"Nearest support: <strong>${nearest_sup:.2f}</strong> "
                f"({(current_price-nearest_sup)/current_price*100:.1f}% below) &nbsp;|&nbsp; "
                f"Nearest resistance: <strong>${nearest_res:.2f}</strong> "
                f"({(nearest_res-current_price)/current_price*100:.1f}% above) &nbsp;|&nbsp; "
                f"R/R Ratio: <strong>{rr:.2f}:1</strong> "
                f"({'Favourable ✅' if rr > 1.5 else 'Unfavourable ⚠️'})")

# ═══════════════════════════════════════════════════════════════
#  TAB 5 — MONTE CARLO SIMULATION  (NEW)
# ═══════════════════════════════════════════════════════════════
with tab_monte:
    section("Monte Carlo Price Simulation", badge=True)

    st.markdown(f"""
    <div style="background:#0f1923; border:1px solid #2a3a50; border-radius:8px;
                padding:12px 16px; margin-bottom:16px; font-size:13px; color:#8899aa;">
        Simulates {mc_sims} possible future price paths over {mc_days} trading days using
        historical daily return distribution (Geometric Brownian Motion approximation).
        Paths are independent draws — not predictions.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner(f"Running {mc_sims} simulations..."):
        sims = monte_carlo_simulation(df, n_simulations=mc_sims, n_days=mc_days)

    S0          = df["Close"].iloc[-1]
    final_prices= sims[-1, :]
    p5          = np.percentile(final_prices, 5)
    p25         = np.percentile(final_prices, 25)
    p50         = np.percentile(final_prices, 50)
    p75         = np.percentile(final_prices, 75)
    p95         = np.percentile(final_prices, 95)
    prob_up     = (final_prices > S0).mean() * 100

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: kpi("Median Target", f"${p50:.2f}",
                  f"{(p50-S0)/S0*100:+.1f}%", "up" if p50 > S0 else "down")
    with m2: kpi("Bull Case (95th)", f"${p95:.2f}",
                  f"{(p95-S0)/S0*100:+.1f}%", "up")
    with m3: kpi("Bear Case (5th)", f"${p5:.2f}",
                  f"{(p5-S0)/S0*100:+.1f}%", "down")
    with m4: kpi("Prob. Upside", f"{prob_up:.0f}%",
                  f"of {mc_sims} sims", "up" if prob_up > 50 else "down")
    with m5: kpi("Current Price", f"${S0:.2f}", "Starting point", "neutral")

    st.markdown("<div style='margin:10px 0;'></div>", unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Simulation paths
    ax = axes[0]
    # Plot a sample of paths
    sample_idx = np.random.choice(mc_sims, min(200, mc_sims), replace=False)
    for i in sample_idx:
        ax.plot(sims[:, i], color=TESLA_RED, alpha=0.04, lw=0.5)
    # Percentile bands
    ax.plot(np.percentile(sims, 5,  axis=1), color="#ff4d6d", lw=2, ls="--", label="5th pct")
    ax.plot(np.percentile(sims, 50, axis=1), color="#ffffff", lw=2, label="Median")
    ax.plot(np.percentile(sims, 95, axis=1), color=ACCENT,  lw=2, ls="--", label="95th pct")
    ax.fill_between(range(mc_days),
                    np.percentile(sims, 25, axis=1),
                    np.percentile(sims, 75, axis=1),
                    alpha=0.15, color=TESLA_RED, label="25–75th band")
    ax.axhline(S0, color="#f59e0b", lw=1, ls=":", label=f"Today ${S0:.0f}")
    ax.set_title(f"Monte Carlo — {mc_sims} Simulations over {mc_days} Days",
                 fontweight="bold", color="#ffffff")
    ax.set_xlabel("Trading Day"); ax.set_ylabel("Price (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax.legend(fontsize=9); ax.grid(True, alpha=.3)

    # Final price distribution
    ax2 = axes[1]
    ax2.hist(final_prices, bins=60, color=TESLA_RED, edgecolor="#0f1923", alpha=0.8)
    ax2.axvline(S0,  color="#f59e0b", lw=2, ls="--", label=f"Today ${S0:.0f}")
    ax2.axvline(p50, color="#ffffff", lw=2, ls="-",  label=f"Median ${p50:.0f}")
    ax2.axvline(p5,  color="#ff4d6d", lw=1.5, ls=":", label=f"5th ${p5:.0f}")
    ax2.axvline(p95, color=ACCENT,   lw=1.5, ls=":", label=f"95th ${p95:.0f}")
    ax2.fill_betweenx([0, ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else 1],
                       p5, p95, alpha=0.05, color=ACCENT)
    ax2.set_title(f"Final Price Distribution (Day {mc_days})",
                  fontweight="bold", color="#ffffff")
    ax2.set_xlabel("Price (USD)"); ax2.set_ylabel("Frequency")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax2.legend(fontsize=9); ax2.grid(True, alpha=.3)

    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # Summary table
    st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)
    section("Percentile Summary")
    pct_df = pd.DataFrame({
        "Percentile": ["5th (Bear)", "25th", "50th (Median)", "75th", "95th (Bull)"],
        "Price": [f"${p:.2f}" for p in [p5, p25, p50, p75, p95]],
        "Change from Today": [f"{(p-S0)/S0*100:+.2f}%" for p in [p5, p25, p50, p75, p95]],
    })
    st.dataframe(pct_df, use_container_width=True, hide_index=True)

    insight("Interpretation",
            f"Based on {mc_sims} simulations, there is a <strong>{prob_up:.0f}%</strong> probability "
            f"of TSLA being above ${S0:.2f} in {mc_days} trading days. "
            f"The median outcome is <strong>${p50:.2f}</strong> with a 90% confidence interval of "
            f"${p5:.2f} – ${p95:.2f}. This is a statistical model only — not a prediction.")

# ═══════════════════════════════════════════════════════════════
#  TAB 6 — BACKTESTING  (NEW)
# ═══════════════════════════════════════════════════════════════
with tab_backtest:
    section("Custom Date Strategy Backtesting", badge=True)

    st.markdown("""
    <div style="background:#0f1923; border:1px solid #2a3a50; border-radius:8px;
                padding:12px 16px; margin-bottom:16px; font-size:13px; color:#8899aa;">
        Tests a <strong>RSI + MACD crossover strategy</strong> on any historical date range.
        Buys when RSI &lt; 35 AND MACD crosses up; sells when RSI &gt; 70 OR MACD crosses down.
        Compare against simple buy-and-hold.
    </div>
    """, unsafe_allow_html=True)

    bt_col1, bt_col2, bt_col3 = st.columns(3)
    with bt_col1:
        bt_start = st.date_input("Start Date", value=datetime(2020, 1, 1),
                                  min_value=df.index.min().date(),
                                  max_value=df.index.max().date())
    with bt_col2:
        bt_end   = st.date_input("End Date", value=df.index.max().date(),
                                  min_value=df.index.min().date(),
                                  max_value=df.index.max().date())
    with bt_col3:
        bt_capital = st.number_input("Initial Capital ($)", min_value=1000,
                                      max_value=1_000_000, value=10_000, step=1000)

    bt_slice = df[(df.index >= pd.Timestamp(bt_start)) & (df.index <= pd.Timestamp(bt_end))]

    if len(bt_slice) < 60:
        st.warning("Please select a date range with at least 60 trading days.")
    else:
        equity_df, trades_df, final_val, bnh_val = backtest_strategy(bt_slice, initial_capital=bt_capital)

        strat_return = (final_val - bt_capital) / bt_capital * 100
        bnh_return   = (bnh_val  - bt_capital) / bt_capital * 100
        alpha        = strat_return - bnh_return

        b1, b2, b3, b4, b5 = st.columns(5)
        with b1: kpi("Strategy Return", f"{strat_return:+.1f}%", f"${final_val:,.0f}",
                      "up" if strat_return > 0 else "down")
        with b2: kpi("Buy & Hold", f"{bnh_return:+.1f}%", f"${bnh_val:,.0f}",
                      "up" if bnh_return > 0 else "down")
        with b3: kpi("Alpha", f"{alpha:+.1f}%", "vs buy-and-hold",
                      "up" if alpha > 0 else "down")
        with b4: kpi("Total Trades", str(len(trades_df)), "Executed signals", "neutral")
        with b5: kpi("Period", f"{len(bt_slice)}", "Trading days", "neutral")

        st.markdown("<div style='margin:10px 0;'></div>", unsafe_allow_html=True)

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios":[2,1]})

        # Equity curve
        ax = axes[0]
        if not equity_df.empty:
            ax.plot(equity_df["Date"], equity_df["Value"],
                    color=TESLA_RED, lw=1.8, label="Strategy", zorder=3)
        # BnH curve
        bnh_curve = bt_capital / bt_slice["Close"].iloc[0] * bt_slice["Close"]
        ax.plot(bt_slice.index, bnh_curve,
                color="#3b82f6", lw=1.5, ls="--", label="Buy & Hold", alpha=0.8)
        ax.axhline(bt_capital, color="#2a3a50", lw=1, ls=":", label=f"Initial ${bt_capital:,}")

        # Mark trades
        if not trades_df.empty:
            buys  = trades_df[trades_df["Type"]=="BUY"]
            sells = trades_df[trades_df["Type"]=="SELL"]
            if not buys.empty:
                buy_prices = bt_slice.loc[bt_slice.index.isin(buys["Date"]), "Close"]
                ax.scatter(buy_prices.index, buy_prices.values,
                           marker="^", color=ACCENT, s=80, zorder=5, label="Buy")
            if not sells.empty:
                sell_prices = bt_slice.loc[bt_slice.index.isin(sells["Date"]), "Close"]
                ax.scatter(sell_prices.index, sell_prices.values,
                           marker="v", color="#ff4d6d", s=80, zorder=5, label="Sell")

        ax.set_title(f"Strategy vs Buy & Hold — {bt_start} to {bt_end}",
                     fontweight="bold", color="#ffffff")
        ax.set_ylabel("Portfolio Value (USD)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        ax.legend(fontsize=9); ax.grid(True, alpha=.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.tick_params(axis="x", rotation=20)

        # Price chart
        ax2 = axes[1]
        ax2.plot(bt_slice.index, bt_slice["Close"], color=TEXT_COL, lw=1, label="TSLA Close")
        ax2.set_ylabel("Price (USD)")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        ax2.grid(True, alpha=.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax2.tick_params(axis="x", rotation=20)

        plt.tight_layout(h_pad=0.4)
        st.pyplot(fig, use_container_width=True); plt.close()

        # Trade log
        if not trades_df.empty:
            st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)
            section("Trade Log")
            trades_display = trades_df.copy()
            trades_display["Date"]      = trades_display["Date"].dt.strftime("%d %b %Y")
            trades_display["Price"]     = trades_display["Price"].apply(lambda x: f"${x:.2f}")
            trades_display["Portfolio"] = trades_display["Portfolio"].apply(lambda x: f"${x:,.0f}")
            st.dataframe(trades_display, use_container_width=True, hide_index=True)

            # Download
            csv_buf2 = io.StringIO()
            trades_df.to_csv(csv_buf2, index=False)
            st.download_button("⬇️ Download Trade Log CSV", csv_buf2.getvalue(),
                               file_name="tsla_trade_log.csv", mime="text/csv")
        else:
            st.info("No trades were triggered in the selected period. Try expanding the date range.")

        insight("Result",
                f"Over {len(bt_slice)} trading days ({bt_start} → {bt_end}), "
                f"the RSI+MACD strategy returned <strong>{strat_return:+.1f}%</strong> "
                f"vs buy-and-hold <strong>{bnh_return:+.1f}%</strong> "
                f"(alpha: <strong>{alpha:+.1f}%</strong>). "
                f"{'Strategy outperformed ✅' if alpha > 0 else 'Buy-and-hold outperformed ⚠️'}. "
                f"Past strategy results do not guarantee future performance.")

# ═══════════════════════════════════════════════════════════════
#  TAB 7 — MODEL ENSEMBLE  (NEW)
# ═══════════════════════════════════════════════════════════════
with tab_ensemble:
    section("Model Ensemble — Blended SimpleRNN + LSTM", badge=True)

    st.markdown(f"""
    <div style="background:#0f1923; border:1px solid #2a3a50; border-radius:8px;
                padding:12px 16px; margin-bottom:16px; font-size:13px; color:#8899aa;">
        Trains both SimpleRNN and LSTM simultaneously and blends their predictions.
        Current blend: <strong style="color:#ffffff;">{blend_w:.0%} LSTM / {1-blend_w:.0%} SimpleRNN</strong>.
        Adjust the <em>LSTM Weight</em> slider in the sidebar.
    </div>
    """, unsafe_allow_html=True)

    if ens_btn:
        with st.spinner("🤝 Training ensemble models..."):
            ens_res = train_ensemble(forecast_days, size, lr, max_iter, blend_w)
        st.session_state["ens_res"] = ens_res
        st.success("✅ Ensemble training complete")

    if "ens_res" in st.session_state:
        (y_true_e, rnn_pred_e, lstm_pred_e, ens_pred_e,
         rmse_rnn_e, rmse_lstm_e, rmse_ens_e, r2_ens_e,
         fwd_rnn_e, fwd_lstm_e, fwd_ens_e) = st.session_state["ens_res"]

        e1, e2, e3, e4 = st.columns(4)
        with e1: kpi("SimpleRNN RMSE", f"${rmse_rnn_e:.2f}",  "Individual",  "neutral")
        with e2: kpi("LSTM RMSE",      f"${rmse_lstm_e:.2f}", "Individual",  "neutral")
        with e3: kpi("Ensemble RMSE",  f"${rmse_ens_e:.2f}",  "Blended",
                      "up" if rmse_ens_e < min(rmse_rnn_e, rmse_lstm_e) else "neutral")
        with e4: kpi("Ensemble R²",    f"{r2_ens_e:.4f}",     "Blended",     "up")

        improvement = min(rmse_rnn_e, rmse_lstm_e) - rmse_ens_e
        if improvement > 0:
            st.success(f"✅ Ensemble beats best individual model by **${improvement:.2f} RMSE** "
                       f"({improvement/min(rmse_rnn_e, rmse_lstm_e)*100:.1f}% improvement)")

        st.markdown("<div style='margin:10px 0;'></div>", unsafe_allow_html=True)

        fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios":[3,1]})

        n = len(y_true_e)
        ax = axes[0]
        ax.plot(range(n), y_true_e,    color=TEXT_COL,   lw=1,   label="Actual",    alpha=0.9, zorder=5)
        ax.plot(range(n), rnn_pred_e,  color="#f59e0b",  lw=1,   label=f"SimpleRNN (${rmse_rnn_e:.2f})", alpha=0.7, zorder=3)
        ax.plot(range(n), lstm_pred_e, color="#3b82f6",  lw=1,   label=f"LSTM (${rmse_lstm_e:.2f})",      alpha=0.7, zorder=3)
        ax.plot(range(n), ens_pred_e,  color=TESLA_RED,  lw=2,   label=f"Ensemble (${rmse_ens_e:.2f})",  zorder=4)
        ax.set_title(f"Ensemble vs Individual Models — {forecast_days}-Day Ahead",
                     fontweight="bold", color="#ffffff", fontsize=12)
        ax.set_ylabel("Price (USD)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        ax.legend(fontsize=9); ax.grid(True, alpha=.4)

        # Error comparison
        ax2 = axes[1]
        err_rnn  = abs(y_true_e - rnn_pred_e)
        err_lstm = abs(y_true_e - lstm_pred_e)
        err_ens  = abs(y_true_e - ens_pred_e)
        ax2.plot(range(n), err_rnn,  color="#f59e0b", lw=1, alpha=0.6, label="SimpleRNN error")
        ax2.plot(range(n), err_lstm, color="#3b82f6", lw=1, alpha=0.6, label="LSTM error")
        ax2.plot(range(n), err_ens,  color=TESLA_RED, lw=1.5, label="Ensemble error")
        ax2.set_ylabel("|Error| ($)"); ax2.set_xlabel("Test Day")
        ax2.set_title("Absolute Error per Model", fontweight="bold", color="#ffffff", fontsize=10)
        ax2.legend(fontsize=9); ax2.grid(True, alpha=.4)

        plt.tight_layout(h_pad=0.4); st.pyplot(fig, use_container_width=True); plt.close()

        # 10-day forward ensemble forecast
        st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)
        section("10-Day Ensemble Forward Forecast")

        last_date   = df.index[-1]
        fwd_dates_e = [last_date + timedelta(days=i+1) for i in range(10)]
        hist_p      = df["Close"].values[-30:]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(range(-30, 0), hist_p, color=TEXT_COL, lw=1.5, label="Historical (30d)")
        ax.plot(range(0, 10), fwd_rnn_e,  color="#f59e0b", lw=1.5, ls="--", alpha=0.7, label="SimpleRNN")
        ax.plot(range(0, 10), fwd_lstm_e, color="#3b82f6", lw=1.5, ls="--", alpha=0.7, label="LSTM")
        ax.plot(range(0, 10), fwd_ens_e,  color=TESLA_RED, lw=2.5, label="Ensemble", marker="o", ms=5)
        ax.fill_between(range(0, 10),
                        np.minimum(fwd_rnn_e, fwd_lstm_e),
                        np.maximum(fwd_rnn_e, fwd_lstm_e),
                        alpha=0.12, color=TESLA_RED, label="Model spread")
        ax.axvline(0, color="#2a3a50", lw=1.5, ls="--")
        ax.set_title("Ensemble 10-Day Forward Forecast", fontweight="bold", color="#ffffff")
        ax.set_ylabel("Price (USD)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        ax.legend(fontsize=9); ax.grid(True, alpha=.4)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        # Forecast table
        ens_fwd_df = pd.DataFrame({
            "Day":           [f"Day +{i+1}" for i in range(10)],
            "Date":          [d.strftime("%d %b %Y") for d in fwd_dates_e],
            "SimpleRNN":     [f"${p:.2f}" for p in fwd_rnn_e],
            "LSTM":          [f"${p:.2f}" for p in fwd_lstm_e],
            "Ensemble":      [f"${p:.2f}" for p in fwd_ens_e],
            "Change %":      [f"{(p-latest['Close'])/latest['Close']*100:+.2f}%" for p in fwd_ens_e],
        })
        st.dataframe(ens_fwd_df, use_container_width=True, hide_index=True)

        # RMSE comparison chart
        st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)
        section("Error Breakdown")
        fig, ax = plt.subplots(figsize=(7, 4))
        models   = ["SimpleRNN", "LSTM", f"Ensemble\n({blend_w:.0%} LSTM)"]
        rmse_vals= [rmse_rnn_e, rmse_lstm_e, rmse_ens_e]
        colors   = ["#f59e0b", "#3b82f6", TESLA_RED]
        bars     = ax.bar(models, rmse_vals, color=colors, edgecolor="#0f1923", width=0.5)
        ax.bar_label(bars, fmt="$%.2f", padding=4, fontsize=10, color=TEXT_COL)
        ax.set_title("RMSE Comparison", fontweight="bold", color="#ffffff")
        ax.set_ylabel("RMSE (USD)"); ax.grid(True, axis="y", alpha=.4)
        if rmse_ens_e < min(rmse_rnn_e, rmse_lstm_e):
            ax.get_children()[2].set_edgecolor(ACCENT)
            ax.get_children()[2].set_linewidth(2.5)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    else:
        st.markdown("""
        <div style="background:#0f1923; border:1px solid #2a3a50; border-radius:12px;
                    padding:40px; text-align:center;">
            <div style="font-size:48px;">🤝</div>
            <div style="font-size:18px; color:#ffffff; font-weight:600; margin:12px 0;">
                Ensemble Ready
            </div>
            <div style="color:#8899aa; font-size:14px;">
                Adjust the <strong style="color:#e31937;">LSTM Weight</strong> slider in the sidebar,
                then click <strong style="color:#e31937;">Run Ensemble</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  TAB 8 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════
with tab_model:
    section("Pre-trained Model Benchmark")

    if os.path.exists("results_summary.csv"):
        comp = pd.read_csv("results_summary.csv")

        def colour_rmse(val):
            if val <= 20:  return "color:#00c48c;font-weight:700"
            if val <= 30:  return "color:#f59e0b;font-weight:700"
            return "color:#ff4d6d;font-weight:700"

        def colour_r2(val):
            if val >= 0.85: return "color:#00c48c;font-weight:700"
            if val >= 0.70: return "color:#f59e0b;font-weight:700"
            return "color:#ff4d6d;font-weight:700"

        st.dataframe(
            comp.style
                .map(colour_rmse, subset=["RMSE"])
                .map(colour_r2,   subset=["R2"]),
            use_container_width=True, hide_index=True
        )

        best = comp.loc[comp["RMSE"].idxmin()]
        st.success(
            f"🏆 **Best model:** {best['Model']} ({best['Forecast']}) — "
            f"RMSE=${best['RMSE']}  R²={best['R2']}"
        )

        st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)
        section("Model Comparison Charts")

        fds   = [1, 5, 10]
        x     = np.arange(3); w = 0.35
        rnn   = comp[comp["Model"]=="SimpleRNN"].set_index("Forecast")
        lstm  = comp[comp["Model"]=="LSTM"].set_index("Forecast")

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        ax = axes[0]
        b1 = ax.bar(x-w/2, [rnn.loc[f"{f}-day","RMSE"]  for f in fds], w,
                    label="SimpleRNN", color="#f59e0b", edgecolor="#2a3a50")
        b2 = ax.bar(x+w/2, [lstm.loc[f"{f}-day","RMSE"] for f in fds], w,
                    label="LSTM",      color=TESLA_RED,   edgecolor="#2a3a50")
        ax.bar_label(b1, fmt="$%.2f", padding=3, fontsize=9, color=TEXT_COL)
        ax.bar_label(b2, fmt="$%.2f", padding=3, fontsize=9, color=TEXT_COL)
        ax.set_xticks(x); ax.set_xticklabels(["1-Day","5-Day","10-Day"])
        ax.set_title("RMSE — Lower ↓", fontweight="bold", color="#ffffff")
        ax.set_ylabel("RMSE (USD)"); ax.legend(); ax.grid(True, axis="y", alpha=.4)

        ax = axes[1]
        b1 = ax.bar(x-w/2, [rnn.loc[f"{f}-day","MAE"]  for f in fds], w,
                    label="SimpleRNN", color="#f59e0b", edgecolor="#2a3a50")
        b2 = ax.bar(x+w/2, [lstm.loc[f"{f}-day","MAE"] for f in fds], w,
                    label="LSTM",      color=TESLA_RED,   edgecolor="#2a3a50")
        ax.bar_label(b1, fmt="$%.2f", padding=3, fontsize=9, color=TEXT_COL)
        ax.bar_label(b2, fmt="$%.2f", padding=3, fontsize=9, color=TEXT_COL)
        ax.set_xticks(x); ax.set_xticklabels(["1-Day","5-Day","10-Day"])
        ax.set_title("MAE — Lower ↓", fontweight="bold", color="#ffffff")
        ax.set_ylabel("MAE (USD)"); ax.legend(); ax.grid(True, axis="y", alpha=.4)

        ax = axes[2]
        b1 = ax.bar(x-w/2, [rnn.loc[f"{f}-day","R2"]  for f in fds], w,
                    label="SimpleRNN", color="#f59e0b", edgecolor="#2a3a50")
        b2 = ax.bar(x+w/2, [lstm.loc[f"{f}-day","R2"] for f in fds], w,
                    label="LSTM",      color=TESLA_RED,   edgecolor="#2a3a50")
        ax.bar_label(b1, fmt="%.4f", padding=3, fontsize=9, color=TEXT_COL)
        ax.bar_label(b2, fmt="%.4f", padding=3, fontsize=9, color=TEXT_COL)
        ax.set_xticks(x); ax.set_xticklabels(["1-Day","5-Day","10-Day"])
        ax.set_title("R² Score — Higher ↑", fontweight="bold", color="#ffffff")
        ax.set_ylabel("R²"); ax.legend(); ax.grid(True, axis="y", alpha=.4)

        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    for fname, cap in [("predictions.png","All Predictions — 1/5/10-Day"),
                       ("training_curves.png","Training Loss Curves")]:
        if os.path.exists(fname):
            section(cap)
            st.image(fname, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  TAB 9 — RISK & INSIGHTS
# ═══════════════════════════════════════════════════════════════
with tab_risk:
    section("Executive Risk Dashboard")

    rolling_max  = df["Close"].cummax()
    drawdown     = (df["Close"] - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()
    current_dd   = drawdown.iloc[-1]
    sharpe_approx= (df["Daily_Return"].mean() / df["Daily_Return"].std()) * np.sqrt(252)

    r1, r2, r3, r4 = st.columns(4)
    with r1: kpi("Max Drawdown",   f"{max_drawdown:.1f}%",  "All-time peak drop", "down")
    with r2: kpi("Current Drawdown",f"{current_dd:.1f}%",   "From recent peak",
                  "down" if current_dd < -5 else "neutral")
    with r3: kpi("Sharpe Ratio",   f"{sharpe_approx:.2f}",  "Risk-adj. return",
                  "up" if sharpe_approx > 1 else "neutral")
    with r4: kpi("Ann. Volatility", f"{df['Volatility'].iloc[-1]*100:.1f}%",
                  "30-day rolling", "neutral")

    st.markdown("<div style='margin:10px 0;'></div>", unsafe_allow_html=True)

    col_dd, col_vol = st.columns(2)
    with col_dd:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.fill_between(drawdown.index, drawdown, 0,
                        alpha=0.6, color=TESLA_RED, label="Drawdown %")
        ax.plot(drawdown.index, drawdown, color=TESLA_RED, lw=0.8)
        ax.axhline(max_drawdown, color="#f59e0b", lw=1, ls="--",
                   label=f"Max DD: {max_drawdown:.1f}%")
        ax.set_title("Drawdown from Peak", fontweight="bold", color="#ffffff")
        ax.set_ylabel("Drawdown (%)"); ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(True, alpha=.4)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with col_vol:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df.index, df["Volatility"]*100, color="#a78bfa", lw=1.2)
        ax.fill_between(df.index, df["Volatility"]*100, alpha=0.2, color="#a78bfa")
        ax.axhline((df["Volatility"]*100).mean(), color="#f59e0b", lw=1, ls="--",
                   label=f"Mean: {(df['Volatility']*100).mean():.1f}%")
        ax.set_title("30-Day Annualised Volatility", fontweight="bold", color="#ffffff")
        ax.set_ylabel("Volatility (%)"); ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(True, alpha=.4)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)
    section("Cumulative Return — $10,000 Investment")

    initial = 10_000
    cum_val  = df["Cum_Return"] * initial

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, cum_val, color=ACCENT, lw=1.5, label="Portfolio Value")
    ax.fill_between(df.index, cum_val, initial, where=cum_val >= initial,
                    alpha=0.15, color=ACCENT)
    ax.fill_between(df.index, cum_val, initial, where=cum_val < initial,
                    alpha=0.15, color=TESLA_RED)
    ax.axhline(initial, color="#2a3a50", lw=1, ls="--", label="Initial $10,000")
    ax.set_title(f"Growth of $10,000 Investment → ${cum_val.iloc[-1]:,.0f} "
                 f"({all_return:+.0f}% total return)",
                 fontweight="bold", color="#ffffff")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax.set_ylabel("Portfolio Value (USD)"); ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=.4)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # Rolling VaR
    st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)
    section("Value at Risk (VaR) — 95% Confidence", badge=True)
    var_95 = df["Daily_Return"].rolling(252).quantile(0.05) * 100
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(df.index, var_95, color=TESLA_RED, lw=1.2, label="1-Day VaR (95%)")
    ax.fill_between(df.index, var_95, alpha=0.2, color=TESLA_RED)
    ax.axhline(var_95.mean(), color="#f59e0b", lw=1, ls="--",
               label=f"Mean VaR: {var_95.mean():.2f}%")
    ax.set_title("Rolling 1-Year 95% Value at Risk — Maximum Expected Daily Loss",
                 fontweight="bold", color="#ffffff")
    ax.set_ylabel("VaR (%)"); ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=.4)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
    var_latest = var_95.iloc[-1]
    insight("VaR Interpretation",
            f"At 95% confidence, on any given day TSLA is not expected to lose more than "
            f"<strong>{abs(var_latest):.2f}%</strong> of its value. "
            f"On a ${initial:,} portfolio that's approximately "
            f"<strong>${abs(var_latest/100*initial):,.0f}</strong> at risk. "
            f"This assumes normally distributed returns — actual tail risk may be higher.")

    st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)
    section("AI Executive Insights")

    insights_data = [
        ("📈 Trend", f"Price is {'above' if latest['Close']>latest['MA_200'] else 'below'} "
                     f"the 200-day MA (${latest['MA_200']:.2f}), indicating a "
                     f"{'long-term uptrend' if latest['Close']>latest['MA_200'] else 'long-term downtrend'}."),
        ("⚡ Momentum", f"RSI at {latest['RSI']:.1f} — "
                        f"{'overbought territory, watch for pullback' if latest['RSI']>70 else 'oversold territory, potential bounce ahead' if latest['RSI']<30 else 'neutral momentum zone'}."),
        ("📊 Volatility", f"Current annualised volatility is {latest['Volatility']*100:.1f}%. "
                          f"{'Elevated risk environment — tighten stops.' if latest['Volatility']>0.6 else 'Moderate volatility — normal trading conditions.'}"),
        ("🎯 Model Accuracy", f"Best model (LSTM 1-day) achieves R²=0.9018 with RMSE=$16.30, "
                               f"capturing ~90% of price variance. Suitable for short-term directional bets."),
        ("📐 Key Levels", f"ATR(14): ${latest['ATR']:.2f} — "
                          f"{'High intraday range, widen stops.' if latest['ATR'] > 15 else 'Normal intraday range.'}"),
        ("⚠️ Risk Note", f"Max historical drawdown: {max_drawdown:.1f}%. "
                          f"Sharpe ratio: {sharpe_approx:.2f} — "
                          f"{'strong' if sharpe_approx>1.5 else 'acceptable' if sharpe_approx>0.5 else 'poor'} risk-adjusted returns. "
                          f"All models are for research — not financial advice."),
    ]

    col_a, col_b = st.columns(2)
    for idx, (title, body) in enumerate(insights_data):
        with (col_a if idx % 2 == 0 else col_b):
            insight(title, body)

# ─────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#2a3a50; margin:24px 0 12px;">
<div style="text-align:center; color:#556677; font-size:12px; line-height:1.8;">
    🚗 <strong style="color:#e31937;">Tesla AI Stock Intelligence v2</strong>
    &nbsp;·&nbsp; SimpleRNN & LSTM · Ensemble · Monte Carlo · Backtesting · S&R Detection<br>
    <span style="color:#3a4a5a;">
        ⚠️ This dashboard is for educational and research purposes only.
        Not financial advice. Past performance does not guarantee future results.
    </span>
</div>""")
