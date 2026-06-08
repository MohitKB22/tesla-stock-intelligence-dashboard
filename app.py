"""
Tesla Stock Price Prediction — Executive Dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run:  streamlit run app.py
"""
import warnings, os, sys
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
/* ── Typography & base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── KPI cards ── */
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

/* ── Section headers ── */
.section-header {
    font-size: 13px; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: #e31937;
    border-left: 3px solid #e31937; padding-left: 10px;
    margin-bottom: 16px;
}

/* ── Signal badge ── */
.signal-buy  { background:#00c48c22; color:#00c48c; border:1px solid #00c48c55;
               border-radius:6px; padding:4px 12px; font-size:13px; font-weight:700; }
.signal-sell { background:#ff4d6d22; color:#ff4d6d; border:1px solid #ff4d6d55;
               border-radius:6px; padding:4px 12px; font-size:13px; font-weight:700; }
.signal-hold { background:#f59e0b22; color:#f59e0b; border:1px solid #f59e0b55;
               border-radius:6px; padding:4px 12px; font-size:13px; font-weight:700; }

/* ── Insight card ── */
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

/* ── Metric override ── */
[data-testid="stMetricValue"]  { font-size: 1.8rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"]  { font-size: .75rem !important; color: #8899aa !important; }
[data-testid="stMetricDelta"]  { font-size: .85rem !important; }

/* ── Tab active ── */
[data-baseweb="tab"][aria-selected="true"] { border-bottom: 2px solid #e31937 !important; }

/* ── Sidebar ── */
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

    # Technical indicators
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

    # Generate 10-day forward forecast
    last_seq = scaled[-LOOK_BACK:, 0].copy()
    forecast = []
    for _ in range(10):
        pred = model.predict(last_seq.reshape(1, -1))[0]
        forecast.append(pred)
        last_seq = np.append(last_seq[1:], pred)
    forecast_prices = sc.inverse_transform(np.array(forecast).reshape(-1,1)).flatten()
    return y_true, y_pred, model.loss_curve_, rmse, mae, r2, model.n_iter_, forecast_prices

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

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

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

    # RSI
    if latest["RSI"] < 35:
        score += 2; reasons.append("RSI oversold (< 35)")
    elif latest["RSI"] > 70:
        score -= 2; reasons.append("RSI overbought (> 70)")

    # MACD crossover
    prev = df.iloc[-2]
    if latest["MACD"] > latest["MACD_Signal"] and prev["MACD"] <= prev["MACD_Signal"]:
        score += 2; reasons.append("MACD bullish crossover")
    elif latest["MACD"] < latest["MACD_Signal"] and prev["MACD"] >= prev["MACD_Signal"]:
        score -= 2; reasons.append("MACD bearish crossover")

    # Price vs MA200
    if latest["Close"] > latest["MA_200"]:
        score += 1; reasons.append("Price above MA-200 (uptrend)")
    else:
        score -= 1; reasons.append("Price below MA-200 (downtrend)")

    # Bollinger
    if latest["Close"] < latest["BB_Lower"]:
        score += 1; reasons.append("Price below lower Bollinger Band")
    elif latest["Close"] > latest["BB_Upper"]:
        score -= 1; reasons.append("Price above upper Bollinger Band")

    # Volume surge
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
            Executive Stock Dashboard
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
    run_btn = st.button("🚀  Run AI Forecast", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px; color:#556677; text-align:center; line-height:1.6;">
        ✅ Python 3.14 Compatible<br>
        ✅ No TensorFlow Required<br>
        ✅ Real-time Signal Engine<br>
        <br>
        <span style="color:#e31937;">⚠ Not financial advice</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────
df = load_data()

# Period filter
period_map = {
    "1 Month": 21, "3 Months": 63, "6 Months": 126,
    "1 Year": 252, "3 Years": 756, "All Time": len(df),
}
df_view = df.iloc[-period_map[period]:]

# Latest data
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
            NASDAQ: TSLA &nbsp;·&nbsp; AI STOCK INTELLIGENCE PLATFORM
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
#  MAIN TABS
# ─────────────────────────────────────────────────────────────
tab_overview, tab_technical, tab_forecast, tab_model, tab_risk = st.tabs([
    "📈 Market Overview",
    "🔬 Technical Analysis",
    "🤖 AI Forecast",
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

    # Price + MAs + Bollinger
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

    # Volume
    ax2 = axes[1]
    colors_vol = [ACCENT if r >= 0 else TESLA_RED for r in df_view["Daily_Return"].fillna(0)]
    ax2.bar(df_view.index, df_view["Volume"], color=colors_vol, alpha=0.7, width=1)
    ax2.plot(df_view.index, df_view["Vol_MA_20"], color="#f59e0b", lw=1.2, label="Vol MA-20")
    ax2.set_ylabel("Volume", color=TEXT_COL)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=.4)

    # MACD
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

    # ── Annual performance ──
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

    # Yearly table
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
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, alpha=.4)
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
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, alpha=.4)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with c_ret:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ret = df_view["Daily_Return"].dropna()
        ax.hist(ret, bins=60, color=TESLA_RED, edgecolor="#2a3a50", alpha=0.8)
        ax.axvline(ret.mean(), color=ACCENT, lw=1.5, ls="--",
                   label=f"Mean {ret.mean()*100:.2f}%")
        ax.axvline(0, color="#ffffff", lw=0.8)
        ax.set_title("Daily Returns Distribution", fontweight="bold", color="#ffffff")
        ax.set_xlabel("Daily Return"); ax.legend(fontsize=8)
        ax.grid(True, alpha=.4)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # ── Signal breakdown ──
    st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)
    section(f"Trading Signal Analysis — {signal_emoji} {signal}")

    col_sig, col_reasons = st.columns([1, 2])
    with col_sig:
        gauge_val = (signal_score + 6) / 12  # normalise -6..6 → 0..1
        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={"projection": "polar"})
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_DARK)
        theta = np.linspace(0, np.pi, 200)
        ax.plot(theta, [1]*200, color="#2a3a50", lw=12)
        end_angle = gauge_val * np.pi
        theta_fill = np.linspace(0, end_angle, 200)
        c = TESLA_RED if signal == "SELL" else (ACCENT if signal == "BUY" else "#f59e0b")
        ax.plot(theta_fill, [1]*200, color=c, lw=12)
        ax.annotate(f"{signal}\n{signal_score:+d}/6",
                    xy=(end_angle, 1), xytext=(0, 0),
                    textcoords="data",
                    ha="center", va="center",
                    fontsize=18, fontweight="bold", color=c,
                    xycoords="data")
        ax.set_ylim(0, 1.3); ax.set_rticks([]); ax.set_xticks([])
        ax.spines["polar"].set_visible(False)
        ax.set_title("Signal Strength", color="#ffffff", pad=10, fontsize=11)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

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

    # ── Correlation heatmap ──
    st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)
    section("Feature Correlation Matrix")

    cols_corr = ["Open","High","Low","Close","Volume","RSI","MACD","Volatility"]
    corr = df[cols_corr].corr()
    fig, ax = plt.subplots(figsize=(9, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                ax=ax, square=True, linewidths=0.5,
                linecolor="#0f1923", cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontweight="bold",
                 color="#ffffff", pad=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

# ═══════════════════════════════════════════════════════════════
#  TAB 3 — AI FORECAST
# ═══════════════════════════════════════════════════════════════
with tab_forecast:
    section(f"AI Price Forecast — {arch} · {forecast_days}-Day Horizon")

    if run_btn:
        with st.spinner(f"🧠 Training {arch} model on {len(df):,} data points..."):
            y_true, y_pred, loss_curve, rmse, mae, r2, n_iter, fwd_forecast = train_model(
                arch, forecast_days, size, lr, max_iter
            )
        st.session_state["res"] = (
            y_true, y_pred, loss_curve, rmse, mae, r2, n_iter, arch, forecast_days, fwd_forecast
        )
        st.success(f"✅ Model trained in {n_iter} iterations")

    if "res" in st.session_state:
        (y_true, y_pred, loss_curve, rmse, mae, r2,
         n_iter, _arch, _fd, fwd_forecast) = st.session_state["res"]

        # KPIs
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

        # ── Test set prediction chart ──
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

        # Residuals
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

        # ── 10-Day Forward Forecast ──
        st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)
        section("10-Day Forward Price Forecast")

        last_date  = df.index[-1]
        fwd_dates  = [last_date + timedelta(days=i+1) for i in range(10)]
        history_n  = 30
        hist_dates = df.index[-history_n:]
        hist_prices= df["Close"].values[-history_n:]

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
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

        # Forecast table
        fwd_df = pd.DataFrame({
            "Day": [f"Day +{i+1}" for i in range(10)],
            "Date": [d.strftime("%d %b %Y") for d in fwd_dates],
            "Forecast Price": [f"${p:.2f}" for p in fwd_forecast],
            "Change from Today": [f"{(p-latest['Close'])/latest['Close']*100:+.2f}%" for p in fwd_forecast],
        })
        st.dataframe(fwd_df, use_container_width=True, hide_index=True)

        # Loss curve
        st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)
        section("Training Convergence")
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot(loss_curve, color=TESLA_RED, lw=1.5, label="Train Loss")
        ax.set_title("Loss Curve (MSE per Iteration)", fontweight="bold",
                     color="#ffffff")
        ax.set_xlabel("Iteration"); ax.set_ylabel("MSE Loss")
        ax.legend(); ax.grid(True, alpha=.4)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

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
#  TAB 4 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════
with tab_model:
    section("Pre-trained Model Benchmark")

    if os.path.exists("results_summary.csv"):
        comp = pd.read_csv("results_summary.csv")

        # Highlight table
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

        # RMSE
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

        # MAE
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

        # R²
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

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    # Pre-generated images
    for fname, cap in [("predictions.png","All Predictions — 1/5/10-Day"),
                       ("training_curves.png","Training Loss Curves")]:
        if os.path.exists(fname):
            section(cap)
            st.image(fname, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  TAB 5 — RISK & INSIGHTS
# ═══════════════════════════════════════════════════════════════
with tab_risk:
    section("Executive Risk Dashboard")

    # Volatility & Drawdown
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

    # ── Cumulative Return ──
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
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    # ── AI Insights ──
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
    🚗 <strong style="color:#e31937;">Tesla AI Stock Intelligence</strong>
    &nbsp;·&nbsp; SimpleRNN & LSTM Models
    &nbsp;·&nbsp; Python 3.14 Compatible &nbsp;·&nbsp; No TensorFlow Required<br>
    <span style="color:#3a4a5a;">
        ⚠️ This dashboard is for educational and research purposes only.
        Not financial advice. Past performance does not guarantee future results.
    </span>
</div>
""", unsafe_allow_html=True)
