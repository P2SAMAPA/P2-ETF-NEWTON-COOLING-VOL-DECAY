import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-newton-cooling-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}

MACRO_COLS_CORE     = ["VIX", "DXY", "T10Y2Y"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]

# ── Rolling windows (trading days) ────────────────────────────────────────────
WINDOWS = [63, 126, 252, 504]

# ── Vol spike detection ───────────────────────────────────────────────────────
# A vol spike is detected when the rolling realised vol exceeds
# SPIKE_THRESH_SIGMA * long_run_vol. We then fit the Newton cooling law
# to the post-spike vol decay path.
SPIKE_THRESH_SIGMA = 1.5   # threshold: 1.5x long-run vol = spike

# Short-term vol window for "current temperature" σ(t)
VOL_SHORT_WIN = 5          # 5-day realised vol

# Long-run vol window for "ambient temperature" σ_inf
VOL_LONG_WIN  = 63         # 63-day realised vol = long-run mean

# ── Newton cooling law fitting ────────────────────────────────────────────────
# Model: dσ/dt = -k * (σ(t) - σ_inf)
# Solution: σ(t) = σ_inf + (σ_0 - σ_inf) * exp(-k * t)
#
# Fit k and σ_inf via OLS on log-linearised form:
#   log(σ(t) - σ_inf_hat) = log(σ_0 - σ_inf_hat) - k * t
#
# We fit on post-spike decay windows.
# Maximum decay window to fit (days after spike)
MAX_DECAY_WIN  = 21        # fit over up to 21 days post-spike
MIN_DECAY_POINTS = 5       # minimum points needed for reliable fit

# ── Score construction ────────────────────────────────────────────────────────
# Three cooling law signals:
#
#   cooling_rate_score : estimated k (cooling rate constant)
#                        High k → vol decays fast → markets self-correct quickly
#                        Low k  → vol persists → sustained stress
#
#   overshoot_score    : current vol deviation above cooling trajectory
#                        σ(t) >> predicted_σ(t) → vol overshooting → buy signal
#                        σ(t) << predicted_σ(t) → vol undershooting → caution
#
#   ambient_score      : long-run vol σ_inf relative to cross-sectional mean
#                        Low σ_inf → calm long-run regime → positive signal

WEIGHT_COOLING_RATE = 0.40
WEIGHT_OVERSHOOT    = 0.40
WEIGHT_AMBIENT      = 0.20

TOP_N = 3
