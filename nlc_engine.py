"""
nlc_engine.py — Newton's Law of Cooling Vol Decay Engine
=========================================================

Theory
------
**Newton's Law of Cooling (Newton 1701)**

A body at temperature T(t) in an environment of ambient temperature T_inf
cools according to the ODE:

    dT/dt = -k * (T(t) - T_inf)

Solution: T(t) = T_inf + (T_0 - T_inf) * exp(-k * t)

Where:
    T_0   : initial temperature (vol at spike peak)
    T_inf : ambient temperature (long-run mean vol)
    k     : cooling rate constant (how fast vol decays)
    t     : time since spike peak (days)

**Application to Volatility Decay**

We model post-spike volatility decay as Newton cooling:

    σ(t) = σ_inf + (σ_0 - σ_inf) * exp(-k * t)

Where:
    σ(t)   : realised vol at time t after spike (5-day rolling)
    σ_inf  : long-run ambient vol (63-day rolling average)
    σ_0    : vol at spike peak
    k      : cooling rate (fitted via OLS on log-linearised form)
    t      : days elapsed since peak

**Log-linearised fitting (OLS)**

Rearranging: σ(t) - σ_inf = (σ_0 - σ_inf) * exp(-k * t)

Taking log: log(σ(t) - σ_inf) = log(σ_0 - σ_inf) - k * t

This is a linear regression: y = a - k*t, where:
    y = log(σ(t) - σ_inf)
    a = log(σ_0 - σ_inf)
    k = slope (cooling rate, should be positive)

Fitted per spike episode over up to MAX_DECAY_WIN days post-peak.

**Signal Construction**

For each ETF over a rolling window, we identify all spike episodes and fit
a cooling law to each. The signals are:

1. **Cooling rate k** — how fast vol decays after spikes
   High k → markets self-correct quickly → positive (mean-reverting behaviour)
   Low k  → persistent vol → potential continued stress → cautious

2. **Overshoot** — current σ vs predicted σ from cooling law
   σ_today >> σ_predicted → vol overshooting the trajectory → contrarian buy
   σ_today << σ_predicted → vol undershooting → caution

3. **Ambient vol σ_inf** — long-run mean vol relative to universe
   Low σ_inf → structurally calm ETF → positive signal
   High σ_inf → structurally volatile ETF → negative signal

**Distinction from HAR-RV (in suite)**

HAR-RV models vol as a linear combination of daily/weekly/monthly lags.
Newton cooling models the *functional form* of decay after shocks — it
captures the exponential shape of the decay trajectory, which HAR misses.
HAR: σ(t) = a + b*σ_{t-1} + c*σ_{t-5} + d*σ_{t-22}  (AR structure)
NLC: σ(t) = σ_inf + (σ_0 - σ_inf)*exp(-k*t)            (physical decay law)

The two signals are orthogonal: HAR captures autocorrelation in vol levels;
Newton cooling captures the shape of the post-spike decay trajectory.

References
----------
- Newton, I. (1701). Scala graduum caloris. Philosophical Transactions, 22,
  824–829.
- Cont, R. (2001). Empirical properties of asset returns. Quantitative Finance.
- Engle, R.F. (1982). Autoregressive conditional heteroscedasticity.
  Econometrica, 50(4), 987–1007.
- Corsi, F. (2009). A simple approximate long-memory model of realized
  volatility. Journal of Financial Econometrics, 7(2), 174–196.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

import config


# ── Realised vol series ───────────────────────────────────────────────────────

def _rolling_vol(log_ret: np.ndarray, win: int) -> np.ndarray:
    """Annualised rolling realised vol, shape (T,)."""
    T   = len(log_ret)
    vol = np.full(T, np.nan)
    for t in range(win, T+1):
        vol[t-1] = log_ret[t-win:t].std() * np.sqrt(252)
    return vol


# ── Spike detection ───────────────────────────────────────────────────────────

def _detect_spikes(
    vol_short: np.ndarray,    # (T,) short-term vol
    vol_long:  np.ndarray,    # (T,) long-run vol
    thresh:    float,
) -> List[int]:
    """
    Detect vol spike peaks: local maxima where vol_short > thresh * vol_long.
    Returns list of peak indices (positions of spike peaks).
    """
    T = len(vol_short)
    spikes = []
    in_spike = False
    spike_start = 0
    spike_max_idx = 0
    spike_max_val = 0.0

    for t in range(T):
        if np.isnan(vol_short[t]) or np.isnan(vol_long[t]):
            continue
        is_above = vol_short[t] > thresh * vol_long[t]
        if is_above:
            if not in_spike:
                in_spike = True
                spike_start   = t
                spike_max_idx = t
                spike_max_val = vol_short[t]
            elif vol_short[t] > spike_max_val:
                spike_max_idx = t
                spike_max_val = vol_short[t]
        else:
            if in_spike:
                spikes.append(spike_max_idx)
                in_spike = False

    if in_spike:
        spikes.append(spike_max_idx)

    return spikes


# ── Newton cooling law fit ────────────────────────────────────────────────────

def _fit_cooling_law(
    vol_series:  np.ndarray,   # (T,) short-term vol series
    vol_ambient: np.ndarray,   # (T,) long-run vol series
    peak_idx:    int,
    max_decay:   int,
    min_pts:     int,
) -> Optional[Tuple[float, float, float]]:
    """
    Fit Newton cooling law to post-spike vol decay.

    Model: log(σ(t) - σ_inf) = log(σ_0 - σ_inf) - k * t

    Parameters
    ----------
    vol_series  : short-term vol path
    vol_ambient : long-run ambient vol
    peak_idx    : index of spike peak
    max_decay   : maximum days after peak to include
    min_pts     : minimum points for reliable fit

    Returns
    -------
    (k, sigma_0, sigma_inf) or None if fit fails
    """
    T = len(vol_series)
    end_idx = min(peak_idx + max_decay + 1, T)

    sigma_0   = vol_series[peak_idx]
    sigma_inf = vol_ambient[peak_idx]

    if np.isnan(sigma_0) or np.isnan(sigma_inf):
        return None
    if sigma_0 <= sigma_inf:
        return None    # no cooling to fit (vol already below ambient)

    # Build decay series: t=0,1,2,... and y = log(σ(t) - σ_inf)
    t_vals, y_vals = [], []
    for i, idx in enumerate(range(peak_idx, end_idx)):
        if np.isnan(vol_series[idx]) or np.isnan(vol_ambient[idx]):
            continue
        diff = vol_series[idx] - sigma_inf
        if diff <= 0:
            break    # vol has crossed below ambient — decay complete
        t_vals.append(float(i))
        y_vals.append(np.log(diff))

    if len(t_vals) < min_pts:
        return None

    t_arr = np.array(t_vals)
    y_arr = np.array(y_vals)

    # OLS: y = a - k*t
    # k = -slope, a = intercept
    t_mean, y_mean = t_arr.mean(), y_arr.mean()
    t_var = ((t_arr - t_mean)**2).sum()
    if t_var < 1e-10:
        return None

    k_neg = ((t_arr - t_mean) * (y_arr - y_mean)).sum() / t_var   # = -k
    k     = float(-k_neg)

    if k <= 0:
        return None    # cooling rate must be positive

    return k, float(sigma_0), float(sigma_inf)


# ── Cooling trajectory prediction ────────────────────────────────────────────

def _predict_cooling(
    k:         float,
    sigma_0:   float,
    sigma_inf: float,
    t:         float,
) -> float:
    """σ_predicted(t) = σ_inf + (σ_0 - σ_inf) * exp(-k * t)"""
    return sigma_inf + (sigma_0 - sigma_inf) * np.exp(-k * t)


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_nlc_scores(
    prices:    pd.DataFrame,
    macro_df:  pd.DataFrame,
    tickers:   List[str],
    window:    int,
) -> pd.Series:
    """
    Fit Newton's law of cooling to post-spike vol decay for each ETF.

    Parameters
    ----------
    prices   : DataFrame of closing prices, DatetimeIndex
    macro_df : DataFrame of macro signal levels
    tickers  : list of ETF tickers in this universe
    window   : lookback window in trading days

    Returns
    -------
    pd.Series indexed by ticker, values = composite NLC z-score
    """
    avail = [t for t in tickers if t in prices.columns]
    if not avail:
        return pd.Series(dtype=float)

    min_rows = window + config.VOL_LONG_WIN + 5
    if len(prices) < min_rows:
        return pd.Series(dtype=float)

    common   = prices.index.intersection(macro_df.index) if not macro_df.empty else prices.index
    prices_a = prices.loc[common]

    raw_scores = {}

    for ticker in avail:
        ps = prices_a[ticker].dropna()
        if len(ps) < min_rows:
            continue

        log_ret = np.log(ps / ps.shift(1)).dropna().values
        ret_win = log_ret[-window:]
        T       = len(ret_win)

        # ── Compute vol series ────────────────────────────────────────────────
        vol_short = _rolling_vol(ret_win, config.VOL_SHORT_WIN)
        vol_long  = _rolling_vol(ret_win, min(config.VOL_LONG_WIN, T//2))

        # ── Detect spikes in this window ──────────────────────────────────────
        spikes = _detect_spikes(vol_short, vol_long, config.SPIKE_THRESH_SIGMA)

        # ── Fit cooling law to each spike ─────────────────────────────────────
        k_vals      = []
        sigma_infs  = []
        overshoots  = []

        for peak_idx in spikes:
            result = _fit_cooling_law(
                vol_series  = vol_short,
                vol_ambient = vol_long,
                peak_idx    = peak_idx,
                max_decay   = config.MAX_DECAY_WIN,
                min_pts     = config.MIN_DECAY_POINTS,
            )
            if result is None:
                continue
            k, sigma_0, sigma_inf = result
            k_vals.append(k)
            sigma_infs.append(sigma_inf)

            # Overshoot: how far is current vol from where the cooling law
            # predicts it should be, given the most recent spike
            t_since_peak = T - 1 - peak_idx
            if t_since_peak >= 0:
                sigma_pred = _predict_cooling(k, sigma_0, sigma_inf,
                                              float(t_since_peak))
                sigma_now  = vol_short[-1]
                if not np.isnan(sigma_now) and sigma_pred > 1e-6:
                    overshoot = (sigma_now - sigma_pred) / sigma_pred
                    overshoots.append(float(overshoot))

        # ── Handle no-spike case ──────────────────────────────────────────────
        if not k_vals:
            # No spikes detected: use simple stats
            vol_now = vol_short[-1] if not np.isnan(vol_short[-1]) else 0.01
            vol_avg = np.nanmean(vol_short)
            if vol_avg < 1e-6:
                continue
            # Treat calm regime as positive (low ambient vol)
            raw_scores[ticker] = float(1.0 - vol_now / vol_avg)
            continue

        # ── Aggregate across spike episodes ──────────────────────────────────
        mean_k        = float(np.mean(k_vals))
        mean_sigma_inf = float(np.mean(sigma_infs))
        mean_overshoot = float(np.mean(overshoots)) if overshoots else 0.0

        print(f"    {ticker}: n_spikes={len(k_vals)}  "
              f"k={mean_k:.4f}  sigma_inf={mean_sigma_inf:.4f}  "
              f"overshoot={mean_overshoot:.4f}")

        # Score components
        s_k         = mean_k            # high k = fast decay = positive
        s_overshoot = mean_overshoot    # positive overshoot = vol above traj = buy
        s_ambient   = -mean_sigma_inf   # low ambient vol = positive (negated)

        composite = (
            config.WEIGHT_COOLING_RATE * s_k
            + config.WEIGHT_OVERSHOOT  * s_overshoot
            + config.WEIGHT_AMBIENT    * s_ambient
        )
        raw_scores[ticker] = composite

    if not raw_scores:
        return pd.Series(dtype=float)

    scores = pd.Series(raw_scores)
    mu, std = scores.mean(), scores.std()
    if std < 1e-10:
        return pd.Series(0.0, index=scores.index)
    return (scores - mu) / std
