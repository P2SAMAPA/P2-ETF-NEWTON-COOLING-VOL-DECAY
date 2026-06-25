# 🌡️ P2-ETF-NEWTON-COOLING-VOL-DECAY

**Newton's Law of Cooling Vol Decay Engine**

Part of the **P2Quant Engine Suite** · [P2SAMAPA](https://github.com/P2SAMAPA)

---

## What This Engine Does

This engine models post-spike volatility decay using **Newton's Law of Cooling**
— fitting an exponential decay trajectory to realised vol after each spike
episode. The fitted cooling rate k and the deviation of current vol from the
predicted trajectory give tradeable signals.

---

## Theory

### Newton's Law of Cooling (1701)

```
dσ/dt = -k * (σ(t) - σ_inf)
```

Solution:
```
σ(t) = σ_inf + (σ_0 - σ_inf) * exp(-k * t)
```

| Parameter | Meaning |
|-----------|---------|
| σ_0 | Vol at spike peak |
| σ_inf | Long-run ambient vol (63-day rolling) |
| k | Cooling rate constant (fitted per episode) |
| t | Days elapsed since spike peak |

### Spike Detection

A spike is detected when:
```
σ_short(t) > SPIKE_THRESH * σ_long(t)    (1.5x threshold)
```

Where σ_short = 5-day realised vol, σ_long = 63-day realised vol.

### Log-Linearised OLS Fit

```
log(σ(t) - σ_inf) = log(σ_0 - σ_inf) - k * t
```

Linear regression gives k (slope), fitted over up to 21 days post-peak.

### Score Construction

```
score = 0.40 * k  +  0.40 * overshoot  +  0.20 * (-σ_inf)
```

| Component | Meaning | Signal |
|-----------|---------|--------|
| k (cooling rate) | High k = vol decays fast | Positive |
| Overshoot | σ_today vs σ_predicted | Positive when vol above trajectory |
| -σ_inf | Low ambient vol | Positive |

### Distinction from HAR-RV

| Engine | Model | Structure |
|--------|-------|-----------|
| HAR-RV | σ(t) = a + b*σ(t-1) + c*σ(t-5) + d*σ(t-22) | Autoregressive lags |
| **NLC (this engine)** | σ(t) = σ_inf + (σ_0 - σ_inf)*exp(-kt) | Physical decay law |

HAR captures autocorrelation in vol levels. Newton cooling captures the
exponential shape of the post-spike decay trajectory. Orthogonal signals.

---

## Universes & Windows

| Universe | Tickers |
|---|---|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

**Windows:** `63d · 126d · 252d · 504d`

---

## Repository Structure

```
P2-ETF-NEWTON-COOLING-VOL-DECAY/
├── config.py          # Universes, spike threshold, decay params, weights
├── data_manager.py    # HuggingFace loader
├── nlc_engine.py      # Core: spike detection, OLS cooling fit, scoring
├── trainer.py         # Orchestrator
├── push_results.py    # HfApi.upload_file wrapper
├── streamlit_app.py   # Two-tab Streamlit dashboard
├── us_calendar.py     # US trading calendar helper
├── requirements.txt
└── .github/
    └── workflows/
        └── daily.yml  # Single job (pure numpy, very fast)
```

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-NEWTON-COOLING-VOL-DECAY
cd P2-ETF-NEWTON-COOLING-VOL-DECAY
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py
streamlit run streamlit_app.py
```

**Required GitHub secret:** `HF_TOKEN`

**Required HuggingFace dataset repo:** `P2SAMAPA/p2-etf-newton-cooling-results`

---

## References

- Newton, I. (1701). Scala graduum caloris. *Philosophical Transactions*, 22, 824–829.
- Cont, R. (2001). Empirical properties of asset returns. *Quantitative Finance*, 1(2), 223–236.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility.
  *Journal of Financial Econometrics*, 7(2), 174–196.
- Engle, R.F. (1982). Autoregressive conditional heteroscedasticity.
  *Econometrica*, 50(4), 987–1007.
