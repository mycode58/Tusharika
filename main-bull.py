"""
ETF Automated Trader — Webull Open API
=======================================
Total capital  : $1,000  split across up to 3 ETFs (~$333 each)
Data window    : 380 daily bars (~18 months); ETFs with < 130 bars
                 of valid data (~6 months) are SKIPPED automatically
Profit target  : +10 % per position → auto MARKET SELL
Stop-loss      : − 5 % per position → GTC STOP_LOSS placed at entry
Monitoring     : one background thread per position (parallel)

Strategy
────────
1. Auth     — Webull App token approval
2. Scan     — fetch 380 daily bars; skip any ETF with < 130 valid bars
3. Indicators — 10 technical indicators over the full 6-month+ window
4. Score    — weighted composite 0-100 (bullish = high)
5. Filter   — score ≥ 65, RSI ≤ 70, 10-day momentum > 0,
               6-month return > 0  (price trend confirmation)
6. Pick     — top 3 qualifying ETFs
7. Allocate — $1,000 / 3 ≈ $333 per ETF; size = floor(slice / ask)
8. Buy      — DAY LIMIT at ask
9. Guard    — GTC STOP_LOSS at entry × 0.95
10. Watch   — thread polls every 60 s; MARKET SELL at +10 %
11. Summary — P&L table printed when all threads finish

.env
────
APP_KEY=your_webull_app_key
APP_SECRET=your_webull_app_secret

Usage
─────
pip install pandas numpy python-dotenv colorama tabulate
python etf_trader.py

⚠  DISCLAIMER — executes REAL trades with REAL money.
   Test in paper-trading mode first. No liability accepted.
"""

from __future__ import annotations

import hashlib, hmac, base64, uuid, urllib.parse
import time, json, os, math, sys
import http.client
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd
import numpy as np
from colorama import Fore, Style, init as colorama_init
from tabulate import tabulate
from dotenv import load_dotenv

colorama_init(autoreset=True)
load_dotenv()

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

APP_KEY    = os.getenv("APP_KEY",    "YOUR_APP_KEY")
APP_SECRET = os.getenv("APP_SECRET", "YOUR_APP_SECRET")
HOST       = "api.webull.com"

TOTAL_BUDGET        = 1000.00   # total USD to deploy
MAX_POSITIONS       = 3         # split into this many ETFs (~$333 each)
PROFIT_TARGET_PCT   = 10.0      # auto-sell at +10 %
STOP_LOSS_PCT       = 5.0       # hard stop at -5 %
ENTRY_THRESHOLD     = 65.0      # min composite score (0-100)
RSI_OVERBOUGHT      = 70.0      # skip if RSI > this
POLL_INTERVAL       = 60        # seconds between position checks

# ── Data window ──────────────────────────────────────────────
BARS_FETCH          = 380       # bars to request  (~18 months of trading days)
MIN_VALID_BARS      = 130       # minimum bars required (~6 months); ETF skipped if fewer
SMA_LONG_PERIOD     = 120       # long SMA uses 6-month window  (replaces SMA-200)
SMA_SHORT_PERIOD    = 50        # short SMA
MOM_PERIOD          = 126       # 6-month momentum  (~126 trading days)
TIMESPAN            = "D"
CATEGORY            = "US_ETF"

WATCHLIST = [
    "SPY","QQQ","IWM","DIA","VTI",
    "XLK","XLF","XLE","XLV","XLI","XLP","XLY","XLB",
    "GLD","SLV","GDX",
    "TLT","HYG","LQD",
    "SOXX","SMH","IBB","ARKK",
    "EFA","EEM","VWO","FXI","EWJ",
    "MTUM","QUAL","USMV","VIG","DVY",
]

# ── Runtime state ─────────────────────────────────────────────
_ACTIVE_TOKEN: str = ""
_ACCOUNT_ID:   str = ""
_print_lock            = threading.Lock()   # keep threaded prints tidy
_results_lock          = threading.Lock()   # protect shared results list


# ═══════════════════════════════════════════════════════════════
#  SECTION 1 — HTTP / signing
# ═══════════════════════════════════════════════════════════════

def _sign(path: str, query_params: dict, body_str: Optional[str],
          timestamp: str, nonce: str) -> str:
    signing_headers = {
        "x-app-key":             APP_KEY,
        "x-timestamp":           timestamp,
        "x-signature-algorithm": "HMAC-SHA1",
        "x-signature-version":   "1.0",
        "x-signature-nonce":     nonce,
        "host":                  HOST,
    }
    merged = {**query_params, **signing_headers}
    str1   = "&".join(f"{k}={merged[k]}" for k in sorted(merged))
    if body_str:
        str2 = hashlib.md5(body_str.encode()).hexdigest().upper()
        str3 = f"{path}&{str1}&{str2}"
    else:
        str3 = f"{path}&{str1}"
    encoded = urllib.parse.quote(str3, safe="")
    key     = f"{APP_SECRET}&"
    return base64.b64encode(
        hmac.new(key.encode(), encoded.encode(), hashlib.sha1).digest()
    ).decode()


def _build_headers(path: str, qp: dict, body_str: Optional[str]) -> dict:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    nc = uuid.uuid4().hex
    h  = {
        "Accept":                  "application/json",
        "Content-Type":            "application/json",
        "x-app-key":               APP_KEY,
        "x-timestamp":             ts,
        "x-signature":             _sign(path, qp, body_str, ts, nc),
        "x-signature-algorithm":   "HMAC-SHA1",
        "x-signature-version":     "1.0",
        "x-signature-nonce":       nc,
        "x-version":               "v2",
    }
    if _ACTIVE_TOKEN:
        h["x-auth-token"] = _ACTIVE_TOKEN
    return h


def _raw_http(method: str, path: str, body_str: Optional[str],
              qp: dict, include_token: bool) -> Tuple[int, str]:
    hdrs = _build_headers(path, qp, body_str)
    if not include_token:
        hdrs.pop("x-auth-token", None)
    qs  = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in qp.items())
    url = f"{path}?{qs}" if qs else path
    try:
        conn = http.client.HTTPSConnection(HOST, timeout=20)
        conn.request(method, url, body_str, hdrs)
        res = conn.getresponse()
        return res.status, res.read().decode()
    except Exception as exc:
        return 0, str(exc)


def _get(path: str, params: Optional[dict] = None) -> Optional[Any]:
    qp = params or {}
    status, raw = _raw_http("GET", path, None, qp, True)
    if status == 200 and raw:
        return json.loads(raw)
    if status:
        with _print_lock:
            print(f"  [WARN] GET {path} → {status}: {raw[:200]}")
    return None


def _post(path: str, body: Optional[dict] = None,
          include_token: bool = True) -> Optional[Any]:
    body_str = json.dumps(body) if body else None
    status, raw = _raw_http("POST", path, body_str, {}, include_token)
    if status == 200 and raw:
        return json.loads(raw)
    if status:
        with _print_lock:
            print(f"  [WARN] POST {path} → {status}: {raw[:200]}")
    return None


def _delete(path: str, params: Optional[dict] = None) -> Optional[Any]:
    qp = params or {}
    status, raw = _raw_http("DELETE", path, None, qp, True)
    if status == 200 and raw:
        return json.loads(raw)
    return None


# ═══════════════════════════════════════════════════════════════
#  SECTION 2 — Authentication
# ═══════════════════════════════════════════════════════════════

def acquire_token(poll_timeout: int = 300, poll_interval: int = 5) -> str:
    global _ACTIVE_TOKEN
    print(f"\n{'─'*60}")
    print("  WEBULL AUTH")
    print(f"{'─'*60}")
    data = _post("/openapi/auth/token/create", include_token=False)
    if not data:
        print("❌  Failed to create token. Check APP_KEY / APP_SECRET.")
        sys.exit(1)
    token = data.get("token")
    print(f"  → Token : {token}")
    print(f"  → Status: {data.get('status')}")
    print(f"\n  ⚠️  Open the Webull App and APPROVE the verification request.")
    print(f"  Polling every {poll_interval}s …\n")
    elapsed = 0
    while elapsed < poll_timeout:
        time.sleep(poll_interval)
        elapsed += poll_interval
        chk  = _post("/openapi/auth/token/check",
                     body={"token": token}, include_token=False)
        status = chk.get("status", "UNKNOWN") if chk else "ERROR"
        filled = elapsed // poll_interval
        bar    = "▓" * filled + "░" * max(0, poll_timeout // poll_interval - filled)
        print(f"  [{elapsed:>4}s] {status}  {bar[:30]}")
        if status == "NORMAL":
            print(f"\n  {Fore.GREEN}✅  Token ACTIVE{Style.RESET_ALL}")
            _ACTIVE_TOKEN = token
            return token
        if status in ("INVALID", "EXPIRED"):
            print("  Token expired — restarting auth …")
            return acquire_token(poll_timeout, poll_interval)
    print("❌  Auth timed out.")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
#  SECTION 3 — Account helpers
# ═══════════════════════════════════════════════════════════════

def get_account_id() -> str:
    global _ACCOUNT_ID
    data = _get("/openapi/account/list")
    if not data:
        print("❌  Could not fetch account list.")
        sys.exit(1)
    accounts = data if isinstance(data, list) else data.get("data", [])
    if not accounts:
        print("❌  No accounts found.")
        sys.exit(1)
    _ACCOUNT_ID = (accounts[0].get("account_id")
                   or accounts[0].get("accountId", ""))
    print(f"  Account ID   : {_ACCOUNT_ID}")
    return _ACCOUNT_ID


def get_balance() -> float:
    data = _get(f"/openapi/account/{_ACCOUNT_ID}/balance")
    if not data:
        return 0.0
    for key in ("buyingPower", "buying_power", "cashBalance", "cash_balance",
                "netLiquidation", "net_liquidation"):
        if key in data:
            return float(data[key])
    bal = data.get("data", data)
    if isinstance(bal, dict):
        for key in ("buyingPower", "buying_power", "cashBalance", "cash_balance"):
            if key in bal:
                return float(bal[key])
    print(f"  [WARN] Cannot parse balance: {str(data)[:200]}")
    return 0.0


def get_positions() -> List[dict]:
    data = _get(f"/openapi/account/{_ACCOUNT_ID}/positions")
    if not data:
        return []
    return data if isinstance(data, list) else data.get("data", [])


def get_ask(symbol: str) -> Optional[float]:
    data = _get("/market-data/snapshot",
                params={"symbol": symbol, "category": CATEGORY})
    if not data:
        return None
    snaps = data if isinstance(data, list) else data.get("data", [data])
    snap  = snaps[0] if snaps else {}
    for key in ("askPrice", "ask_price", "ask", "lastPrice", "last_price", "close"):
        v = snap.get(key)
        if v is not None:
            return float(v)
    return None


# ═══════════════════════════════════════════════════════════════
#  SECTION 4 — Market data  (6-month+ window)
# ═══════════════════════════════════════════════════════════════

def fetch_bars(symbol: str) -> Optional[pd.DataFrame]:
    """
    Fetch BARS_FETCH (~18 months) daily bars.
    Returns None if fewer than MIN_VALID_BARS (~6 months) come back.
    """
    data = _get(
        "/market-data/bars",
        params={"symbol":   symbol,
                "category": CATEGORY,
                "timespan": TIMESPAN,
                "count":    BARS_FETCH},
    )
    if not data:
        return None
    bars = data if isinstance(data, list) else data.get("data", [])
    if not bars:
        return None

    df = pd.DataFrame(bars)
    col_map = {
        "openPrice":  "open",   "open":   "open",
        "highPrice":  "high",   "high":   "high",
        "lowPrice":   "low",    "low":    "low",
        "closePrice": "close",  "close":  "close",
        "tradeVolume":"volume", "volume": "volume",
        "timestamp":  "ts",     "time":   "ts",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns},
              inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    if "ts" in df.columns:
        df.sort_values("ts", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── 6-month data gate ────────────────────────────────────
    if len(df) < MIN_VALID_BARS:
        return None          # not enough history — skip this ETF

    return df


# ═══════════════════════════════════════════════════════════════
#  SECTION 5 — Technical indicators  (10 total, 6-month+ window)
# ═══════════════════════════════════════════════════════════════

def _nan(v: float) -> bool:
    return math.isnan(v)


# 1. RSI-14
def calc_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    val   = 100 - 100 / (1 + rs)
    v = val.iloc[-1]
    return float(v) if not np.isnan(v) else float("nan")


# 2 & 3. MACD line + histogram
def calc_macd(close: pd.Series,
              fast=12, slow=26, sig=9) -> Tuple[float, float, float]:
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    line   = ef - es
    signal = line.ewm(span=sig, adjust=False).mean()
    hist   = line - signal
    return float(line.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])


# 4. Bollinger %B  (20-day)
def calc_bb_pctb(close: pd.Series, period=20, nstd=2) -> float:
    sma   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = sma + nstd * std
    lower = sma - nstd * std
    pctb  = (close - lower) / (upper - lower)
    v = pctb.iloc[-1]
    return float(v) if not np.isnan(v) else float("nan")


# 5. ADX-14
def calc_adx(high: pd.Series, low: pd.Series,
             close: pd.Series, period=14) -> float:
    tr   = pd.concat([high - low,
                      (high - close.shift()).abs(),
                      (low  - close.shift()).abs()], axis=1).max(axis=1)
    dmp  = high.diff().clip(lower=0)
    dmm  = (-low.diff()).clip(lower=0)
    dmp[dmp < dmm] = 0
    dmm[dmm < high.diff().clip(lower=0)] = 0
    atr  = tr.ewm(span=period, adjust=False).mean()
    dip  = 100 * dmp.ewm(span=period, adjust=False).mean() / atr
    dim  = 100 * dmm.ewm(span=period, adjust=False).mean() / atr
    dx   = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
    adx  = dx.ewm(span=period, adjust=False).mean()
    v = adx.iloc[-1]
    return float(v) if not np.isnan(v) else float("nan")


# 6. SMA cross score  — uses SMA_SHORT_PERIOD (50) vs SMA_LONG_PERIOD (120 ≈ 6 months)
def calc_sma_score(close: pd.Series) -> float:
    if len(close) < SMA_LONG_PERIOD:
        return float("nan")
    s_short = close.rolling(SMA_SHORT_PERIOD).mean().iloc[-1]
    s_long  = close.rolling(SMA_LONG_PERIOD).mean().iloc[-1]
    pct     = (s_short - s_long) / s_long * 100
    return float(np.clip(50 + pct * 5, 0, 100))


# 7. OBV trend score (20-day slope)
def calc_obv_score(close: pd.Series,
                   volume: pd.Series, lookback=20) -> float:
    direction = np.sign(close.diff()).fillna(0)
    obv  = (direction * volume).cumsum()
    tail = obv.iloc[-lookback:]
    if len(tail) < 5:
        return float("nan")
    slope = np.polyfit(np.arange(len(tail)), tail.values, 1)[0]
    mean  = tail.abs().mean() + 1e-9
    return float(np.clip(50 + slope / mean * 500, 0, 100))


# 8. ATR % (14-day)
def calc_atr_pct(high: pd.Series, low: pd.Series,
                 close: pd.Series, period=14) -> float:
    tr  = pd.concat([high - low,
                     (high - close.shift()).abs(),
                     (low  - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    v   = atr / close.iloc[-1] * 100
    return float(v) if not np.isnan(v) else float("nan")


# 9. 10-day short-term momentum %
def calc_momentum_short(close: pd.Series, period=10) -> float:
    if len(close) < period + 1:
        return float("nan")
    return float((close.iloc[-1] / close.iloc[-(period + 1)] - 1) * 100)


# 10. 6-month momentum %  (MOM_PERIOD ≈ 126 trading days)
def calc_momentum_6m(close: pd.Series) -> float:
    if len(close) < MOM_PERIOD + 1:
        return float("nan")
    return float((close.iloc[-1] / close.iloc[-(MOM_PERIOD + 1)] - 1) * 100)


# 11. Stochastic %K-14
def calc_stoch(high: pd.Series, low: pd.Series,
               close: pd.Series, k=14) -> float:
    lo  = low.rolling(k).min()
    hi  = high.rolling(k).max()
    stk = 100 * (close - lo) / (hi - lo).replace(0, np.nan)
    v   = stk.iloc[-1]
    return float(v) if not np.isnan(v) else float("nan")


# ═══════════════════════════════════════════════════════════════
#  SECTION 6 — Scoring  (each → 0-100, higher = more bullish)
# ═══════════════════════════════════════════════════════════════

def score_rsi(v: float) -> float:
    if _nan(v): return 50.0
    if v <= 30: return 90.0
    if v <= 45: return 68.0
    if v <= 55: return 50.0
    if v <= 70: return 32.0
    return 10.0

def score_macd_hist(hist: float, price: float) -> float:
    if _nan(hist) or price == 0: return 50.0
    return float(np.clip(50 + (hist / price * 100) * 25, 0, 100))

def score_macd_line(line: float, signal: float) -> float:
    if _nan(line) or _nan(signal): return 50.0
    diff = line - signal
    return min(90.0, 55 + abs(diff) * 200) if diff > 0 else max(10.0, 45 - abs(diff) * 200)

def score_bb(v: float) -> float:
    if _nan(v): return 50.0
    if v < 0: return 92.0
    if v > 1: return  8.0
    return float(100 - v * 80)

def score_adx(v: float) -> float:
    return 50.0 if _nan(v) else float(np.clip(v * 2, 0, 100))

def score_sma(v: float) -> float:
    return 50.0 if _nan(v) else v

def score_obv(v: float) -> float:
    return 50.0 if _nan(v) else v

def score_atr(v: float) -> float:
    return 50.0 if _nan(v) else float(np.clip(100 - v * 10, 0, 100))

def score_mom_short(v: float) -> float:
    return 50.0 if _nan(v) else float(np.clip(50 + v * 3, 0, 100))

def score_mom_6m(v: float) -> float:
    """6-month return: positive = bullish trend over entire data window."""
    if _nan(v): return 50.0
    return float(np.clip(50 + v * 1.5, 0, 100))

def score_stoch(v: float) -> float:
    if _nan(v): return 50.0
    if v <= 20: return 88.0
    if v >= 80: return 12.0
    return float(100 - v)


# Weights — 11 sub-scores across 10 indicators, sum = 1.0
WEIGHTS = {
    "RSI":        0.15,
    "MACD_HIST":  0.13,
    "MACD_LINE":  0.09,
    "SMA":        0.14,   # now SMA-50 vs SMA-120 (6-month)
    "ADX":        0.11,
    "BB":         0.09,
    "OBV":        0.08,
    "MOM_SHORT":  0.06,
    "MOM_6M":     0.10,   # 6-month price return — new, high weight
    "STOCH":      0.03,
    "ATR":        0.02,
}


def composite(scores: Dict[str, float]) -> float:
    total = w_sum = 0.0
    for k, w in WEIGHTS.items():
        v = scores.get(k, float("nan"))
        if not math.isnan(v):
            total += w * v
            w_sum += w
    return round(total / w_sum if w_sum else 50.0, 2)


def signal_label(score: float) -> str:
    if score >= 75: return Fore.GREEN  + Style.BRIGHT + "STRONG BUY "
    if score >= 65: return Fore.GREEN                 + "BUY        "
    if score >= 50: return Fore.YELLOW                + "NEUTRAL    "
    if score >= 35: return Fore.RED                   + "SELL       "
    return                  Fore.RED   + Style.BRIGHT + "STRONG SELL"


# ═══════════════════════════════════════════════════════════════
#  SECTION 7 — ETF analysis
# ═══════════════════════════════════════════════════════════════

def analyse(symbol: str) -> Optional[Dict[str, Any]]:
    df = fetch_bars(symbol)
    if df is None:
        return None          # either API failed OR < 6 months of data

    c = df["close"]
    h = df["high"]   if "high"   in df.columns else c
    l = df["low"]    if "low"    in df.columns else c
    v = df["volume"] if "volume" in df.columns else pd.Series(
        np.zeros(len(c)), index=c.index)

    rsi_v               = calc_rsi(c)
    macd_l, macd_s, mh  = calc_macd(c)
    bb_v                = calc_bb_pctb(c)
    adx_v               = calc_adx(h, l, c)
    sma_v               = calc_sma_score(c)
    obv_v               = calc_obv_score(c, v)
    atr_v               = calc_atr_pct(h, l, c)
    mom_s               = calc_momentum_short(c)
    mom_6m              = calc_momentum_6m(c)
    stoch_v             = calc_stoch(h, l, c)
    price               = float(c.iloc[-1])
    bars_used           = len(df)

    scores = {
        "RSI":       score_rsi(rsi_v),
        "MACD_HIST": score_macd_hist(mh, price),
        "MACD_LINE": score_macd_line(macd_l, macd_s),
        "BB":        score_bb(bb_v),
        "ADX":       score_adx(adx_v),
        "SMA":       score_sma(sma_v),
        "OBV":       score_obv(obv_v),
        "ATR":       score_atr(atr_v),
        "MOM_SHORT": score_mom_short(mom_s),
        "MOM_6M":    score_mom_6m(mom_6m),
        "STOCH":     score_stoch(stoch_v),
    }

    return {
        "Symbol":   symbol,
        "Bars":     bars_used,
        "Price":    round(price, 2),
        "RSI":      round(rsi_v,  1) if not _nan(rsi_v)  else None,
        "MACD_H":   round(mh,     4) if not _nan(mh)     else None,
        "BB_%B":    round(bb_v,   3) if not _nan(bb_v)   else None,
        "ADX":      round(adx_v,  1) if not _nan(adx_v)  else None,
        "SMA_Sc":   round(sma_v,  1) if not _nan(sma_v)  else None,
        "ATR%":     round(atr_v,  2) if not _nan(atr_v)  else None,
        "MOM%":     round(mom_s,  2) if not _nan(mom_s)  else None,
        "MOM_6M%":  round(mom_6m, 2) if not _nan(mom_6m) else None,
        "STOCH":    round(stoch_v,1) if not _nan(stoch_v)else None,
        "Score":    composite(scores),
        "_rsi":     rsi_v,
        "_mom_s":   mom_s,
        "_mom_6m":  mom_6m,
        "_scores":  scores,
    }


# ═══════════════════════════════════════════════════════════════
#  SECTION 8 — Order helpers
# ═══════════════════════════════════════════════════════════════

def _oid() -> str:
    return uuid.uuid4().hex[:32]


def place_buy_limit(symbol: str, qty: int, limit_price: float) -> Optional[str]:
    cid  = _oid()
    body = {
        "account_id": _ACCOUNT_ID,
        "new_orders": [{
            "client_order_id":         cid,
            "combo_type":              "NORMAL",
            "symbol":                  symbol,
            "instrument_type":         "EQUITY",
            "market":                  "US",
            "order_type":              "LIMIT",
            "limit_price":             f"{limit_price:.4f}",
            "quantity":                str(qty),
            "side":                    "BUY",
            "time_in_force":           "DAY",
            "support_trading_session": "CORE",
            "entrust_type":            "QTY",
        }],
    }
    resp = _post("/openapi/trade/order/place", body=body)
    if resp:
        with _print_lock:
            print(f"  {Fore.GREEN}✅  BUY  {qty}×{symbol} @ ${limit_price:.2f}{Style.RESET_ALL}")
        return cid
    return None


def place_stop_loss(symbol: str, qty: int, stop_price: float) -> Optional[str]:
    cid  = _oid()
    body = {
        "account_id": _ACCOUNT_ID,
        "new_orders": [{
            "client_order_id":         cid,
            "combo_type":              "NORMAL",
            "symbol":                  symbol,
            "instrument_type":         "EQUITY",
            "market":                  "US",
            "order_type":              "STOP_LOSS",
            "stop_price":              f"{stop_price:.4f}",
            "quantity":                str(qty),
            "side":                    "SELL",
            "time_in_force":           "GTC",
            "support_trading_session": "CORE",
            "entrust_type":            "QTY",
        }],
    }
    resp = _post("/openapi/trade/order/place", body=body)
    if resp:
        with _print_lock:
            print(f"  {Fore.YELLOW}🛡  STOP  {symbol} @ ${stop_price:.2f}{Style.RESET_ALL}")
        return cid
    return None


def place_market_sell(symbol: str, qty: int) -> Optional[str]:
    cid  = _oid()
    body = {
        "account_id": _ACCOUNT_ID,
        "new_orders": [{
            "client_order_id":         cid,
            "combo_type":              "NORMAL",
            "symbol":                  symbol,
            "instrument_type":         "EQUITY",
            "market":                  "US",
            "order_type":              "MARKET",
            "quantity":                str(qty),
            "side":                    "SELL",
            "time_in_force":           "DAY",
            "support_trading_session": "CORE",
            "entrust_type":            "QTY",
        }],
    }
    resp = _post("/openapi/trade/order/place", body=body)
    if resp:
        with _print_lock:
            print(f"  {Fore.CYAN}💰  SELL {qty}×{symbol} MARKET{Style.RESET_ALL}")
        return cid
    return None


def cancel_order(cid: str) -> bool:
    return _delete(f"/openapi/trade/order/{cid}",
                   params={"account_id": _ACCOUNT_ID}) is not None


def order_status(cid: str) -> str:
    data = _get(f"/openapi/trade/order/{cid}",
                params={"account_id": _ACCOUNT_ID})
    if not data:
        return "UNKNOWN"
    o = data if isinstance(data, dict) else (data.get("data") or [{}])[0]
    return o.get("status", o.get("orderStatus", "UNKNOWN"))


# ═══════════════════════════════════════════════════════════════
#  SECTION 9 — Position helpers
# ═══════════════════════════════════════════════════════════════

def find_position(symbol: str) -> Optional[dict]:
    for pos in get_positions():
        sym = pos.get("symbol") or pos.get("ticker", {}).get("symbol", "")
        if sym == symbol:
            return pos
    return None


def current_price(symbol: str, pos: dict) -> float:
    ask = get_ask(symbol)
    if ask:
        return ask
    for key in ("lastPrice", "last_price"):
        v = pos.get(key)
        if v is not None:
            return float(v)
    mval = pos.get("marketValue") or pos.get("market_value")
    if mval:
        qty = int(pos.get("quantity") or pos.get("qty") or 1)
        return float(mval) / max(qty, 1)
    return 0.0


# ═══════════════════════════════════════════════════════════════
#  SECTION 10 — Per-position monitor thread
# ═══════════════════════════════════════════════════════════════

def monitor_position(symbol: str, qty: int, entry: float,
                     stop_cid: str, results: list) -> None:
    """
    Runs in its own thread.
    Polls every POLL_INTERVAL seconds.
    Exits when:
      • P&L ≥ +PROFIT_TARGET_PCT  → cancel stop, MARKET SELL
      • Stop-loss already filled  → log it
    Appends outcome dict to shared `results` list.
    """
    target = entry * (1 + PROFIT_TARGET_PCT / 100)
    stop_p = entry * (1 - STOP_LOSS_PCT      / 100)

    with _print_lock:
        print(f"\n  [{symbol}]  entry=${entry:.2f}  "
              f"target=${target:.2f} (+{PROFIT_TARGET_PCT:.0f}%)  "
              f"stop=${stop_p:.2f} (-{STOP_LOSS_PCT:.0f}%)")

    outcome = {"symbol": symbol, "qty": qty, "entry": entry,
               "exit": None, "pnl_pct": None, "reason": "running"}

    while True:
        time.sleep(POLL_INTERVAL)

        # ── Stop-loss already triggered by Webull? ────────────
        if stop_cid != "__none__":
            sl_st = order_status(stop_cid)
            if sl_st in ("FILLED", "PARTIAL_FILLED"):
                with _print_lock:
                    print(f"  {Fore.RED}⛔ [{symbol}]  Stop-loss hit ({sl_st}){Style.RESET_ALL}")
                outcome.update(reason="stop_loss_hit")
                break

        # ── Check live price ──────────────────────────────────
        pos = find_position(symbol)
        if pos is None:
            with _print_lock:
                print(f"  ⚠️  [{symbol}] position gone — may have closed via stop.")
            outcome.update(reason="position_not_found")
            break

        cur  = current_price(symbol, pos)
        if cur <= 0:
            continue

        cur_qty = int(pos.get("quantity") or pos.get("qty") or qty)
        pnl     = (cur - entry) / entry * 100
        col     = Fore.GREEN if pnl >= 0 else Fore.RED

        with _print_lock:
            print(f"  {datetime.now():%H:%M:%S}  [{symbol}]  "
                  f"${cur:.2f}  {col}{pnl:+.2f}%{Style.RESET_ALL}  qty={cur_qty}")

        # ── Profit target reached ─────────────────────────────
        if pnl >= PROFIT_TARGET_PCT:
            with _print_lock:
                print(f"\n  {Fore.GREEN+Style.BRIGHT}"
                      f"🎯  [{symbol}]  +{PROFIT_TARGET_PCT:.0f}% target hit!{Style.RESET_ALL}")
            if stop_cid != "__none__" and cancel_order(stop_cid):
                with _print_lock:
                    print(f"  🗑  [{symbol}]  stop-loss order cancelled.")
            sell_cid = place_market_sell(symbol, cur_qty)
            exit_price = cur
            if sell_cid:
                for _ in range(12):
                    time.sleep(5)
                    st = order_status(sell_cid)
                    with _print_lock:
                        print(f"     [{symbol}] sell status: {st}")
                    if st in ("FILLED", "PARTIAL_FILLED"):
                        break
            gross = (exit_price - entry) * cur_qty
            with _print_lock:
                print(f"  {Fore.GREEN}💵  [{symbol}]  gross P&L ≈ ${gross:.2f}{Style.RESET_ALL}")
            outcome.update(exit=exit_price, pnl_pct=pnl, reason="profit_target")
            break

    with _results_lock:
        results.append(outcome)


# ═══════════════════════════════════════════════════════════════
#  SECTION 11 — Main orchestration
# ═══════════════════════════════════════════════════════════════

def main() -> None:

    # ── 1. Auth ───────────────────────────────────────────────
    acquire_token()

    # ── 2. Account ───────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  ACCOUNT")
    print(f"{'─'*65}")
    get_account_id()
    buying_power = get_balance()
    print(f"  Buying power : ${buying_power:,.2f}")

    budget = min(TOTAL_BUDGET, buying_power)
    if budget < 5:
        print("❌  Insufficient buying power (need ≥ $5).")
        sys.exit(1)

    per_pos = budget / MAX_POSITIONS
    print(f"  Total budget : ${budget:.2f}")
    print(f"  Positions    : {MAX_POSITIONS}  (~${per_pos:.0f} each)")
    print(f"  Data window  : {BARS_FETCH} bars fetched  |  min {MIN_VALID_BARS} required (~6 months)")

    # ── 3. Scan ───────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  ETF SCAN  |  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Scoring {len(WATCHLIST)} ETFs  (skipping any with < {MIN_VALID_BARS} days of history) …")
    print(f"{'═'*72}\n")

    results_raw  = []
    skipped_data = []

    for i, sym in enumerate(WATCHLIST, 1):
        print(f"  [{i:>2}/{len(WATCHLIST)}] {sym:<6}", end=" ", flush=True)
        rec = analyse(sym)
        if rec:
            results_raw.append(rec)
            sc  = rec["Score"]
            bar = "█" * int(sc / 10) + "░" * (10 - int(sc / 10))
            col = Fore.GREEN if sc >= 65 else (Fore.YELLOW if sc >= 50 else Fore.RED)
            print(f"{col}{bar} {sc:.1f}{Style.RESET_ALL}  "
                  f"[{rec['Bars']}d / 6m-ret={rec['MOM_6M%']:+.1f}%]"
                  if rec["MOM_6M%"] is not None else
                  f"{col}{bar} {sc:.1f}{Style.RESET_ALL}  [{rec['Bars']}d]")
        else:
            skipped_data.append(sym)
            print(Fore.RED + f"SKIPPED  (< {MIN_VALID_BARS} bars or API fail)")
        time.sleep(0.15)

    if skipped_data:
        print(f"\n  ⚠️  Skipped (insufficient 6-month history): {', '.join(skipped_data)}")

    if not results_raw:
        print("❌  No ETFs with enough history. Exiting.")
        return

    # ── 4. Rank & display ────────────────────────────────────
    df = (pd.DataFrame(results_raw)
            .sort_values("Score", ascending=False)
            .reset_index(drop=True))
    df.index += 1

    print(f"\n\n{'═'*120}")
    print("  RANKINGS  — 11 sub-scores · 6-month+ data window")
    print("  Weights: RSI 15% | MACD 22% | SMA-50/120 14% | ADX 11% | BB 9% | OBV 8% | 6M-MOM 10% | 10d-MOM 6% | STOCH 3% | ATR 2%")
    print(f"{'═'*120}\n")

    rows = []
    for rank, row in df.iterrows():
        def fmt(v, f):
            return f.format(v) if v is not None else "—"

        entry_ok = (
            row["Score"]   >= ENTRY_THRESHOLD
            and (row["RSI"]    is None or row["RSI"]    <= RSI_OVERBOUGHT)
            and (row["MOM%"]   is None or row["MOM%"]   > 0)
            and (row["MOM_6M%"]is None or row["MOM_6M%"]> 0)
        )
        flag = f"{Fore.GREEN}★ ENTER{Style.RESET_ALL}" if entry_ok else ""

        rows.append([
            rank,
            row["Symbol"],
            row["Bars"],
            fmt(row["Price"],   "${:>7.2f}"),
            fmt(row["RSI"],     "{:>5.1f}"),
            fmt(row["MACD_H"],  "{:>+8.4f}"),
            fmt(row["BB_%B"],   "{:>6.3f}"),
            fmt(row["ADX"],     "{:>5.1f}"),
            fmt(row["SMA_Sc"],  "{:>5.1f}"),
            fmt(row["MOM%"],    "{:>+5.2f}%"),
            fmt(row["MOM_6M%"], "{:>+6.1f}%"),
            fmt(row["STOCH"],   "{:>5.1f}"),
            f"{row['Score']:>6.2f}",
            signal_label(row["Score"]) + Style.RESET_ALL,
            flag,
        ])

    headers = ["#","ETF","Bars","Price","RSI","MACD_H","BB%B","ADX",
               "SMA","10dMOM","6mMOM","STOCH","Score","Signal","Action"]
    print(tabulate(rows, headers=headers, tablefmt="simple"))

    # ── 5. Filter candidates ─────────────────────────────────
    cand = df[
        (df["Score"]   >= ENTRY_THRESHOLD) &
        (df["RSI"].isna()     | (df["RSI"]     <= RSI_OVERBOUGHT)) &
        (df["MOM%"].isna()    | (df["MOM%"]    > 0)) &
        (df["MOM_6M%"].isna() | (df["MOM_6M%"] > 0))
    ].head(MAX_POSITIONS)

    if cand.empty:
        print(f"\n  ⚠️  No ETF passes all entry filters today. No trades placed.")
        return

    n_trades = len(cand)
    per_pos  = budget / n_trades          # re-compute for actual trade count
    print(f"\n  {Fore.GREEN+Style.BRIGHT}★  {n_trades} ETF(s) selected "
          f"(${per_pos:.0f} each){Style.RESET_ALL}")
    for _, r in cand.iterrows():
        print(f"     {r['Symbol']:<6}  score={r['Score']:.1f}  "
              f"6m-return={r['MOM_6M%']:+.1f}%"
              if r["MOM_6M%"] is not None else
              f"     {r['Symbol']:<6}  score={r['Score']:.1f}")

    # ── 6. Confirm ────────────────────────────────────────────
    print(f"\n  {Fore.YELLOW}⚠️  This will place REAL orders totalling "
          f"~${n_trades*per_pos:.0f} on your Webull account.{Style.RESET_ALL}")
    confirm = input("  Type 'yes' to proceed, anything else to abort: ").strip().lower()
    if confirm != "yes":
        print("  Aborted.")
        return

    # ── 7. Buy each ETF ──────────────────────────────────────
    positions_opened: List[dict] = []

    for _, row in cand.iterrows():
        sym = row["Symbol"]
        print(f"\n  ── {sym} ──")

        ask = get_ask(sym)
        if ask is None or ask <= 0:
            ask = row["Price"]
            print(f"  [WARN] No live ask; using last close ${ask:.2f}")

        shares = int(per_pos // ask)
        if shares < 1:
            print(f"  ❌  Slice ${per_pos:.0f} < ask ${ask:.2f}. Skipping {sym}.")
            continue

        cost = shares * ask
        print(f"  Sizing: {shares} share(s) × ${ask:.2f} = ${cost:.2f}")

        buy_cid = place_buy_limit(sym, shares, ask)
        if not buy_cid:
            print(f"  ❌  BUY order failed for {sym}. Skipping.")
            continue

        # Wait up to 2 min for fill
        print(f"  Waiting for fill …")
        entry_price = ask
        filled = False
        for _ in range(24):
            time.sleep(5)
            st = order_status(buy_cid)
            print(f"     {sym} order: {st}")
            if st == "FILLED":
                filled = True
                break
            if st in ("CANCELLED", "FAILED"):
                print(f"  ❌  Order {st}. Skipping {sym}.")
                break
        if not filled:
            print(f"  ⚠️  {sym} not confirmed filled within 2 min; continuing anyway.")

        # Place stop-loss guard
        stop_px  = round(entry_price * (1 - STOP_LOSS_PCT / 100), 4)
        stop_cid = place_stop_loss(sym, shares, stop_px) or "__none__"

        positions_opened.append({
            "symbol":    sym,
            "shares":    shares,
            "entry":     entry_price,
            "stop_cid":  stop_cid,
        })
        time.sleep(0.5)   # brief pause between orders

    if not positions_opened:
        print("\n❌  No positions opened. Exiting.")
        return

    # ── 8. Monitor all positions in parallel threads ──────────
    thread_results: List[dict] = []
    threads: List[threading.Thread] = []

    print(f"\n{'═'*65}")
    print(f"  MONITORING {len(positions_opened)} POSITION(S) IN PARALLEL")
    print(f"  Poll every {POLL_INTERVAL}s  |  "
          f"Exit at +{PROFIT_TARGET_PCT:.0f}%  |  "
          f"Stop at -{STOP_LOSS_PCT:.0f}%")
    print(f"{'═'*65}")

    for p in positions_opened:
        t = threading.Thread(
            target=monitor_position,
            args=(p["symbol"], p["shares"], p["entry"],
                  p["stop_cid"], thread_results),
            daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()          # wait for every thread to finish

    # ── 9. Final summary ─────────────────────────────────────
    print(f"\n\n{'═'*65}")
    print(f"  SESSION SUMMARY  |  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'═'*65}\n")

    summary_rows = []
    total_pnl = 0.0
    for r in thread_results:
        pnl_usd = ((r["exit"] or r["entry"]) - r["entry"]) * r["qty"] \
                  if r["exit"] else None
        pnl_pct = r["pnl_pct"]
        if pnl_usd:
            total_pnl += pnl_usd
        col = Fore.GREEN if (pnl_pct or 0) >= 0 else Fore.RED
        summary_rows.append([
            r["symbol"],
            r["qty"],
            f"${r['entry']:.2f}",
            f"${r['exit']:.2f}" if r["exit"] else "—",
            f"{col}{pnl_pct:+.2f}%{Style.RESET_ALL}" if pnl_pct else "—",
            f"${pnl_usd:.2f}" if pnl_usd else "—",
            r["reason"],
        ])
    print(tabulate(summary_rows,
                   headers=["ETF","Qty","Entry","Exit","P&L%","P&L $","Reason"],
                   tablefmt="simple"))
    colour = Fore.GREEN if total_pnl >= 0 else Fore.RED
    print(f"\n  {colour}Total est. P&L : ${total_pnl:+.2f}{Style.RESET_ALL}")
    print(f"  Return on $1,000 : {total_pnl / TOTAL_BUDGET * 100:+.2f}%")
    print(f"\n{'═'*65}\n")


if __name__ == "__main__":
    main()
