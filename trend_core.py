"""
trend_core.py — Ortak trend analiz motoru

Hem Streamlit Aktif Trendler sayfası hem de scripts/daily_scan.py
bu modülü import eder. Formül değişiklikleri SADECE bu dosyada yapılır.

Konum: repo kökü (bist_tickers.txt ile aynı seviye)

Skor (0-100) — üç bileşen, Heikin Ashi SKORA DAHİL DEĞİLDİR:
  Momentum (MACD) : 0-33
  Sağlık  (RSI)   : 0-33
  Hacim           : 5-34

Heikin Ashi — yalnızca gösterge/bayrak:
  🟢 AL  : alt fitilsiz dolgun yeşil HA mumu (güçlü yükseliş)
  🔴 SAT : üst fitilsiz dolgun kırmızı HA mumu (güçlü düşüş)
  —      : diğer tüm mum tipleri (sinyal yok)
"""
import numpy as np
import pandas as pd


# ============================================================
# Temel göstergeler
# ============================================================

def compute_rsi(close, length=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(length).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig


# ============================================================
# Heikin Ashi (gösterge — skora dahil değil)
# ============================================================

def compute_heikin_ashi(df):
    """HA open/high/low/close dizilerini döndürür (numpy)."""
    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)

    ha_close = (o + h + l + c) / 4.0
    ha_open = np.empty(len(df))
    ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    ha_high = np.maximum.reduce([h, ha_open, ha_close])
    ha_low = np.minimum.reduce([l, ha_open, ha_close])
    return ha_open, ha_high, ha_low, ha_close


def classify_ha_candle(o, h, l, c, wick_tol=0.10):
    """
    Tek HA mumu:
      strong_bull : yeşil + alt fitil ≤ aralığın %10'u (fitilsiz kabul edilir)
      strong_bear : kırmızı + üst fitil ≤ aralığın %10'u
      other       : diğer her şey (normal mum, doji vb.)
    """
    rng = h - l
    if rng <= 0:
        return "other"
    upper = h - max(o, c)
    lower = min(o, c) - l

    if c >= o and lower / rng <= wick_tol:
        return "strong_bull"
    if c < o and upper / rng <= wick_tol:
        return "strong_bear"
    return "other"


def analyze_heikin_ashi(df, lookback=5):
    """
    Son HA mumuna göre AL/SAT bayrağı üret.

    Döndürür: dict
      signal : "AL" | "SAT" | None
      streak : sondan geriye kesintisiz aynı-tip güçlü mum sayısı
    """
    if len(df) < 5 or not {"Open", "High", "Low", "Close"}.issubset(df.columns):
        return None

    ha_o, ha_h, ha_l, ha_c = compute_heikin_ashi(df)
    n = len(df)
    classes = [
        classify_ha_candle(ha_o[i], ha_h[i], ha_l[i], ha_c[i])
        for i in range(max(0, n - lookback), n)
    ]
    last = classes[-1]

    if last == "strong_bull":
        signal = "AL"
    elif last == "strong_bear":
        signal = "SAT"
    else:
        return {"signal": None, "streak": 0}

    streak = 0
    for cls in reversed(classes):
        if cls == last:
            streak += 1
        else:
            break

    return {"signal": signal, "streak": streak}


def ha_label(ha):
    """Tablo kolonu için kısa etiket."""
    if ha is None or ha.get("signal") is None:
        return "—"
    if ha["signal"] == "AL":
        return f"🟢 AL ×{ha['streak']}" if ha["streak"] >= 2 else "🟢 AL"
    return f"🔴 SAT ×{ha['streak']}" if ha["streak"] >= 2 else "🔴 SAT"


# ============================================================
# Ana analiz
# ============================================================

def analyze_ticker(df):
    """Tek hisse için tam analiz. Crossover tespiti numpy ile (dtype güvenli)."""
    if len(df) < 35:
        return None
    df = df.copy()

    df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA30"] = df["Close"].ewm(span=30, adjust=False).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = compute_macd(df["Close"])
    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["vol_ma5"] = df["Volume"].rolling(5).mean()

    above = df["EMA10"].to_numpy() > df["EMA30"].to_numpy()
    n = len(above)
    cross_up_pos = np.where(above[1:] & ~above[:-1])[0] + 1
    cross_down_pos = np.where(~above[1:] & above[:-1])[0] + 1
    days_up = int(n - 1 - cross_up_pos[-1]) if len(cross_up_pos) else None
    days_down = int(n - 1 - cross_down_pos[-1]) if len(cross_down_pos) else None

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    above_now = bool(above[-1])

    if above_now:
        if days_up is not None and days_up <= 2:
            category = "yeni"
        elif days_up is not None and days_up <= 10:
            category = "aktif"
        else:
            category = "olgun"
    else:
        category = "son" if (days_down is not None and days_down <= 5) else "yok"

    ema_spread = ((last["EMA10"] - last["EMA30"]) / last["EMA30"]) * 100 if last["EMA30"] else 0
    macd_above = bool(last["MACD"] > last["MACD_signal"])
    macd_hist_growing = bool(last["MACD_hist"] > prev["MACD_hist"]) if pd.notna(prev["MACD_hist"]) else False
    vol_ratio = float(last["vol_ma5"] / last["vol_ma20"]) if last["vol_ma20"] and last["vol_ma20"] > 0 else 1.0

    # Heikin Ashi göstergesi (OHLC eksikse None — sistem çalışmaya devam eder)
    try:
        ha = analyze_heikin_ashi(df)
    except Exception:
        ha = None

    return {
        "fiyat": float(last["Close"]),
        "rsi": float(last["RSI"]) if pd.notna(last["RSI"]) else None,
        "macd_above": macd_above,
        "macd_hist_growing": macd_hist_growing,
        "vol_ratio": vol_ratio,
        "ema_spread": float(ema_spread),
        "days_since_cross_up": days_up,
        "days_since_cross_down": days_down,
        "category": category,
        "ha": ha,
    }


def compute_score(a):
    """
    0-100 trend sağlık skoru: MACD (33) + RSI (33) + Hacim (34).
    Heikin Ashi skora DAHİL DEĞİLDİR — yalnızca gösterge kolonudur.
    Döndürür: (skor, momentum, health, volume)
    """
    if a["macd_above"]:
        momentum = 33 if a["macd_hist_growing"] else 20
    else:
        momentum = 0

    rsi = a["rsi"]
    if rsi is None:
        health = 0
    elif 50 <= rsi <= 65:
        health = 33
    elif 65 < rsi <= 75:
        health = 25
    elif 45 <= rsi < 50:
        health = 15
    elif rsi > 75:
        health = 10
    else:
        health = 0

    vr = a["vol_ratio"]
    if vr >= 2.0:
        volume = 15  # blow-off riski
    elif vr >= 1.2:
        volume = 34
    elif vr >= 1.0:
        volume = 25
    elif vr >= 0.7:
        volume = 15
    else:
        volume = 5

    return momentum + health + volume, momentum, health, volume


def make_comment(a, score):
    """Türkçe yorum üret. HA yalnızca AL/SAT bayrağı varsa belirtilir."""
    if score >= 75:
        general = "Güçlü trend"
    elif score >= 50:
        general = "Trend devam ediyor"
    elif score >= 25:
        general = "Trend zayıflıyor"
    else:
        general = "Trend bozuluyor"

    parts = []

    ha = a.get("ha")
    if ha and ha.get("signal") == "AL":
        k = ha["streak"]
        parts.append(f"HA {k} gündür fitilsiz yeşil (güçlü yükseliş)" if k >= 2
                     else "HA fitilsiz yeşil (güçlü yükseliş)")
    elif ha and ha.get("signal") == "SAT":
        k = ha["streak"]
        parts.append(f"HA {k} gündür fitilsiz kırmızı (güçlü düşüş)" if k >= 2
                     else "HA fitilsiz kırmızı (güçlü düşüş)")

    rsi = a["rsi"]
    if rsi is not None:
        if rsi > 75:
            parts.append(f"RSI {rsi:.0f} aşırı uzanmış")
        elif rsi < 45:
            parts.append(f"RSI {rsi:.0f} momentum zayıf")

    if not a["macd_above"]:
        parts.append("MACD aşağı kesti")
    elif not a["macd_hist_growing"]:
        parts.append("MACD yavaşlıyor")

    vr = a["vol_ratio"]
    if vr >= 2.0:
        parts.append(f"hacim {vr:.1f}× anormal (blow-off riski)")
    elif vr < 0.7:
        parts.append("hacim zayıf")
    elif vr >= 1.2:
        parts.append("hacim destekli")

    return general + (" — " + ", ".join(parts) if parts else "")
