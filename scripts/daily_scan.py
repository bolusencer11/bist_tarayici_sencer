"""
Günlük BIST trend taraması — GitHub Actions için standalone script.

Streamlit'e ihtiyaç duymaz. Repo kökündeki bist_tickers.txt'yi okur,
tüm hisseleri analiz eder, results/score_history.json dosyasına
bugünün skorlarını ekler (90 günden eski kayıtları temizler).

Piyasa kapalıysa (bugüne ait veri yoksa) kayıt YAPMAZ ve sessizce çıkar
— böylece resmi tatillerde bayat/mükerrer kayıt oluşmaz.

Konum: scripts/daily_scan.py
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parent.parent
TICKERS_FILE = REPO_ROOT / "bist_tickers.txt"
HISTORY_FILE = REPO_ROOT / "results" / "score_history.json"
TZ = ZoneInfo("Europe/Istanbul")


# ============================================================
# Analiz fonksiyonları (Aktif Trendler v2 ile birebir aynı mantık)
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


def analyze_ticker(df):
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

    macd_above = bool(last["MACD"] > last["MACD_signal"])
    macd_hist_growing = bool(last["MACD_hist"] > prev["MACD_hist"]) if pd.notna(prev["MACD_hist"]) else False
    vol_ratio = float(last["vol_ma5"] / last["vol_ma20"]) if last["vol_ma20"] and last["vol_ma20"] > 0 else 1.0

    return {
        "fiyat": float(last["Close"]),
        "rsi": float(last["RSI"]) if pd.notna(last["RSI"]) else None,
        "macd_above": macd_above,
        "macd_hist_growing": macd_hist_growing,
        "vol_ratio": vol_ratio,
        "category": category,
        "last_bar_date": df.index[-1],
    }


def compute_score(a):
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
        volume = 15
    elif vr >= 1.2:
        volume = 34
    elif vr >= 1.0:
        volume = 25
    elif vr >= 0.7:
        volume = 15
    else:
        volume = 5

    return momentum + health + volume, momentum, health, volume


# ============================================================
# Ana akış
# ============================================================

def main():
    today = datetime.now(TZ).date()
    print(f"[{datetime.now(TZ):%Y-%m-%d %H:%M}] Tarama başlıyor (TR saati)")

    # Tickers
    if not TICKERS_FILE.exists():
        print(f"HATA: {TICKERS_FILE} bulunamadı.")
        sys.exit(1)
    with open(TICKERS_FILE, encoding="utf-8") as f:
        tickers = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    print(f"{len(tickers)} hisse okunacak")

    # Veri indir (batch)
    end = datetime.now(TZ)
    start = end - timedelta(days=90)
    all_data = {}
    batch_size = 50

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            d = yf.download(
                batch, start=start.date(), end=end.date() + timedelta(days=1),
                group_by="ticker", auto_adjust=True, progress=False, threads=True,
            )
        except Exception as e:
            print(f"Batch {i // batch_size + 1} indirilemedi: {e}")
            continue

        if len(batch) == 1:
            t = batch[0]
            if not d.empty and "Close" in d.columns:
                sub = d.dropna(subset=["Close"])
                if len(sub) > 35:
                    all_data[t] = sub
        else:
            for t in batch:
                try:
                    if t in d.columns.get_level_values(0):
                        sub = d[t].dropna(subset=["Close"])
                        if len(sub) > 35:
                            all_data[t] = sub
                except (KeyError, AttributeError):
                    continue

    print(f"{len(all_data)} hisse için veri alındı")
    if not all_data:
        print("HATA: Hiç veri indirilemedi.")
        sys.exit(1)

    # Piyasa açık mıydı? En güncel bar tarihi bugüne ait değilse kayıt yapma.
    latest_bar = max(
        pd.to_datetime(df.index[-1]).date() for df in all_data.values()
    )
    if latest_bar < today:
        print(f"Bugüne ({today}) ait bar yok — en güncel veri {latest_bar}. "
              "Piyasa kapalı görünüyor, kayıt atlanıyor.")
        sys.exit(0)

    # Analiz
    today_scores = {}
    errors = 0
    for ticker, df in all_data.items():
        try:
            a = analyze_ticker(df)
        except Exception:
            errors += 1
            continue
        if a is None or a["category"] in ("yok", "olgun"):
            continue
        score, mom, health, vol = compute_score(a)
        symbol = ticker.replace(".IS", "")
        today_scores[symbol] = {
            "score": score,
            "momentum": mom,
            "health": health,
            "volume": vol,
            "price": round(a["fiyat"], 2),
            "rsi": round(a["rsi"], 1) if a["rsi"] is not None else None,
            "category": a["category"],
        }

    cats = {}
    for v in today_scores.values():
        cats[v["category"]] = cats.get(v["category"], 0) + 1
    print(f"Sinyal dağılımı: {cats} | analiz hatası: {errors}")

    if not today_scores:
        print("Kaydedilecek sinyal yok — dosyaya dokunulmadı.")
        sys.exit(0)

    # Geçmişi yükle, bugünü ekle, buda, yaz
    history = {}
    if HISTORY_FILE.exists():
        try:
            history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"UYARI: mevcut geçmiş okunamadı ({e}), sıfırdan başlanıyor.")

    history[today.isoformat()] = today_scores
    cutoff = (today - timedelta(days=90)).isoformat()
    history = {k: v for k, v in history.items() if k >= cutoff}

    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✅ {len(today_scores)} hissenin skoru kaydedildi → {HISTORY_FILE.relative_to(REPO_ROOT)} "
          f"(toplam {len(history)} günlük kayıt)")


if __name__ == "__main__":
    main()
