"""
Günlük BIST trend taraması — GitHub Actions için standalone script.

Analiz mantığı repo kökündeki trend_core.py modülünden gelir;
formül değişiklikleri SADECE orada yapılır.

- Piyasa kapalıysa (bugüne ait bar yoksa) kayıt yapmaz.
- Bugün zaten kayıtlıysa üzerine yazmaz (FORCE_OVERWRITE=1 hariç)
  → 18:23 yedek taraması 17:33 ana taramayı ezmez.

Konum: scripts/daily_scan.py
"""
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from trend_core import analyze_ticker, compute_score  # noqa: E402

TICKERS_FILE = REPO_ROOT / "bist_tickers.txt"
HISTORY_FILE = REPO_ROOT / "results" / "score_history.json"
TZ = ZoneInfo("Europe/Istanbul")


def main():
    today = datetime.now(TZ).date()
    print(f"[{datetime.now(TZ):%Y-%m-%d %H:%M}] Tarama başlıyor (TR saati)")

    force = os.environ.get("FORCE_OVERWRITE", "").lower() in ("1", "true", "yes")

    # Geçmişi en başta yükle — bugün zaten kayıtlıysa yedek tarama
    # veri indirmeye hiç girmeden çıkar (üzerine yazmaz).
    history = {}
    if HISTORY_FILE.exists():
        try:
            history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"UYARI: mevcut geçmiş okunamadı ({e}), sıfırdan başlanıyor.")

    if today.isoformat() in history and not force:
        print(f"Bugün ({today}) zaten kayıtlı — üzerine yazılmıyor, çıkılıyor. "
              "(Elle üzerine yazmak için FORCE_OVERWRITE=1)")
        sys.exit(0)

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

    # Piyasa açık mıydı? En güncel bar bugüne ait değilse kayıt yapma.
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
        ha = a.get("ha")
        today_scores[symbol] = {
            "score": score,
            "momentum": mom,
            "health": health,
            "volume": vol,
            "ha": (ha or {}).get("signal"),
            "ha_streak": (ha or {}).get("streak", 0),
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

    # Bugünü ekle, buda, yaz
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
