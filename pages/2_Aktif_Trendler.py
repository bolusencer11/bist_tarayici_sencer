"""
BIST Aktif Trend Takip Sayfası — v4 (Heikin Ashi gösterge kolonu, skora dahil değil)

Analiz/skorlama mantığı repo kökündeki trend_core.py modülünden gelir.
Yenilikler:
- Heikin Ashi mum analizi: skorlamada -12..+10 düzeltme + tabloda HA kolonu
- Güçlü yükseliş / doji dönüş sinyalleri yorumda açıkça belirtilir
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import base64
import requests
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trend_core import analyze_ticker, compute_score, make_comment, ha_label

st.set_page_config(page_title="BIST Aktif Trendler", page_icon="📈", layout="wide")
st.title("📈 BIST Aktif Trend Takibi")
st.caption("Yeni sinyaller · Devam eden trendler · Trend sonu uyarıları — tek ekranda")


# ============================================================
# Veri yardımcıları
# ============================================================

def load_tickers():
    p = ROOT / "bist_tickers.txt"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_batch(tickers_tuple, days_back=90):
    end = datetime.now()
    start = end - timedelta(days=days_back)
    data = yf.download(
        list(tickers_tuple),
        start=start, end=end,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    return data


# ============================================================
# GitHub geçmiş yönetimi
# ============================================================

GH_HISTORY_PATH = "results/score_history.json"


def github_get_history():
    try:
        token = st.secrets["GITHUB_TOKEN"]
        repo = st.secrets["GITHUB_REPO"]
    except (KeyError, FileNotFoundError):
        return None, None

    url = f"https://api.github.com/repos/{repo}/contents/{GH_HISTORY_PATH}"
    headers = {"Authorization": f"token {token}"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            data = r.json()
            content = base64.b64decode(data["content"]).decode("utf-8")
            return json.loads(content), data["sha"]
        elif r.status_code == 404:
            return {}, None
    except Exception:
        pass
    return None, None


# ============================================================
# UI Yardımcıları
# ============================================================

def color_score(v):
    if pd.isna(v):
        return ""
    if v >= 75:
        return "background-color: #c8e6c9; color: #1b5e20; font-weight: 600"
    elif v >= 50:
        return "background-color: #fff9c4; color: #f57f17"
    elif v >= 25:
        return "background-color: #ffe0b2; color: #e65100"
    else:
        return "background-color: #ffcdd2; color: #b71c1c; font-weight: 600"


def color_delta(v):
    if pd.isna(v) or v is None:
        return ""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ""
    if v >= 10:
        return "color: #1b5e20; font-weight: 600"
    elif v >= 5:
        return "color: #2e7d32"
    elif v <= -10:
        return "color: #b71c1c; font-weight: 600"
    elif v <= -5:
        return "color: #c62828"
    return ""


def render_table(rows, section_title, days_label="Trend Yaşı"):
    if not rows:
        st.info(f"{section_title} kategorisinde hisse yok.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values("Skor", ascending=False).reset_index(drop=True)

    cols = ["Sembol", "HA", "Fiyat", days_label, "EMA Açılım %", "MACD", "RSI", "Hacim ×", "Skor", "Δ Skor", "Yorum"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    styled = (
        df.style
        .map(color_score, subset=["Skor"])
        .map(color_delta, subset=["Δ Skor"] if "Δ Skor" in df.columns else [])
        .format({
            "Fiyat": "₺{:.2f}",
            "EMA Açılım %": "{:+.2f}%",
            "RSI": lambda x: f"{x:.0f}" if pd.notna(x) else "-",
            "Hacim ×": "{:.2f}×",
            "Skor": "{:.0f}",
            "Δ Skor": lambda x: f"{x:+.0f}" if pd.notna(x) else "-",
        })
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("⚙ Ayarlar")
    show_new = st.checkbox("🆕 Yeni Sinyaller (0-2 gün)", value=True)
    show_active = st.checkbox("📈 Aktif Trendler (3-10 gün)", value=True)
    show_end = st.checkbox("⚠ Trend Sonu Uyarısı", value=True)

    st.divider()
    min_score_active = st.slider(
        "Aktif Trendlerde minimum skor",
        0, 100, 30,
        help="Bu eşiğin altındaki düşük skorlu aktif trendler gizlenir"
    )

    run = st.button("▶ Analizi Çalıştır", type="primary", use_container_width=True)


# ============================================================
# Ana akış
# ============================================================

if not run:
    st.info("Sol menüden ayarları yapıp **▶ Analizi Çalıştır** butonuna bas.")
    st.markdown("""
    ### Üç bölümün anlamı

    **🆕 Yeni Sinyaller (0-2 gün):** Son 1-2 işlem gününde EMA10, EMA30'u yukarı kesmiş hisseler. Potansiyel **yeni giriş** adayları.

    **📈 Aktif Trendler (3-10 gün):** Kesişimden 3-10 gün geçmiş, trend hala canlı. Pozisyon açtıysan **buradayken sistem "tut" diyor**.

    **⚠ Trend Sonu Uyarısı:** EMA10, EMA30'u aşağı kesmiş hisseler (son 5 gün). **Açık pozisyonun varsa çıkış sinyali.**

    ### Skor mantığı (0-100)

    Üç eşit ağırlıklı bileşen (Heikin Ashi SKORA DAHİL DEĞİLDİR):
    - **Momentum (MACD)** — sinyal üstünde ve histogram büyüyorsa tam puan (33)
    - **Sağlık (RSI)** — 50-65 ideal (33); 75 üstü aşırı; 45 altı zayıf
    - **Hacim** — 5g/20g oranı; 1.2× üstü ideal (34), 2.0× şüpheli blow-off

    **HA kolonu (bağımsız gösterge):** 🟢 AL ×N = N gündür alt fitilsiz dolgun yeşil HA mumu · 🔴 SAT = üst fitilsiz dolgun kırmızı HA mumu · — = sinyal yok

    **Δ Skor** dünkü kayıtlı skordan değişim — asıl izlenecek metrik. Not: günlük kayıtları artık GitHub Actions otomatik atıyor, bu sayfa yalnızca görüntüleme yapar.
    """)
    st.stop()


# --- Veriyi çek ---
tickers = load_tickers()
if not tickers:
    st.error("bist_tickers.txt dosyası bulunamadı (repo kökünde olmalı).")
    st.stop()

st.info(f"📥 {len(tickers)} hisse için son 90 günlük veri indiriliyor...")
progress = st.progress(0.0, text="İndiriliyor...")

all_data = {}
batch_size = 50
n_batches = (len(tickers) + batch_size - 1) // batch_size

for i in range(n_batches):
    batch = tickers[i * batch_size:(i + 1) * batch_size]
    try:
        d = fetch_batch(tuple(batch), days_back=90)
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
    except Exception:
        continue
    progress.progress((i + 1) / n_batches, text=f"İndiriliyor... ({i+1}/{n_batches})")

progress.empty()

if not all_data:
    st.error("Veri indirilemedi. İnternet bağlantısını kontrol et.")
    st.stop()

# --- Geçmişi al (Δ Skor için) ---
history, _ = github_get_history()
delta_lookup = {}
last_history_date = None

if history:
    today_key = datetime.now().date().isoformat()
    past_dates = sorted([k for k in history.keys() if k < today_key])
    if past_dates:
        last_history_date = past_dates[-1]
        for sym, info in history[last_history_date].items():
            delta_lookup[sym] = info.get("score", 0)

# --- Analiz ---
new_signals, active_trends, trend_endings = [], [], []
debug_rows = []
analyze_errors = 0

for ticker, df in all_data.items():
    try:
        a = analyze_ticker(df)
    except Exception:
        analyze_errors += 1
        continue
    if a is None:
        continue

    debug_rows.append({
        "Sembol": ticker.replace(".IS", ""),
        "Kategori": a["category"],
        "Yukarı kesişim (gün önce)": a["days_since_cross_up"],
        "Aşağı kesişim (gün önce)": a["days_since_cross_down"],
    })

    if a["category"] in ("yok", "olgun"):
        continue

    score, mom, health, vol = compute_score(a)
    comment = make_comment(a, score)
    symbol = ticker.replace(".IS", "")

    macd_label = (
        "✅ Yukarı + güçleniyor" if a["macd_above"] and a["macd_hist_growing"]
        else "⚠ Yukarı ama yavaşlıyor" if a["macd_above"]
        else "❌ Aşağı kesti"
    )

    delta = None
    if symbol in delta_lookup:
        delta = score - delta_lookup[symbol]

    base_row = {
        "Sembol": symbol,
        "HA": ha_label(a.get("ha")),
        "Fiyat": a["fiyat"],
        "EMA Açılım %": a["ema_spread"],
        "MACD": macd_label,
        "RSI": a["rsi"],
        "Hacim ×": a["vol_ratio"],
        "Skor": score,
        "Δ Skor": delta,
        "Yorum": comment,
    }

    if a["category"] == "yeni":
        base_row["Trend Yaşı"] = a["days_since_cross_up"]
        new_signals.append(base_row)
    elif a["category"] == "aktif":
        if score >= min_score_active:
            base_row["Trend Yaşı"] = a["days_since_cross_up"]
            active_trends.append(base_row)
    elif a["category"] == "son":
        base_row["Bozulma (gün)"] = a["days_since_cross_down"]
        trend_endings.append(base_row)

# --- Özet ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("📥 Veri", f"{len(all_data)} hisse")
c2.metric("🆕 Yeni", len(new_signals))
c3.metric("📈 Aktif", len(active_trends))
c4.metric("⚠ Trend sonu", len(trend_endings))

if last_history_date:
    st.caption(f"📅 Önceki skor karşılaştırması: {last_history_date}")
else:
    st.caption("📅 Henüz tarihsel kayıt yok — Δ Skor kolonu boş gelecek.")

# --- Debug paneli ---
with st.expander("🔍 Debug: Kategori dağılımı", expanded=False):
    if analyze_errors:
        st.warning(f"⚠ {analyze_errors} hisse analiz sırasında hata verdi ve atlandı.")
    if debug_rows:
        dbg = pd.DataFrame(debug_rows)
        st.write("**Kategori dağılımı:**")
        st.write(dbg["Kategori"].value_counts().to_frame("Adet").T)
    else:
        st.write("Analiz edilebilen hisse yok.")

st.divider()

# --- Bölümleri göster ---
if show_new:
    st.subheader("🆕 Yeni Sinyaller (son 2 gün crossover)")
    st.caption("Potansiyel yeni giriş adayları. Skor ne kadar yüksekse, trend o kadar sağlıklı başlamış.")
    render_table(new_signals, "Yeni Sinyal", days_label="Trend Yaşı")
    st.divider()

if show_active:
    st.subheader("📈 Aktif Trendler (3-10 gün crossover)")
    st.caption("Trend hala canlı. Açık pozisyonun buradayken sistem **tut** diyor.")
    render_table(active_trends, "Aktif Trend", days_label="Trend Yaşı")
    st.divider()

if show_end:
    st.subheader("⚠ Trend Sonu Uyarısı (son 5 gün)")
    st.caption("EMA10 aşağı kesti. Açık pozisyonun varsa **çıkış sinyali**.")
    render_table(trend_endings, "Trend Sonu", days_label="Bozulma (gün)")
