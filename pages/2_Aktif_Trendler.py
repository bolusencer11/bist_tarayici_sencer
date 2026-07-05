"""
BIST Aktif Trend Takip Sayfası — DÜZELTİLMİŞ SÜRÜM

Değişiklikler:
1. Crossover tespiti pandas shift/fillna/~ zinciri yerine saf numpy ile yapılıyor
   (pandas sürümüne bağlı dtype/pozisyonel indeksleme bug'ı gideriliyor)
2. 🔍 Debug paneli eklendi: kategori dağılımı ve 3-10 gün bandı kontrolü

Üç bölüm:
1. 🆕 Yeni Sinyaller — Son 1-2 işlem günü içinde EMA10↑EMA30 crossover
2. 📈 Aktif Trendler — Son 3-10 işlem günü crossover, trend hala canlı
3. ⚠ Trend Sonu Uyarısı — EMA10↓EMA30 crossover (çıkış sinyali)
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import base64
import requests
from datetime import datetime, timedelta
from pathlib import Path

st.set_page_config(page_title="BIST Aktif Trendler", page_icon="📈", layout="wide")
st.title("📈 BIST Aktif Trend Takibi")
st.caption("Yeni sinyaller · Devam eden trendler · Trend sonu uyarıları — tek ekranda")


# ============================================================
# Yardımcı fonksiyonlar
# ============================================================

def load_tickers():
    p = Path(__file__).parent.parent / "bist_tickers.txt"
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
    hist = macd - sig
    return macd, sig, hist


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

    # --- Crossover tespiti: saf numpy, pandas dtype tuzaklarından bağımsız ---
    above = df["EMA10"].to_numpy() > df["EMA30"].to_numpy()  # bool array
    n = len(above)

    # i. günde kesişim: bugün üstte, dün değildi (ilk gün kesişim sayılmaz)
    cross_up_pos = np.where(above[1:] & ~above[:-1])[0] + 1
    cross_down_pos = np.where(~above[1:] & above[:-1])[0] + 1

    days_since_cross_up = int(n - 1 - cross_up_pos[-1]) if len(cross_up_pos) else None
    days_since_cross_down = int(n - 1 - cross_down_pos[-1]) if len(cross_down_pos) else None

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    above_now = bool(above[-1])

    # Kategori belirleme
    if above_now:
        if days_since_cross_up is not None and days_since_cross_up <= 2:
            category = "yeni"
        elif days_since_cross_up is not None and days_since_cross_up <= 10:
            category = "aktif"
        else:
            category = "olgun"
    else:
        if days_since_cross_down is not None and days_since_cross_down <= 5:
            category = "son"
        else:
            category = "yok"

    ema_spread = ((last["EMA10"] - last["EMA30"]) / last["EMA30"]) * 100 if last["EMA30"] else 0
    macd_above = bool(last["MACD"] > last["MACD_signal"])
    macd_hist_growing = bool(last["MACD_hist"] > prev["MACD_hist"]) if pd.notna(prev["MACD_hist"]) else False
    vol_ratio = float(last["vol_ma5"] / last["vol_ma20"]) if last["vol_ma20"] and last["vol_ma20"] > 0 else 1.0

    return {
        "fiyat": float(last["Close"]),
        "rsi": float(last["RSI"]) if pd.notna(last["RSI"]) else None,
        "macd_above": macd_above,
        "macd_hist_growing": macd_hist_growing,
        "vol_ratio": vol_ratio,
        "ema_spread": float(ema_spread),
        "days_since_cross_up": days_since_cross_up,
        "days_since_cross_down": days_since_cross_down,
        "category": category,
    }


def compute_score(a):
    """0-100 trend sağlık skoru: MACD (33) + RSI (33) + Hacim (34)."""
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
    """Türkçe yorum üret."""
    if score >= 75:
        general = "Güçlü trend"
    elif score >= 50:
        general = "Trend devam ediyor"
    elif score >= 25:
        general = "Trend zayıflıyor"
    else:
        general = "Trend bozuluyor"

    parts = []
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


def github_save_history(history, sha=None):
    try:
        token = st.secrets["GITHUB_TOKEN"]
        repo = st.secrets["GITHUB_REPO"]
    except (KeyError, FileNotFoundError):
        return False, "secrets ayarlı değil"

    url = f"https://api.github.com/repos/{repo}/contents/{GH_HISTORY_PATH}"
    headers = {"Authorization": f"token {token}"}
    content = json.dumps(history, ensure_ascii=False, indent=2)
    encoded = base64.b64encode(content.encode()).decode()
    payload = {
        "message": f"Trend skor güncelleme {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "content": encoded,
    }
    if sha:
        payload["sha"] = sha
    try:
        r = requests.put(url, headers=headers, json=payload, timeout=15)
        if r.status_code in (200, 201):
            return True, None
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, str(e)


def prune_history(history, days_keep=90):
    if not history:
        return history
    cutoff = (datetime.now() - timedelta(days=days_keep)).date().isoformat()
    return {k: v for k, v in history.items() if k >= cutoff}


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

    cols = ["Sembol", "Fiyat", days_label, "EMA Açılım %", "MACD", "RSI", "Hacim ×", "Skor", "Δ Skor", "Yorum"]
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

    st.divider()
    save_to_gh = st.checkbox(
        "📤 Bugünü GitHub'a kaydet",
        value=False,
        help="results/score_history.json dosyasına bugünkü skorları ekler"
    )

    run = st.button("▶ Analizi Çalıştır", type="primary", use_container_width=True)


# ============================================================
# Ana akış
# ============================================================

if not run:
    st.info("Sol menüden ayarları yapıp **▶ Analizi Çalıştır** butonuna bas.")
    st.markdown("""
    ### Üç bölümün anlamı

    **🆕 Yeni Sinyaller (0-2 gün):** Son 1-2 işlem gününde EMA10, EMA30'u yukarı kesmiş hisseler. Bunlar potansiyel **yeni giriş** adayları.

    **📈 Aktif Trendler (3-10 gün):** Kesişimden sonra 3-10 gün geçmiş, trend hala canlı. Pozisyon açtıysan **buradayken sistem sana "tut" diyor**. Listeden düşerse çıkış zamanı yaklaşıyor demektir.

    **⚠ Trend Sonu Uyarısı:** EMA10, EMA30'u aşağı kesmiş hisseler (son 5 gün). **Açık pozisyonun varsa çıkış sinyali.**

    ### Skor mantığı (0-100)

    Üç eşit ağırlıklı bileşen:
    - **Momentum (MACD)** — MACD sinyal üstünde ve histogram büyüyorsa tam puan
    - **Sağlık (RSI)** — 50-65 ideal; 75 üstü aşırı uzanmış; 45 altı zayıf
    - **Hacim** — Son 5 gün / 20 gün ortalama oranı (1.2× üstü ideal, 2.0× şüpheli blow-off)

    **Δ Skor** kolonu dünkü skordan değişimi gösterir — asıl izleyeceğin metrik bu.
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

# --- Geçmişi al ---
history, sha = github_get_history()
delta_lookup = {}
last_history_date = None

if history:
    today_key = datetime.now().date().isoformat()
    past_dates = sorted([k for k in history.keys() if k < today_key])
    if past_dates:
        last_history_date = past_dates[-1]
        last_scores = history[last_history_date]
        for sym, info in last_scores.items():
            delta_lookup[sym] = info.get("score", 0)

# --- Analiz ---
new_signals, active_trends, trend_endings = [], [], []
today_scores = {}
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

    today_scores[symbol] = {
        "score": score,
        "momentum": mom,
        "health": health,
        "volume": vol,
        "price": round(a["fiyat"], 2),
        "rsi": round(a["rsi"], 1) if a["rsi"] is not None else None,
        "category": a["category"],
    }

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

# --- GitHub'a kaydet ---
if save_to_gh:
    if history is None:
        st.warning("⚠ GitHub secrets ayarlanmamış. Sol menüde anlatılan adımları kontrol et.")
    elif not today_scores:
        st.warning("Kaydedilecek skor yok.")
    else:
        today_key = datetime.now().date().isoformat()
        history[today_key] = today_scores
        history = prune_history(history, days_keep=90)
        ok, err = github_save_history(history, sha=sha)
        if ok:
            st.success(f"✅ {len(today_scores)} hissenin skoru GitHub'a kaydedildi ({today_key})")
        else:
            st.error(f"GitHub'a kayıt başarısız: {err}")

# --- Özet ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("📥 Veri", f"{len(all_data)} hisse")
c2.metric("🆕 Yeni", len(new_signals))
c3.metric("📈 Aktif", len(active_trends))
c4.metric("⚠ Trend sonu", len(trend_endings))

if last_history_date:
    st.caption(f"📅 Önceki skor karşılaştırması: {last_history_date}")
else:
    st.caption("📅 Henüz tarihsel kayıt yok — Δ Skor kolonu boş gelecek. Bugünü kaydedince yarın görmeye başlarsın.")

# --- Debug paneli ---
with st.expander("🔍 Debug: Kategori dağılımı", expanded=False):
    if analyze_errors:
        st.warning(f"⚠ {analyze_errors} hisse analiz sırasında hata verdi ve atlandı.")
    if debug_rows:
        dbg = pd.DataFrame(debug_rows)
        counts = dbg["Kategori"].value_counts()
        st.write("**Kategori dağılımı:**")
        st.write(counts.to_frame("Adet").T)

        aktif_band = dbg[dbg["Yukarı kesişim (gün önce)"].between(3, 10, inclusive="both")]
        st.write(f"**3-10 gün bandında yukarı kesişimi olan hisse:** {len(aktif_band)}")
        if len(aktif_band):
            st.dataframe(aktif_band, use_container_width=True, hide_index=True)

        st.caption(
            "3-10 gün bandı doluysa ama Aktif tablosu boşsa min skor eşiğini kontrol et. "
            "Band da boşsa piyasada gerçekten o yaşta trend yok demektir (nadir ama mümkün)."
        )
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
