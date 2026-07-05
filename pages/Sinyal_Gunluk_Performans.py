"""
BIST Sinyal Günlük Performans Sayfası

Dropdown'dan score_history.json içindeki kayıtlı bir günü seç →
o tarihte sinyal veren hisselerin, kayıt gününden BUGÜNE kadar
her işlem günündeki günlük % yükseliş/düşüşü tabloda gösterilir.

Tablo yapısı:
Sembol | Skor | Kategori | 27.06 | 30.06 | 01.07 | ... | Toplam %
(her tarih kolonu = o günün bir önceki kapanışa göre günlük % değişimi)
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import base64
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="Sinyal Günlük Performans", page_icon="📅", layout="wide")
st.title("📅 Sinyal Günlük Performans")
st.caption("Kayıtlı bir sinyal günü seç — o günden bugüne hisselerin günlük % hareketlerini ve o günkü skoru gör.")


# ============================================================
# GitHub'dan score_history.json çek
# ============================================================

GH_HISTORY_PATH = "results/score_history.json"


@st.cache_data(ttl=1800, show_spinner=False)
def github_get_history():
    try:
        token = st.secrets["GITHUB_TOKEN"]
        repo = st.secrets["GITHUB_REPO"]
    except (KeyError, FileNotFoundError):
        return None

    url = f"https://api.github.com/repos/{repo}/contents/{GH_HISTORY_PATH}"
    headers = {"Authorization": f"token {token}"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            content = base64.b64decode(r.json()["content"]).decode("utf-8")
            return json.loads(content)
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(symbols_tuple, start_str):
    """Kayıt tarihinden bugüne fiyat verisi. Baseline için 7 gün geriden başla."""
    start = datetime.fromisoformat(start_str) - timedelta(days=7)
    end = datetime.now() + timedelta(days=1)
    tickers = [s + ".IS" for s in symbols_tuple]
    data = yf.download(
        tickers,
        start=start, end=end,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    return data


def extract_close_series(data, symbol, single):
    try:
        if single:
            s = data["Close"].dropna()
        else:
            s = data[symbol + ".IS"]["Close"].dropna()
        if s.empty:
            return None
        s = s.copy()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        return s
    except (KeyError, AttributeError):
        return None


# ============================================================
# Veri kaynağı
# ============================================================

history = github_get_history()

if history is None:
    st.warning("GitHub secrets bulunamadı veya dosya çekilemedi. JSON dosyasını manuel yükleyebilirsin.")
    uploaded = st.file_uploader("score_history.json yükle", type=["json"])
    if uploaded:
        try:
            history = json.load(uploaded)
        except Exception:
            st.error("JSON okunamadı — dosya formatını kontrol et.")
            st.stop()

if not history:
    st.info("Henüz kayıtlı sinyal geçmişi yok. Önce Aktif Trendler sayfasında **Bugünü GitHub'a kaydet** ile skor biriktir.")
    st.stop()


# ============================================================
# Sidebar — tarih seçimi ve filtreler
# ============================================================

dates = sorted(history.keys(), reverse=True)

with st.sidebar:
    st.header("⚙ Ayarlar")
    selected_date = st.selectbox("📅 Kayıtlı sinyal günü", dates)

    day_data = history[selected_date]
    categories = sorted({v.get("category", "?") for v in day_data.values()})
    cat_filter = st.multiselect("Kategori filtresi", categories, default=categories)
    min_score = st.slider("Minimum skor", 0, 100, 0)

    run = st.button("▶ Tabloyu Oluştur", type="primary", use_container_width=True)

candidates = {
    sym: info for sym, info in day_data.items()
    if info.get("category") in cat_filter and info.get("score", 0) >= min_score
}

st.subheader(f"📅 {selected_date} — kayıtlı {len(candidates)} hisse")

if not candidates:
    st.warning("Seçilen filtrelere uyan hisse yok.")
    st.stop()

if not run:
    st.info("Sol menüden tarihi seçip **▶ Tabloyu Oluştur** butonuna bas.")
    st.stop()


# ============================================================
# Fiyat verisi çek
# ============================================================

symbols = sorted(candidates.keys())
st.info(f"📥 {len(symbols)} hisse için {selected_date} → bugün fiyat verisi indiriliyor...")
data = fetch_prices(tuple(symbols), selected_date)
single = len(symbols) == 1

sig_date = pd.Timestamp(selected_date)

rows = []
all_dates = set()

for sym in symbols:
    close = extract_close_series(data, sym, single)
    if close is None:
        continue

    # Baseline: sinyal günü kapanışı (veride yoksa history'deki kayıtlı fiyat)
    on_or_before = close[close.index <= sig_date]
    if not on_or_before.empty and on_or_before.index[-1] == sig_date:
        baseline = float(on_or_before.iloc[-1])
    else:
        baseline = candidates[sym].get("price") or (
            float(on_or_before.iloc[-1]) if not on_or_before.empty else None
        )
    if not baseline or baseline <= 0:
        continue

    # Sinyal gününden SONRAKİ işlem günleri
    after = close[close.index > sig_date]
    if after.empty:
        continue

    # Günlük % değişim: ilk gün baseline'a göre, sonrakiler bir önceki kapanışa göre
    daily = {}
    prev = baseline
    for dt, price in after.items():
        pct = (float(price) / prev - 1.0) * 100.0
        daily[dt.date()] = pct
        prev = float(price)
        all_dates.add(dt.date())

    total = (float(after.iloc[-1]) / baseline - 1.0) * 100.0

    info = candidates[sym]
    row = {
        "Sembol": sym,
        "Skor": info.get("score", 0),
        "Kategori": info.get("category", "-"),
        "Sinyal Fiyatı": baseline,
        "Toplam %": total,
    }
    row.update(daily)
    rows.append(row)

if not rows:
    st.error("Fiyat verisi alınamadı. Tarih çok yeni olabilir (henüz işlem günü geçmemiş) veya bağlantı sorunu var.")
    st.stop()


# ============================================================
# Tabloyu kur
# ============================================================

date_cols = sorted(all_dates)
df = pd.DataFrame(rows)

# Kolon sırası: kimlik → günlük kolonlar (kronolojik) → toplam
base_cols = ["Sembol", "Skor", "Kategori", "Sinyal Fiyatı"]
df = df[base_cols + [d for d in date_cols if d in df.columns] + ["Toplam %"]]
df = df.sort_values("Toplam %", ascending=False).reset_index(drop=True)

# Tarih kolonlarını GG.AA formatına çevir
rename_map = {d: d.strftime("%d.%m") for d in date_cols}
df = df.rename(columns=rename_map)
pct_cols = list(rename_map.values()) + ["Toplam %"]


def color_pct(v):
    if pd.isna(v):
        return ""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ""
    if v >= 5:
        return "background-color: #a5d6a7; color: #1b5e20; font-weight: 600"
    elif v >= 2:
        return "background-color: #c8e6c9; color: #1b5e20"
    elif v > 0:
        return "color: #2e7d32"
    elif v <= -5:
        return "background-color: #ef9a9a; color: #b71c1c; font-weight: 600"
    elif v <= -2:
        return "background-color: #ffcdd2; color: #b71c1c"
    elif v < 0:
        return "color: #c62828"
    return ""


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


styled = (
    df.style
    .map(color_pct, subset=pct_cols)
    .map(color_score, subset=["Skor"])
    .format({
        "Sinyal Fiyatı": "₺{:.2f}",
        "Skor": "{:.0f}",
        **{c: lambda x: f"{x:+.2f}%" if pd.notna(x) else "-" for c in pct_cols},
    })
)

# Özet metrikler
c1, c2, c3, c4 = st.columns(4)
c1.metric("Hisse", len(df))
c2.metric("Geçen işlem günü", len(date_cols))
c3.metric("Ortalama toplam getiri", f"{df['Toplam %'].mean():+.2f}%")
c4.metric("Pozitif getiri oranı", f"{(df['Toplam %'] > 0).mean() * 100:.0f}%")

st.divider()
st.dataframe(styled, use_container_width=True, hide_index=True)
st.caption(
    "Her tarih kolonu, o günün **bir önceki kapanışa göre günlük % değişimidir**. "
    "İlk kolon sinyal günü kapanışına göredir. **Toplam %** = sinyal gününden bugüne kümülatif getiri. "
    "**Skor**, Aktif Trendler scriptinin o tarihte verdiği 0-100 trend sağlık skorudur."
)
