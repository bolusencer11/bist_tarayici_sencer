"""
BIST Sinyal Günlük Performans — v4

v3'e göre yenilikler:
1. Günlük % hücrelerinde o günün HA bayrağı ikon olarak gösterilir:
   "+2.31% 🟢" = o gün alt fitilsiz dolgun yeşil HA mumu (AL)
   "-1.80% 🔴" = o gün üst fitilsiz dolgun kırmızı HA mumu (SAT)
   (JSON'da ha alanı olan günler için — eski kayıtlarda ikon çıkmaz)
2. Sinyal günündeki HA bayrağı ayrı kolonda (Skor'un yanında)
v3 özellikleri korundu: Yahoo'da eksik günlerin JSON'dan doldurulması,
cache zehirlenmesi koruması, GitHub yenileme butonu.
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import base64
import requests
import time
from datetime import datetime, timedelta

st.set_page_config(page_title="Sinyal Günlük Performans", page_icon="📅", layout="wide")
st.title("📅 Sinyal Günlük Performans")
st.caption("Kayıtlı bir sinyal günü seç — trend sonu uyarısı alan hisseler o günün kapanışında satılmış kabul edilir.")


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
        elif r.status_code == 404:
            return {}
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(symbols_tuple, start_str):
    """Boş sonuç cache'lenmez: exception fırlatılır, bir kez yeniden denenir."""
    start = datetime.fromisoformat(start_str) - timedelta(days=7)
    end = datetime.now() + timedelta(days=1)
    tickers = [s + ".IS" for s in symbols_tuple]
    for attempt in range(2):
        data = yf.download(
            tickers, start=start, end=end,
            group_by="ticker", auto_adjust=True, progress=False, threads=True,
        )
        if data is not None and not data.empty:
            return data
        time.sleep(4)
    raise RuntimeError("Yahoo veri döndürmedi (rate limit olabilir)")


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


def find_exit_date(history, symbol, signal_date):
    """Sinyal sonrası kayıtlarda ilk 'son' kategorisinin tarihi. Yoksa None."""
    for d in sorted(k for k in history.keys() if k > signal_date):
        info = history[d].get(symbol)
        if info and info.get("category") == "son":
            return d
    return None


def history_price_map(history, symbol, after_date_str):
    """Sinyal sonrası kayıtlı günlerdeki fiyatlar: {Timestamp: fiyat}."""
    out = {}
    for d, day in history.items():
        if d > after_date_str and symbol in day:
            p = day[symbol].get("price")
            if p:
                out[pd.Timestamp(d)] = float(p)
    return out


HA_ICON = {"AL": "🟢", "SAT": "🔴"}


def ha_icon_map(history, symbol, after_date_str):
    """Sinyal sonrası günlerdeki HA bayrakları: {date: '🟢'/'🔴'}."""
    out = {}
    for d, day in history.items():
        if d > after_date_str and symbol in day:
            flag = day[symbol].get("ha")
            if flag in HA_ICON:
                out[pd.Timestamp(d).date()] = HA_ICON[flag]
    return out


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
    st.info("Henüz kayıtlı sinyal geçmişi yok. Otomatik tarama ilk kaydı attığında burada görünecek.")
    st.stop()


# ============================================================
# Sidebar
# ============================================================

dates = sorted(history.keys(), reverse=True)

with st.sidebar:
    st.header("⚙ Ayarlar")

    if st.button("🔄 Veriyi GitHub'dan yenile", use_container_width=True):
        github_get_history.clear()
        st.rerun()

    selected_date = st.selectbox("📅 Kayıtlı sinyal günü", dates)

    day_data = history[selected_date]
    categories = sorted({v.get("category", "?") for v in day_data.values()})
    default_cats = [c for c in categories if c in ("yeni", "aktif")] or categories
    cat_filter = st.multiselect(
        "Kategori filtresi", categories, default=default_cats,
        help="Genelde 'yeni' ve 'aktif' seçilir — 'son' zaten çıkış sinyalidir, giriş adayı değildir",
    )
    min_score = st.slider("Minimum skor", 0, 100, 0)

    st.divider()
    apply_exit = st.checkbox(
        "🔴 Trend sonu uyarısında sat",
        value=True,
        help="Sonraki kayıtlarda 'son' kategorisi görülen hisse, o günün kapanışında satılmış kabul edilir.",
    )

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

try:
    data = fetch_prices(tuple(symbols), selected_date)
except Exception:
    st.error("Yahoo Finance şu an veri vermiyor (muhtemelen geçici limit). "
             "Birkaç dakika sonra tekrar dene — başarısız deneme cache'lenmez.")
    st.stop()

single = len(symbols) == 1
sig_date = pd.Timestamp(selected_date)
today_ts = pd.Timestamp(datetime.now().date())

rows = []
icons = {}            # (sembol, tarih) -> 🟢/🔴
all_dates = set()
filled_days = set()
n_sold = 0

for sym in symbols:
    close = extract_close_series(data, sym, single)
    if close is None:
        continue

    on_or_before = close[close.index <= sig_date]
    if not on_or_before.empty and on_or_before.index[-1] == sig_date:
        baseline = float(on_or_before.iloc[-1])
    else:
        baseline = candidates[sym].get("price") or (
            float(on_or_before.iloc[-1]) if not on_or_before.empty else None
        )
    if not baseline or baseline <= 0:
        continue

    after = close[close.index > sig_date].copy()

    # Yahoo'da eksik günleri JSON kayıtlarından doldur
    for ts, price in history_price_map(history, sym, selected_date).items():
        if ts <= today_ts and ts not in after.index:
            after.loc[ts] = price
            filled_days.add(ts.date())
    after = after.sort_index()

    if after.empty:
        continue

    # O hissenin sinyal sonrası HA bayrakları
    for d, icon in ha_icon_map(history, sym, selected_date).items():
        icons[(sym, d)] = icon

    exit_date_str = find_exit_date(history, sym, selected_date) if apply_exit else None
    exit_ts = pd.Timestamp(exit_date_str) if exit_date_str else None

    daily = {}
    prev = baseline
    last_price = baseline
    exit_price = None
    sold = False

    for dt, price in after.items():
        if exit_ts is not None and dt > exit_ts:
            sold = True
            break
        pct = (float(price) / prev - 1.0) * 100.0
        daily[dt.date()] = pct
        prev = float(price)
        last_price = float(price)
        all_dates.add(dt.date())
        if exit_ts is not None and dt == exit_ts:
            exit_price = float(price)
            sold = True
            break

    if exit_ts is not None and exit_price is None and sold:
        exit_price = last_price

    total = ((exit_price if sold and exit_price else last_price) / baseline - 1.0) * 100.0

    if sold:
        n_sold += 1
        status = f"🔴 Satıldı ({pd.Timestamp(exit_date_str).strftime('%d.%m')})"
    else:
        status = "🟢 Açık"

    info = candidates[sym]
    row = {
        "Sembol": sym,
        "Skor": info.get("score", 0),
        "HA": HA_ICON.get(info.get("ha"), "—") + (" " + info.get("ha") if info.get("ha") in HA_ICON else ""),
        "Kategori": info.get("category", "-"),
        "Sinyal Fiyatı": baseline,
        "Durum": status,
        "Toplam %": total,
    }
    row.update(daily)
    rows.append(row)

if not rows:
    if sig_date.date() >= datetime.now().date():
        st.info("⏳ Bu sinyal günü bugüne ait — performans ölçümü için henüz işlem günü geçmedi. "
                "İlk günlük % kolonu yarınki kapanıştan sonra oluşacak.")
    else:
        st.error("Fiyat verisi alınamadı. Bağlantı sorunu olabilir.")
    st.stop()


# ============================================================
# Tabloyu kur
# ============================================================

date_cols = sorted(all_dates)
df = pd.DataFrame(rows)

base_cols = ["Sembol", "Skor", "HA", "Kategori", "Sinyal Fiyatı", "Durum"]
df = df[base_cols + [d for d in date_cols if d in df.columns] + ["Toplam %"]]
df = df.sort_values("Toplam %", ascending=False).reset_index(drop=True)

rename_map = {d: d.strftime("%d.%m") for d in date_cols}
df = df.rename(columns=rename_map)
pct_cols = list(rename_map.values()) + ["Toplam %"]


def color_pct(v):
    if pd.isna(v):
        return "background-color: #f5f5f5"
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


# Renk matrisi SAYISAL değerlerden hesaplanır (ikon eklenmeden önce)
styles_df = pd.DataFrame("", index=df.index, columns=df.columns)
for c in pct_cols:
    styles_df[c] = df[c].map(color_pct)
styles_df["Skor"] = df["Skor"].map(color_score)

# Görüntü tablosu: sayılar formatlanır, günlük hücrelere HA ikonu eklenir
df_disp = df.copy()
for d, col in rename_map.items():
    df_disp[col] = [
        (f"{v:+.2f}%" + (f" {icons[(sym, d)]}" if (sym, d) in icons else "")) if pd.notna(v) else ""
        for sym, v in zip(df["Sembol"], df[col])
    ]
df_disp["Toplam %"] = df["Toplam %"].map(lambda v: f"{v:+.2f}%" if pd.notna(v) else "")
df_disp["Sinyal Fiyatı"] = df["Sinyal Fiyatı"].map(lambda v: f"₺{v:.2f}")
df_disp["Skor"] = df["Skor"].map(lambda v: f"{v:.0f}")

styled = df_disp.style.apply(lambda _: styles_df, axis=None)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Hisse", len(df))
c2.metric("🟢 Açık", len(df) - n_sold)
c3.metric("🔴 Satıldı", n_sold)
c4.metric("Ortalama getiri", f"{df['Toplam %'].mean():+.2f}%")
c5.metric("Pozitif getiri oranı", f"{(df['Toplam %'] > 0).mean() * 100:.0f}%")

if filled_days:
    gunler = ", ".join(d.strftime("%d.%m") for d in sorted(filled_days))
    st.warning(f"⚠ Yahoo verisinde eksik olan şu günler JSON tarama fiyatlarıyla dolduruldu: **{gunler}**. "
               "Bu değerler 17:33 tarama anı fiyatıdır (~kapanış); Yahoo barı geldiğinde resmi kapanışa döner.")

if datetime.now().date() in all_dates:
    st.caption("ℹ Bugünün kolonu seans kapanana kadar anlık fiyatı gösterir; kapanışta kesinleşir.")

st.divider()
st.dataframe(styled, use_container_width=True, hide_index=True)
st.caption(
    "Her tarih kolonu o günün bir önceki kapanışa göre **günlük % değişimidir**. "
    "Hücredeki **🟢** = o gün HA alt fitilsiz dolgun yeşil mum (AL), **🔴** = üst fitilsiz dolgun kırmızı mum (SAT) "
    "— HA kayıtları olan günler için gösterilir. Soldaki **HA kolonu** sinyal günündeki bayraktır. "
    "🔴 Satılan hisselerde satış, trend sonu uyarısının kaydedildiği günün kapanış fiyatından yapılmış kabul edilir; "
    "sonraki günler gri/boş bırakılır ve **Toplam %** o günde donar. 🟢 Açık pozisyonlarda Toplam % son kapanışa göredir."
)
st.caption(
    "📐 Korelasyon okuma notu: 🟢 ikonlu hücrenin kendisinin yeşil olması beklenen bir durumdur (büyük yükseliş, "
    "fitilsiz mumu zaten üretir). Asıl bakılacak şey ikonun **sonraki** hücreleri: 🟢 sonrası yeşil devam ediyorsa "
    "HA süreklilik sinyali olarak işe yarıyor demektir."
)
