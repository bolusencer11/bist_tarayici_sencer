"""
BIST Backtest — Filtre Karşılaştırma Sayfası (v2)

5 senaryoyu aynı veri üzerinde paralel çalıştırır:
  0. Baseline:   sadece EMA10↑EMA30 crossover
  1. + EMA200:   fiyat EMA200 üstündeyse al
  2. + Hacim:    crossover günü hacim 20g ortalamanın 1.5× üstüyse al
  3. + RSI:      RSI 45-65 arasındaysa al
  4. Hepsi:      üç filtreyi birden uygula

Çıkış kuralı (tüm senaryolarda aynı): EMA10↓EMA30 ters crossover
Benchmark: XU100.IS (BIST-100)
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

st.set_page_config(page_title="BIST Backtest — Filtre Testi", page_icon="🧪", layout="wide")
st.title("🧪 BIST Backtest — Filtre Etki Analizi")
st.caption("5 senaryo · Aynı dönem · Aynı evren · Hangi filtre işe yarıyor?")


# ============================================================
# Yardımcılar
# ============================================================

def load_tickers():
    p = Path(__file__).parent.parent / "bist_tickers.txt"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_batch(tickers_tuple, days_back):
    end = datetime.now()
    start = end - timedelta(days=days_back + 60)  # EMA200 için tampon
    data = yf.download(
        list(tickers_tuple),
        start=start, end=end,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    return data


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_benchmark(days_back):
    end = datetime.now()
    start = end - timedelta(days=days_back + 60)
    bist = yf.download("XU100.IS", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(bist.columns, pd.MultiIndex):
        bist.columns = bist.columns.get_level_values(0)
    return bist


def compute_rsi(close, length=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(length).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def prepare_indicators(df):
    """Tek hisse için tüm göstergeleri hesapla."""
    df = df.copy()
    df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA30"] = df["Close"].ewm(span=30, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["above"] = df["EMA10"] > df["EMA30"]
    df["cross_up"] = df["above"] & ~df["above"].shift(1).fillna(False)
    df["cross_down"] = ~df["above"] & df["above"].shift(1).fillna(False)
    return df


# ============================================================
# Backtest motoru
# ============================================================

def run_scenario(df, scenario, start_date):
    """
    Tek hisse, tek senaryo backtest.
    scenario: "baseline", "ema200", "volume", "rsi", "all"
    Dönüş: trade listesi [{"entry_date", "entry_price", "exit_date", "exit_price", "pnl_pct", "days"}]
    """
    trades = []
    # start_date öncesini ana göstergeler için kullan; sinyalleri start_date sonrasından al
    df_active = df[df.index >= start_date]
    if len(df_active) < 5:
        return trades

    in_position = False
    entry_date = None
    entry_price = None

    for i in range(len(df_active)):
        row = df_active.iloc[i]
        date = df_active.index[i]

        if not in_position:
            # Giriş sinyali var mı?
            if not row["cross_up"] or pd.isna(row["EMA200"]) or pd.isna(row["RSI"]) or pd.isna(row["vol_ma20"]):
                continue

            # Filtreleri uygula
            pass_ema200 = row["Close"] > row["EMA200"]
            pass_volume = row["Volume"] >= 1.5 * row["vol_ma20"] if row["vol_ma20"] > 0 else False
            pass_rsi = 45 <= row["RSI"] <= 65

            if scenario == "baseline":
                ok = True
            elif scenario == "ema200":
                ok = pass_ema200
            elif scenario == "volume":
                ok = pass_volume
            elif scenario == "rsi":
                ok = pass_rsi
            elif scenario == "all":
                ok = pass_ema200 and pass_volume and pass_rsi
            else:
                ok = False

            if ok:
                in_position = True
                entry_date = date
                entry_price = float(row["Close"])

        else:
            # Çıkış sinyali (ters crossover)
            if row["cross_down"]:
                exit_price = float(row["Close"])
                pnl_pct = (exit_price / entry_price - 1) * 100
                days = (date - entry_date).days
                trades.append({
                    "entry_date": entry_date.strftime("%Y-%m-%d"),
                    "entry_price": entry_price,
                    "exit_date": date.strftime("%Y-%m-%d"),
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "days": days,
                })
                in_position = False
                entry_date = None
                entry_price = None

    # Açık pozisyon kalırsa son fiyatla kapat
    if in_position:
        last = df_active.iloc[-1]
        exit_price = float(last["Close"])
        pnl_pct = (exit_price / entry_price - 1) * 100
        days = (df_active.index[-1] - entry_date).days
        trades.append({
            "entry_date": entry_date.strftime("%Y-%m-%d"),
            "entry_price": entry_price,
            "exit_date": df_active.index[-1].strftime("%Y-%m-%d") + " (açık)",
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
            "days": days,
        })

    return trades


def summarize(trades, benchmark_return_pct):
    """Trade listesinden özet metrikler."""
    if not trades:
        return {
            "n_trades": 0,
            "win_rate": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "expectancy": np.nan,
            "total_return": 0.0,
            "alpha": -benchmark_return_pct,
            "avg_days": np.nan,
        }
    pnls = np.array([t["pnl_pct"] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = len(wins) / len(pnls) * 100
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    expectancy = pnls.mean()
    # Bileşik getiri (her trade'i bir sonrakine bağlayarak)
    total_return = (np.prod(1 + pnls / 100) - 1) * 100
    avg_days = np.mean([t["days"] for t in trades])
    return {
        "n_trades": len(trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "total_return": total_return,
        "alpha": total_return - benchmark_return_pct,
        "avg_days": avg_days,
    }


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("⚙ Test Parametreleri")
    period_years = st.radio(
        "Test dönemi",
        options=[1, 2, 3],
        index=0,
        format_func=lambda x: f"{x} yıl",
        help="1 yıl ~2-3 dk, 3 yıl ~7-10 dk sürebilir"
    )

    universe_choice = st.radio(
        "Hisse evreni",
        options=["all", "top100", "custom"],
        index=0,
        format_func=lambda x: {
            "all": "Tüm BIST (~500 hisse)",
            "top100": "İlk 100 (alfabetik)",
            "custom": "Tek hisse seç"
        }[x]
    )

    custom_ticker = None
    if universe_choice == "custom":
        custom_ticker = st.text_input("Hisse kodu (örn. AKBNK)", value="AKBNK").upper().strip()

    st.divider()
    st.caption(
        "**Çıkış kuralı:** EMA10↓EMA30 ters crossover\n\n"
        "**Filtreler:**\n"
        "- EMA200: Close > EMA200\n"
        "- Hacim: Vol ≥ 1.5× 20g ortalama\n"
        "- RSI: 45 ≤ RSI ≤ 65"
    )

    run = st.button("▶ Backtest'i Çalıştır", type="primary", use_container_width=True)


# ============================================================
# Ana akış
# ============================================================

if not run:
    st.info("Sol menüden parametreleri seçip **▶ Backtest'i Çalıştır** butonuna bas.")
    st.markdown("""
    ### 5 Senaryo nedir?

    | # | Senaryo | Filtre |
    |---|---------|--------|
    | 0 | **Baseline** | Sadece EMA10↑EMA30 crossover (mevcut sistem) |
    | 1 | **+ EMA200** | Fiyat EMA200 üstündeyse al |
    | 2 | **+ Hacim** | Crossover günü hacim 20g ort. × 1.5 üstüyse al |
    | 3 | **+ RSI** | RSI 45-65 bandındaysa al |
    | 4 | **Hepsi** | Üç filtreyi birden uygula |

    ### Metrik kılavuzu

    - **İşlem sayısı:** Filtre ne kadar daraltıyor — düşmesi normal
    - **Kazanma oranı:** Trade'lerin yüzde kaçı kârla kapandı
    - **Ort. kazanç / kayıp:** Kazanan/kaybeden trade'lerin ortalama getirisi
    - **Beklenti:** Trade başına ortalama P&L — pozitif olmalı
    - **Toplam getiri:** Tüm trade'lerin bileşik getirisi
    - **Alfa:** BIST-100 getirisi düşülmüş — pozitif = piyasayı yendi

    ### Karar mantığı

    Bir filtre **iyi** sayılır eğer:
    - Trade sayısı düştüğü halde **beklenti yükseldi** (kalite arttı)
    - Toplam getiri baseline'dan yüksek
    - Alfa pozitif

    Bir filtre **kötü** sayılır eğer:
    - Trade sayısı çok düştüğü için istatistik anlamsızlaştı (örn. 10 altı)
    - Beklenti düştü (filtre değil, gürültü)

    "Hepsi" senaryosunun her zaman en iyi olması beklenmez — bazen tek bir filtre yeterli olur, üçü birlikte aşırı kısıtlayıcı olabilir.
    """)
    st.stop()


# --- Hisse listesi hazırla ---
all_tickers = load_tickers()
if not all_tickers:
    st.error("bist_tickers.txt bulunamadı.")
    st.stop()

if universe_choice == "all":
    tickers = all_tickers
elif universe_choice == "top100":
    tickers = all_tickers[:100]
else:
    if not custom_ticker:
        st.error("Hisse kodu boş.")
        st.stop()
    t = custom_ticker if custom_ticker.endswith(".IS") else f"{custom_ticker}.IS"
    tickers = [t]

days_back = period_years * 365
start_date = pd.Timestamp(datetime.now() - timedelta(days=days_back))

st.info(f"🔄 {len(tickers)} hisse · son {period_years} yıl · 5 senaryo paralel çalışıyor...")

# --- Benchmark ---
bench = fetch_benchmark(days_back)
if bench.empty or "Close" not in bench.columns:
    st.warning("XU100 verisi alınamadı, alfa hesaplanmayacak.")
    bench_return = 0
else:
    bench_active = bench[bench.index >= start_date]
    if len(bench_active) >= 2:
        bench_return = (bench_active["Close"].iloc[-1] / bench_active["Close"].iloc[0] - 1) * 100
    else:
        bench_return = 0

# --- Veriyi çek ---
all_data = {}
progress = st.progress(0.0, text="Veri indiriliyor...")
batch_size = 50
n_batches = (len(tickers) + batch_size - 1) // batch_size

for i in range(n_batches):
    batch = tickers[i * batch_size:(i + 1) * batch_size]
    try:
        d = fetch_batch(tuple(batch), days_back=days_back)
        if len(batch) == 1:
            t = batch[0]
            if not d.empty and "Close" in d.columns:
                sub = d.dropna(subset=["Close"])
                if len(sub) > 220:  # EMA200 için yeterli veri
                    all_data[t] = prepare_indicators(sub)
        else:
            for t in batch:
                try:
                    if t in d.columns.get_level_values(0):
                        sub = d[t].dropna(subset=["Close"])
                        if len(sub) > 220:
                            all_data[t] = prepare_indicators(sub)
                except (KeyError, AttributeError):
                    continue
    except Exception:
        continue
    progress.progress((i + 1) / n_batches, text=f"Veri indiriliyor... ({i+1}/{n_batches})")

progress.empty()

if not all_data:
    st.error("Hiç hisse verisi indirilemedi.")
    st.stop()

st.success(f"✅ {len(all_data)} hisse için veri hazır. Senaryolar çalıştırılıyor...")

# --- 5 senaryoyu çalıştır ---
scenarios = ["baseline", "ema200", "volume", "rsi", "all"]
scenario_labels = {
    "baseline": "0. Baseline",
    "ema200": "1. + EMA200",
    "volume": "2. + Hacim",
    "rsi": "3. + RSI",
    "all": "4. Hepsi",
}

all_trades = {s: [] for s in scenarios}  # senaryo → tüm hisselerin tüm trade'leri
trades_by_ticker = {s: {} for s in scenarios}  # senaryo → {sembol: trades}

progress2 = st.progress(0.0, text="Senaryolar işleniyor...")
total = len(all_data)
for idx, (ticker, df) in enumerate(all_data.items()):
    for s in scenarios:
        trades = run_scenario(df, s, start_date)
        if trades:
            for tr in trades:
                tr["ticker"] = ticker.replace(".IS", "")
            all_trades[s].extend(trades)
            trades_by_ticker[s][ticker.replace(".IS", "")] = trades
    progress2.progress((idx + 1) / total, text=f"Senaryolar işleniyor... ({idx+1}/{total})")

progress2.empty()

# --- Özet tablo ---
st.divider()
st.subheader("📊 Senaryo Karşılaştırma")

summary_rows = []
for s in scenarios:
    summary = summarize(all_trades[s], bench_return)
    summary_rows.append({
        "Senaryo": scenario_labels[s],
        "İşlem": summary["n_trades"],
        "Kazanma %": summary["win_rate"],
        "Ort. Kazanç %": summary["avg_win"],
        "Ort. Kayıp %": summary["avg_loss"],
        "Beklenti %": summary["expectancy"],
        "Toplam %": summary["total_return"],
        "Alfa %": summary["alpha"],
        "Ort. Gün": summary["avg_days"],
    })

summary_df = pd.DataFrame(summary_rows)


def highlight_best(s, higher_better=True):
    """Bir kolonda en iyi değeri yeşil yap."""
    if s.dtype == object:
        return [""] * len(s)
    valid = s.dropna()
    if len(valid) == 0:
        return [""] * len(s)
    target = valid.max() if higher_better else valid.min()
    return ["background-color: #c8e6c9; font-weight: 600" if v == target else "" for v in s]


styled_summary = (
    summary_df.style
    .apply(lambda s: highlight_best(s, True), subset=["Kazanma %", "Ort. Kazanç %", "Beklenti %", "Toplam %", "Alfa %"])
    .apply(lambda s: highlight_best(s, False), subset=["Ort. Kayıp %"])
    .format({
        "Kazanma %": lambda x: f"{x:.1f}" if pd.notna(x) else "-",
        "Ort. Kazanç %": lambda x: f"{x:+.2f}" if pd.notna(x) else "-",
        "Ort. Kayıp %": lambda x: f"{x:+.2f}" if pd.notna(x) else "-",
        "Beklenti %": lambda x: f"{x:+.2f}" if pd.notna(x) else "-",
        "Toplam %": "{:+.2f}",
        "Alfa %": "{:+.2f}",
        "Ort. Gün": lambda x: f"{x:.0f}" if pd.notna(x) else "-",
    })
)
st.dataframe(styled_summary, use_container_width=True, hide_index=True)

st.caption(f"📈 BIST-100 (benchmark) aynı dönemde: **{bench_return:+.2f}%** · Yeşil = o kolonda en iyi senaryo")

# --- Yorum ---
st.divider()
st.subheader("🧠 Hızlı Yorum")

baseline = summarize(all_trades["baseline"], bench_return)
all_filters = summarize(all_trades["all"], bench_return)

comments = []
if baseline["n_trades"] > 0 and all_filters["n_trades"] > 0:
    if all_filters["expectancy"] > baseline["expectancy"]:
        comments.append(f"✅ **Hepsi** senaryosu baseline'a göre daha iyi beklenti veriyor: {all_filters['expectancy']:+.2f}% vs {baseline['expectancy']:+.2f}%")
    else:
        comments.append(f"⚠ **Hepsi** senaryosu baseline'ı geçemedi: {all_filters['expectancy']:+.2f}% vs {baseline['expectancy']:+.2f}%")

    if all_filters["n_trades"] < baseline["n_trades"] * 0.2:
        comments.append(f"⚠ Hepsi senaryosu trade sayısını çok daralttı ({all_filters['n_trades']} vs {baseline['n_trades']}) — istatistik güveni düşük olabilir")

# En iyi tek filtre
single_filters = ["ema200", "volume", "rsi"]
best_single = max(single_filters, key=lambda s: summarize(all_trades[s], bench_return)["expectancy"] if all_trades[s] else -999)
best_single_summary = summarize(all_trades[best_single], bench_return)
if best_single_summary["n_trades"] > 0:
    comments.append(f"🏆 En iyi tek filtre: **{scenario_labels[best_single]}** (beklenti {best_single_summary['expectancy']:+.2f}%, {best_single_summary['n_trades']} trade)")

for c in comments:
    st.markdown(c)

# --- Detay: her senaryonun trade listesi ---
st.divider()
st.subheader("📋 Trade Detayları (senaryo seç)")

selected = st.selectbox(
    "Hangi senaryonun trade'lerini görmek istersin?",
    options=scenarios,
    format_func=lambda s: scenario_labels[s],
    index=4
)

if all_trades[selected]:
    detail_df = pd.DataFrame(all_trades[selected])
    detail_df = detail_df[["ticker", "entry_date", "entry_price", "exit_date", "exit_price", "pnl_pct", "days"]]
    detail_df.columns = ["Sembol", "Giriş", "Giriş ₺", "Çıkış", "Çıkış ₺", "P&L %", "Gün"]
    detail_df = detail_df.sort_values("P&L %", ascending=False).reset_index(drop=True)

    def color_pnl(v):
        if pd.isna(v):
            return ""
        if v > 5:
            return "background-color: #c8e6c9; color: #1b5e20"
        elif v > 0:
            return "background-color: #e8f5e9"
        elif v > -5:
            return "background-color: #ffebee"
        else:
            return "background-color: #ffcdd2; color: #b71c1c"

    styled_detail = (
        detail_df.style
        .map(color_pnl, subset=["P&L %"])
        .format({
            "Giriş ₺": "₺{:.2f}",
            "Çıkış ₺": "₺{:.2f}",
            "P&L %": "{:+.2f}",
        })
    )
    st.dataframe(styled_detail, use_container_width=True, hide_index=True, height=600)

    # CSV indir
    csv = detail_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇ {scenario_labels[selected]} trade'lerini CSV indir",
        csv,
        file_name=f"backtest_{selected}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
else:
    st.info("Bu senaryoda hiç trade yok (filtre çok kısıtlayıcı olabilir).")
