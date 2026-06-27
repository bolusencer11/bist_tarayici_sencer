"""
Aktif Trend Backtest — Skor Sistemi Doğrulama

Aktif Trendler sayfasındaki skor mantığını geçmişte simüle eder:
- Her gün "Yeni Sinyal" kategorisindeki hisseleri tara
- Skoru ≥ eşik olan en yüksek skorlu hisseleri al
- "Trend Sonu" kategorisine girene kadar tut, oraya girince sat
- 3 eşik (70, 80, 90) yan yana karşılaştır

Sermaye modeli: 10 eşit slot, her slot kendi içinde compound.
Maks 5 yeni pozisyon/gün, maks 10 açık pozisyon toplam.
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

st.set_page_config(page_title="Aktif Trend Backtest", page_icon="🎯", layout="wide")
st.title("🎯 Aktif Trend Backtest")
st.caption("Skor sisteminin geçmiş performansı — 3 eşik yan yana")


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
    start = end - timedelta(days=days_back + 90)  # tampon
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
    start = end - timedelta(days=days_back + 90)
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


def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def prepare_indicators(df):
    """Tüm indikatörleri + her satır için kategori + skor ekle (vectorize)."""
    df = df.copy()
    df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA30"] = df["Close"].ewm(span=30, adjust=False).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = compute_macd(df["Close"])
    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["vol_ma5"] = df["Volume"].rolling(5).mean()

    df["above"] = df["EMA10"] > df["EMA30"]
    df["cross_up"] = df["above"] & ~df["above"].shift(1).fillna(False)
    df["cross_down"] = ~df["above"] & df["above"].shift(1).fillna(False)

    # Her satır için: son cross_up/down'tan kaç gün geçti (vectorize)
    n = len(df)
    pos = pd.Series(np.arange(n), index=df.index)

    cross_up_pos = pos.where(df["cross_up"]).ffill()
    df["days_since_cross_up"] = (pos - cross_up_pos).astype("float")

    cross_down_pos = pos.where(df["cross_down"]).ffill()
    df["days_since_cross_down"] = (pos - cross_down_pos).astype("float")

    # Kategori: yeni / aktif / olgun / son / yok
    def categorize(row):
        if row["above"]:
            d = row["days_since_cross_up"]
            if pd.isna(d):
                return "yok"
            if d <= 2:
                return "yeni"
            elif d <= 10:
                return "aktif"
            else:
                return "olgun"
        else:
            d = row["days_since_cross_down"]
            if pd.isna(d):
                return "yok"
            if d <= 5:
                return "son"
            else:
                return "yok"

    df["category"] = df.apply(categorize, axis=1)

    # Skor bileşenleri (Aktif Trendler ile birebir aynı formül)
    macd_above = df["MACD"] > df["MACD_signal"]
    macd_hist_growing = df["MACD_hist"] > df["MACD_hist"].shift(1)

    momentum = pd.Series(0, index=df.index, dtype="float")
    momentum[macd_above & macd_hist_growing] = 33
    momentum[macd_above & ~macd_hist_growing] = 20

    rsi = df["RSI"]
    health = pd.Series(0, index=df.index, dtype="float")
    health[(rsi >= 50) & (rsi <= 65)] = 33
    health[(rsi > 65) & (rsi <= 75)] = 25
    health[(rsi >= 45) & (rsi < 50)] = 15
    health[rsi > 75] = 10

    vol_ratio = (df["vol_ma5"] / df["vol_ma20"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    volume = pd.Series(5, index=df.index, dtype="float")
    volume[vol_ratio >= 2.0] = 15  # blow-off
    volume[(vol_ratio >= 1.2) & (vol_ratio < 2.0)] = 34
    volume[(vol_ratio >= 1.0) & (vol_ratio < 1.2)] = 25
    volume[(vol_ratio >= 0.7) & (vol_ratio < 1.0)] = 15

    df["score"] = momentum + health + volume
    df["vol_ratio"] = vol_ratio

    return df


# ============================================================
# Simülasyon motoru
# ============================================================

def simulate(prepared_data, threshold, trade_dates, max_positions=10, max_new_per_day=5, start_capital_per_slot=10000.0):
    """
    Slot bazlı portföy simülasyonu.

    prepared_data: {ticker: df with indicators, category, score}
    threshold: minimum skor (örn. 70)
    trade_dates: simülasyonun yapılacağı işlem günleri (DatetimeIndex)
    """
    slots = [
        {"ticker": None, "entry_date": None, "entry_price": None, "entry_capital": start_capital_per_slot, "capital": start_capital_per_slot}
        for _ in range(max_positions)
    ]
    closed_trades = []
    equity_curve = []  # (date, total_equity)
    skipped_signals = 0  # slot doluysa atlanan sinyaller

    for current_date in trade_dates:
        # === 1. Çıkış kontrolü ===
        for slot in slots:
            if slot["ticker"] is None:
                continue
            ticker = slot["ticker"]
            df = prepared_data.get(ticker)
            if df is None or current_date not in df.index:
                continue
            row = df.loc[current_date]
            if row["category"] == "son":
                exit_price = float(row["Close"])
                pnl_pct = (exit_price / slot["entry_price"] - 1) * 100
                new_capital = slot["entry_capital"] * (1 + pnl_pct / 100)
                closed_trades.append({
                    "ticker": ticker.replace(".IS", ""),
                    "entry_date": slot["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": current_date.strftime("%Y-%m-%d"),
                    "entry_price": slot["entry_price"],
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "days": (current_date - slot["entry_date"]).days,
                    "entry_score": slot.get("entry_score"),
                })
                slot["capital"] = new_capital
                slot["ticker"] = None
                slot["entry_date"] = None
                slot["entry_price"] = None
                slot["entry_capital"] = new_capital  # bir sonraki pozisyonda bu sermaye kullanılacak

        # === 2. Yeni sinyal taraması ===
        held = {s["ticker"] for s in slots if s["ticker"]}
        candidates = []
        for ticker, df in prepared_data.items():
            if ticker in held:
                continue
            if current_date not in df.index:
                continue
            row = df.loc[current_date]
            if row["category"] == "yeni" and row["score"] >= threshold:
                candidates.append((ticker, float(row["score"]), float(row["Close"])))

        candidates.sort(key=lambda x: -x[1])  # en yüksek skor başta

        new_count = 0
        for ticker, score, price in candidates:
            if new_count >= max_new_per_day:
                break
            empty_slot = next((s for s in slots if s["ticker"] is None), None)
            if empty_slot is None:
                skipped_signals += len(candidates) - new_count
                break
            empty_slot["ticker"] = ticker
            empty_slot["entry_date"] = current_date
            empty_slot["entry_price"] = price
            empty_slot["entry_capital"] = empty_slot["capital"]
            empty_slot["entry_score"] = score
            new_count += 1

        # === 3. Equity curve günlük güncelle ===
        total_eq = 0.0
        for slot in slots:
            if slot["ticker"] is None:
                total_eq += slot["capital"]
            else:
                df = prepared_data.get(slot["ticker"])
                if df is not None and current_date in df.index:
                    current_price = float(df.loc[current_date, "Close"])
                    unrealized = slot["entry_capital"] * (current_price / slot["entry_price"])
                    total_eq += unrealized
                else:
                    total_eq += slot["entry_capital"]
        equity_curve.append((current_date, total_eq))

    # Simülasyon sonu: hâlâ açık pozisyonları son fiyatla kapat (raporlama için)
    for slot in slots:
        if slot["ticker"] is None:
            continue
        ticker = slot["ticker"]
        df = prepared_data.get(ticker)
        if df is None:
            continue
        last_date = trade_dates[-1]
        if last_date in df.index:
            exit_price = float(df.loc[last_date, "Close"])
            pnl_pct = (exit_price / slot["entry_price"] - 1) * 100
            closed_trades.append({
                "ticker": ticker.replace(".IS", ""),
                "entry_date": slot["entry_date"].strftime("%Y-%m-%d"),
                "exit_date": last_date.strftime("%Y-%m-%d") + " (açık)",
                "entry_price": slot["entry_price"],
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "days": (last_date - slot["entry_date"]).days,
                "entry_score": slot.get("entry_score"),
            })

    return {
        "trades": closed_trades,
        "equity_curve": equity_curve,
        "final_equity": equity_curve[-1][1] if equity_curve else max_positions * start_capital_per_slot,
        "initial_equity": max_positions * start_capital_per_slot,
        "skipped_signals": skipped_signals,
    }


def summarize_sim(sim_result, bench_return_pct):
    trades = sim_result["trades"]
    initial = sim_result["initial_equity"]
    final = sim_result["final_equity"]
    total_return = (final / initial - 1) * 100

    if not trades:
        return {
            "n_trades": 0,
            "win_rate": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "expectancy": np.nan,
            "best": np.nan,
            "worst": np.nan,
            "avg_days": np.nan,
            "total_return": total_return,
            "alpha": total_return - bench_return_pct,
            "skipped": sim_result["skipped_signals"],
        }

    pnls = np.array([t["pnl_pct"] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    return {
        "n_trades": len(trades),
        "win_rate": len(wins) / len(pnls) * 100,
        "avg_win": wins.mean() if len(wins) else 0,
        "avg_loss": losses.mean() if len(losses) else 0,
        "expectancy": pnls.mean(),
        "best": pnls.max(),
        "worst": pnls.min(),
        "avg_days": np.mean([t["days"] for t in trades]),
        "total_return": total_return,
        "alpha": total_return - bench_return_pct,
        "skipped": sim_result["skipped_signals"],
    }


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("⚙ Test Parametreleri")
    period_months = st.radio(
        "Test dönemi",
        options=[3, 6],
        index=0,
        format_func=lambda x: f"{x} ay (~{x*21} işlem günü)"
    )

    universe_choice = st.radio(
        "Hisse evreni",
        options=["all", "top100"],
        index=0,
        format_func=lambda x: "Tüm BIST (~500 hisse)" if x == "all" else "İlk 100 hisse (alfabetik)"
    )

    st.divider()
    st.caption(
        "**Sabit parametreler:**\n\n"
        "- Eşikler: **70, 80, 90** (karşılaştırmalı)\n"
        "- Maks açık pozisyon: **10**\n"
        "- Maks yeni/gün: **5**\n"
        "- Slot başına sermaye: **₺10.000**\n"
        "- Giriş: 'Yeni Sinyal' kategorisi + skor ≥ eşik\n"
        "- Çıkış: 'Trend Sonu' kategorisi"
    )

    run = st.button("▶ Backtest'i Çalıştır", type="primary", use_container_width=True)

    if "atb_results" in st.session_state:
        if st.button("🗑 Sonuçları Temizle", use_container_width=True):
            del st.session_state["atb_results"]
            st.rerun()


# ============================================================
# Ana akış
# ============================================================

if not run and "atb_results" not in st.session_state:
    st.info("Sol menüden parametreleri seçip **▶ Backtest'i Çalıştır** butonuna bas.")
    st.markdown("""
    ### Bu sayfa ne yapıyor?

    Aktif Trendler sayfasındaki **skor sisteminin gerçekten para kazandırıp kazandırmadığını** geçmiş veriyle test eder.

    **Simülasyon mantığı:**

    1. Test döneminin **her işlem gününde**, sanki o gün Aktif Trendler sayfasını açmış gibi tarama yapar
    2. "Yeni Sinyal" kategorisindeki (son 0-2 günde EMA10↑EMA30) hisselere bakar
    3. Skoru eşiğin üzerinde olanları, en yüksek skordan başlayarak alır (max 5 yeni/gün)
    4. Maks 10 açık pozisyon tutar; daha fazla sinyal gelirse atlar
    5. Her gün açık pozisyonları kontrol eder; **"Trend Sonu" kategorisine giren** hisseleri satar

    **3 eşik yan yana:**

    | Eşik | Beklenen davranış |
    |------|---|
    | **70** | Daha çok sinyal, daha gürültülü, kazanma oranı düşer |
    | **80** | Orta yol |
    | **90** | Az sinyal, ama her biri yüksek kaliteli olmalı |

    En iyi eşik **alfasının** (BIST-100 üstü getirinin) en yüksek olduğudur.

    **Lookahead bias yok:** Her gün için sadece o güne kadarki veri kullanılır — gerçek hayatta sayfayı her gün açmış gibi.
    """)
    st.stop()


# === Backtest çalıştır ===
if run:
    all_tickers = load_tickers()
    if not all_tickers:
        st.error("bist_tickers.txt bulunamadı.")
        st.stop()

    tickers = all_tickers if universe_choice == "all" else all_tickers[:100]
    days_back = period_months * 30

    st.info(f"📥 {len(tickers)} hisse · son {period_months} ay simülasyonu hazırlanıyor...")

    # Benchmark
    bench = fetch_benchmark(days_back)

    # Veri çek
    all_data = {}
    debug_msgs = []
    progress = st.progress(0.0, text="Veri indiriliyor...")
    batch_size = 50
    n_batches = (len(tickers) + batch_size - 1) // batch_size

    for i in range(n_batches):
        batch = tickers[i * batch_size:(i + 1) * batch_size]
        try:
            d = fetch_batch(tuple(batch), days_back=days_back)
            if d is None or d.empty:
                continue
            if isinstance(d.columns, pd.MultiIndex):
                level0 = d.columns.get_level_values(0).unique()
                for t in batch:
                    if t in level0:
                        try:
                            sub = d[t].dropna(subset=["Close"])
                            if len(sub) > 60:  # min veri eşiği
                                all_data[t] = sub
                        except (KeyError, AttributeError):
                            continue
            else:
                if "Close" in d.columns and len(batch) == 1:
                    sub = d.dropna(subset=["Close"])
                    if len(sub) > 60:
                        all_data[batch[0]] = sub
        except Exception as e:
            debug_msgs.append(f"Batch {i+1}: {type(e).__name__}")
            continue
        progress.progress((i + 1) / n_batches, text=f"Veri indiriliyor... ({i+1}/{n_batches})")
    progress.empty()

    if not all_data:
        st.error("Hiç veri indirilemedi.")
        st.stop()

    # İndikatörleri hazırla
    st.info(f"📊 {len(all_data)} hisse için indikatör + skor matrisi hesaplanıyor...")
    progress2 = st.progress(0.0, text="İndikatörler...")
    prepared = {}
    total = len(all_data)
    for idx, (ticker, df) in enumerate(all_data.items()):
        try:
            prepared[ticker] = prepare_indicators(df)
        except Exception as e:
            debug_msgs.append(f"{ticker}: indikatör hesaplama hatası")
        progress2.progress((idx + 1) / total, text=f"İndikatörler... ({idx+1}/{total})")
    progress2.empty()

    # Simülasyon tarihleri: BIST-100 index'inden son N ay
    if bench.empty:
        st.error("Benchmark indirilemedi.")
        st.stop()

    sim_start = pd.Timestamp(datetime.now() - timedelta(days=period_months * 30))
    trade_dates = bench.index[bench.index >= sim_start]

    if len(trade_dates) < 10:
        st.error(f"Test dönemi için yeterli işlem günü yok ({len(trade_dates)})")
        st.stop()

    # Benchmark getirisi
    bench_in_period = bench[bench.index.isin(trade_dates)]
    bench_return = (bench_in_period["Close"].iloc[-1] / bench_in_period["Close"].iloc[0] - 1) * 100

    # 3 eşik için simülasyon
    st.info(f"🎬 3 eşik için simülasyon çalışıyor ({len(trade_dates)} işlem günü)...")
    progress3 = st.progress(0.0, text="Simülasyon...")
    thresholds = [70, 80, 90]
    sim_results = {}

    for idx, th in enumerate(thresholds):
        sim_results[th] = simulate(prepared, threshold=th, trade_dates=trade_dates)
        progress3.progress((idx + 1) / len(thresholds), text=f"Eşik {th} tamamlandı...")
    progress3.empty()

    st.session_state["atb_results"] = {
        "sim_results": sim_results,
        "bench_return": bench_return,
        "bench_curve": [(d, float(bench.loc[d, "Close"])) for d in trade_dates],
        "period_months": period_months,
        "universe_label": f"Tüm BIST ({len(prepared)} hisse)" if universe_choice == "all" else f"İlk {len(prepared)} hisse",
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_trade_days": len(trade_dates),
    }
    st.success(f"✅ Simülasyon tamamlandı. {len(prepared)} hisse, {len(trade_dates)} işlem günü.")


# === Session_state'ten oku ve göster ===
r = st.session_state["atb_results"]
sim_results = r["sim_results"]
bench_return = r["bench_return"]

st.caption(
    f"📌 Son çalıştırma: {r['run_time']} · {r['universe_label']} · "
    f"{r['period_months']} ay · {r['n_trade_days']} işlem günü"
)

# --- Karşılaştırma tablosu ---
st.divider()
st.subheader("📊 Eşik Karşılaştırma")

summary_rows = []
for th in [70, 80, 90]:
    s = summarize_sim(sim_results[th], bench_return)
    summary_rows.append({
        "Eşik": f"Skor ≥ {th}",
        "Trade": s["n_trades"],
        "Kazanma %": s["win_rate"],
        "Ort. Kazanç %": s["avg_win"],
        "Ort. Kayıp %": s["avg_loss"],
        "Beklenti %": s["expectancy"],
        "En İyi %": s["best"],
        "En Kötü %": s["worst"],
        "Ort. Gün": s["avg_days"],
        "Toplam %": s["total_return"],
        "Alfa %": s["alpha"],
        "Atlanan Sinyal": s["skipped"],
    })

summary_df = pd.DataFrame(summary_rows)


def highlight_best(s, higher_better=True):
    if s.dtype == object:
        return [""] * len(s)
    valid = s.dropna()
    if len(valid) == 0:
        return [""] * len(s)
    target = valid.max() if higher_better else valid.min()
    return ["background-color: #c8e6c9; font-weight: 600" if v == target else "" for v in s]


styled_summary = (
    summary_df.style
    .apply(lambda s: highlight_best(s, True),
           subset=["Kazanma %", "Ort. Kazanç %", "Beklenti %", "En İyi %", "Toplam %", "Alfa %"])
    .apply(lambda s: highlight_best(s, False), subset=["Ort. Kayıp %", "En Kötü %"])
    .format({
        "Kazanma %": lambda x: f"{x:.1f}" if pd.notna(x) else "-",
        "Ort. Kazanç %": lambda x: f"{x:+.2f}" if pd.notna(x) else "-",
        "Ort. Kayıp %": lambda x: f"{x:+.2f}" if pd.notna(x) else "-",
        "Beklenti %": lambda x: f"{x:+.2f}" if pd.notna(x) else "-",
        "En İyi %": lambda x: f"{x:+.2f}" if pd.notna(x) else "-",
        "En Kötü %": lambda x: f"{x:+.2f}" if pd.notna(x) else "-",
        "Ort. Gün": lambda x: f"{x:.0f}" if pd.notna(x) else "-",
        "Toplam %": "{:+.2f}",
        "Alfa %": "{:+.2f}",
    })
)
st.dataframe(styled_summary, use_container_width=True, hide_index=True)
st.caption(f"📈 BIST-100 aynı dönemde: **{bench_return:+.2f}%** · Yeşil = o kolonda en iyi")

# --- Hızlı yorum ---
st.divider()
st.subheader("🧠 Hızlı Yorum")

best_alpha_th = max([70, 80, 90], key=lambda t: summarize_sim(sim_results[t], bench_return)["alpha"])
best_alpha_summary = summarize_sim(sim_results[best_alpha_th], bench_return)

comments = []
if best_alpha_summary["alpha"] > 0:
    comments.append(
        f"🏆 **En iyi eşik: Skor ≥ {best_alpha_th}** — alfa {best_alpha_summary['alpha']:+.2f}% "
        f"({best_alpha_summary['n_trades']} trade, kazanma %{best_alpha_summary['win_rate']:.1f})"
    )
    comments.append("✅ Skor sistemi BIST-100'ü yendi — strateji **alfa üretiyor**.")
else:
    comments.append(
        f"⚠ **En iyi eşik bile alfa negatif:** Skor ≥ {best_alpha_th}, alfa {best_alpha_summary['alpha']:+.2f}%"
    )
    comments.append("❌ Skor sistemi bu dönem BIST-100'ün altında performans verdi. Skor formülü gözden geçirilmeli.")

# Trade sayısı kontrolü
for th in [70, 80, 90]:
    s = summarize_sim(sim_results[th], bench_return)
    if s["n_trades"] < 10:
        comments.append(f"⚠ Eşik {th}: sadece {s['n_trades']} trade — istatistiksel güven düşük, daha uzun dönem test edilmeli")

# Atlanan sinyal kontrolü
for th in [70, 80, 90]:
    s = summarize_sim(sim_results[th], bench_return)
    if s["skipped"] > s["n_trades"] * 0.5 and s["skipped"] > 10:
        comments.append(f"⚠ Eşik {th}: {s['skipped']} sinyal slot dolu olduğu için atlandı — max pozisyon limiti darboğaz")

for c in comments:
    st.markdown(c)


# --- Equity curve karşılaştırma ---
st.divider()
st.subheader("📈 Equity Curve — Portföy Büyümesi")

curve_data = {}
for th in [70, 80, 90]:
    curve = sim_results[th]["equity_curve"]
    if curve:
        initial = sim_results[th]["initial_equity"]
        curve_data[f"Eşik {th}"] = pd.Series(
            [eq / initial * 100 for _, eq in curve],
            index=[d for d, _ in curve]
        )

# Benchmark'ı normalize et
bench_curve = pd.Series(
    {d: c for d, c in r["bench_curve"]}
)
if len(bench_curve) > 0:
    bench_norm = bench_curve / bench_curve.iloc[0] * 100
    curve_data["BIST-100"] = bench_norm

if curve_data:
    chart_df = pd.DataFrame(curve_data)
    st.line_chart(chart_df)
    st.caption("100 = başlangıç sermayesi. Çizginin 100 üstünde olması kâr, altında olması zarar.")


# --- Trade detayı seçilebilir ---
st.divider()
st.subheader("📋 Trade Detayları")

selected_th = st.selectbox(
    "Hangi eşiğin trade'lerini görmek istersin?",
    options=[70, 80, 90],
    format_func=lambda x: f"Skor ≥ {x}",
    index=2,
    key="atb_threshold_selector"
)

trades = sim_results[selected_th]["trades"]
if trades:
    detail_df = pd.DataFrame(trades)
    detail_df = detail_df[["ticker", "entry_date", "entry_price", "entry_score",
                          "exit_date", "exit_price", "pnl_pct", "days"]]
    detail_df.columns = ["Sembol", "Giriş", "Giriş ₺", "Giriş Skoru",
                         "Çıkış", "Çıkış ₺", "P&L %", "Gün"]
    detail_df = detail_df.sort_values("P&L %", ascending=False).reset_index(drop=True)

    def color_pnl(v):
        if pd.isna(v):
            return ""
        if v > 10:
            return "background-color: #a5d6a7; color: #1b5e20; font-weight: 600"
        elif v > 0:
            return "background-color: #e8f5e9"
        elif v > -5:
            return "background-color: #ffebee"
        else:
            return "background-color: #ef9a9a; color: #b71c1c"

    styled_detail = (
        detail_df.style
        .map(color_pnl, subset=["P&L %"])
        .format({
            "Giriş ₺": "₺{:.2f}",
            "Çıkış ₺": "₺{:.2f}",
            "Giriş Skoru": lambda x: f"{x:.0f}" if pd.notna(x) else "-",
            "P&L %": "{:+.2f}",
        })
    )
    st.dataframe(styled_detail, use_container_width=True, hide_index=True, height=500)

    csv = detail_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇ Eşik {selected_th} trade'lerini CSV indir",
        csv,
        file_name=f"aktif_trend_backtest_th{selected_th}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
else:
    st.info(f"Eşik {selected_th} için hiç trade yok.")
