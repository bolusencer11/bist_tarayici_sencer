"""
Aktif Trend Backtest v2 — Saf Strateji Testi

Slot/portföy mantığı KALDIRILDI. Skor sisteminin kendisi test ediliyor.

Kurallar:
- Her gün her hissede: "Yeni Sinyal" + skor ≥ eşik → al (eğer açık değilse)
- Aynı hissede aynı anda 1 pozisyon (re-entry yok, çıkana kadar bekle)
- Çıkış: kategori "Trend Sonu" olunca sat
- Sermaye limiti YOK — strateji "ham" performansı görelim

Birincil metrik: BEKLENTİ % (per trade ortalama)
İkincil metrikler: kazanma oranı, win/loss asimetri, hipotetik eşit-ağırlık portföy
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

st.set_page_config(page_title="Aktif Trend Backtest v2", page_icon="🎯", layout="wide")
st.title("🎯 Aktif Trend Backtest — Saf Strateji Testi")
st.caption("Slot/portföy limiti yok · Skor sisteminin ham gücünü ölç")


# ============================================================
# Veri & İndikatörler
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
    start = end - timedelta(days=days_back + 90)
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
    """Her satır için kategori + skor (vectorize, lookahead bias yok)."""
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

    n = len(df)
    pos = pd.Series(np.arange(n), index=df.index)

    cross_up_pos = pos.where(df["cross_up"]).ffill()
    df["days_since_cross_up"] = (pos - cross_up_pos).astype("float")

    cross_down_pos = pos.where(df["cross_down"]).ffill()
    df["days_since_cross_down"] = (pos - cross_down_pos).astype("float")

    # Kategori (vectorize)
    cat = pd.Series("yok", index=df.index)
    above_mask = df["above"].fillna(False)
    d_up = df["days_since_cross_up"]
    d_down = df["days_since_cross_down"]

    cat[above_mask & (d_up <= 2)] = "yeni"
    cat[above_mask & (d_up > 2) & (d_up <= 10)] = "aktif"
    cat[above_mask & (d_up > 10)] = "olgun"
    cat[~above_mask & (d_down <= 5)] = "son"
    df["category"] = cat

    # Skor (Aktif Trendler ile birebir)
    macd_above = df["MACD"] > df["MACD_signal"]
    macd_hist_growing = df["MACD_hist"] > df["MACD_hist"].shift(1)

    momentum = pd.Series(0.0, index=df.index)
    momentum[macd_above & macd_hist_growing] = 33
    momentum[macd_above & ~macd_hist_growing] = 20

    rsi = df["RSI"]
    health = pd.Series(0.0, index=df.index)
    health[(rsi >= 50) & (rsi <= 65)] = 33
    health[(rsi > 65) & (rsi <= 75)] = 25
    health[(rsi >= 45) & (rsi < 50)] = 15
    health[rsi > 75] = 10

    vol_ratio = (df["vol_ma5"] / df["vol_ma20"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    volume = pd.Series(5.0, index=df.index)
    volume[vol_ratio >= 2.0] = 15
    volume[(vol_ratio >= 1.2) & (vol_ratio < 2.0)] = 34
    volume[(vol_ratio >= 1.0) & (vol_ratio < 1.2)] = 25
    volume[(vol_ratio >= 0.7) & (vol_ratio < 1.0)] = 15

    df["score"] = momentum + health + volume
    df["vol_ratio"] = vol_ratio

    return df


# ============================================================
# Saf Strateji Simülasyonu (slot yok)
# ============================================================

def simulate_pure(prepared_data, threshold, trade_dates):
    """
    Slot/sermaye limiti olmadan saf strateji.
    Her sinyal alınır (aynı hissede tek pozisyon kuralı dışında).
    """
    positions = {}  # {ticker: {entry_date, entry_price, entry_score}}
    closed_trades = []

    for current_date in trade_dates:
        # === 1. Çıkış kontrolü (açık pozisyonlar) ===
        to_close = []
        for ticker, pos in positions.items():
            df = prepared_data.get(ticker)
            if df is None or current_date not in df.index:
                continue
            row = df.loc[current_date]
            if row["category"] == "son":
                to_close.append((ticker, float(row["Close"])))

        for ticker, exit_price in to_close:
            pos = positions[ticker]
            pnl_pct = (exit_price / pos["entry_price"] - 1) * 100
            closed_trades.append({
                "ticker": ticker.replace(".IS", ""),
                "entry_date": pos["entry_date"],
                "exit_date": current_date,
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "days": (current_date - pos["entry_date"]).days,
                "entry_score": pos["entry_score"],
            })
            del positions[ticker]

        # === 2. Yeni sinyal: skoru ≥ eşik olan tüm "yeni" sinyaller ===
        for ticker, df in prepared_data.items():
            if ticker in positions:
                continue  # zaten açık, re-entry yok
            if current_date not in df.index:
                continue
            row = df.loc[current_date]
            if row["category"] == "yeni" and row["score"] >= threshold:
                positions[ticker] = {
                    "entry_date": current_date,
                    "entry_price": float(row["Close"]),
                    "entry_score": float(row["score"]),
                }

    # Simülasyon sonu: açık kalanları son gün fiyatıyla "açık" olarak kapat
    last_date = trade_dates[-1]
    for ticker, pos in positions.items():
        df = prepared_data.get(ticker)
        if df is not None and last_date in df.index:
            exit_price = float(df.loc[last_date, "Close"])
            pnl_pct = (exit_price / pos["entry_price"] - 1) * 100
            closed_trades.append({
                "ticker": ticker.replace(".IS", ""),
                "entry_date": pos["entry_date"],
                "exit_date": last_date,
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "days": (last_date - pos["entry_date"]).days,
                "entry_score": pos["entry_score"],
                "still_open": True,
            })

    return closed_trades


def compute_equal_weight_equity(trades, prepared_data, trade_dates):
    """
    Hipotetik eşit-ağırlık portföy: her gün açık olan tüm trade'lerin
    günlük getirilerinin ORTALAMASI = portföy günlük getirisi. Compound.
    """
    if not trades:
        return pd.Series(100.0, index=trade_dates)

    # Her trade için günlük getiri serisi oluştur
    daily_returns = []
    for t in trades:
        ticker = t["ticker"]
        # ticker'ı .IS ile aramamız lazım
        ticker_key = ticker if ticker.endswith(".IS") else f"{ticker}.IS"
        df = prepared_data.get(ticker_key)
        if df is None:
            continue

        entry = t["entry_date"]
        exit = t["exit_date"]
        mask = (df.index >= entry) & (df.index <= exit)
        prices = df.loc[mask, "Close"]
        if len(prices) < 2:
            continue
        rets = prices.pct_change().fillna(0)
        daily_returns.append(rets)

    if not daily_returns:
        return pd.Series(100.0, index=trade_dates)

    # Her trade'in günlük getirisini bir matrise yerleştir
    returns_df = pd.concat(daily_returns, axis=1)
    returns_df = returns_df.reindex(trade_dates)

    # Her gün için: o gün açık olan trade'lerin günlük getirisi ortalaması
    portfolio_daily = returns_df.mean(axis=1, skipna=True).fillna(0)
    equity = (1 + portfolio_daily).cumprod() * 100

    return equity


def summarize(trades, eq_curve_final, bench_return_pct):
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
            "portfolio_return": 0.0,
            "alpha": -bench_return_pct,
            "win_loss_ratio": np.nan,
        }
    pnls = np.array([t["pnl_pct"] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    avg_win = wins.mean() if len(wins) else 0
    avg_loss = losses.mean() if len(losses) else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
    return {
        "n_trades": len(trades),
        "win_rate": len(wins) / len(pnls) * 100,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": pnls.mean(),
        "best": pnls.max(),
        "worst": pnls.min(),
        "avg_days": np.mean([t["days"] for t in trades]),
        "portfolio_return": eq_curve_final - 100,  # 100 başlangıçtan ne kadar arttı
        "alpha": (eq_curve_final - 100) - bench_return_pct,
        "win_loss_ratio": win_loss_ratio,
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
        format_func=lambda x: "Tüm BIST (~500 hisse)" if x == "all" else "İlk 100 hisse"
    )

    st.divider()
    st.caption(
        "**Test mantığı (v2):**\n\n"
        "- Skoru ≥ eşik olan HER yeni sinyal alınır (slot limiti yok)\n"
        "- Aynı hissede aynı anda 1 pozisyon\n"
        "- 'Trend Sonu' kategorisinde sat\n"
        "- Eşikler: **70 / 80 / 90** karşılaştırmalı\n\n"
        "**Birincil metrik:** Beklenti % (trade başına ortalama)\n\n"
        "**Hipotetik portföy:** Her gün açık trade'lerin eşit-ağırlık ortalaması (BIST-100 ile kıyas için)"
    )

    run = st.button("▶ Backtest'i Çalıştır", type="primary", use_container_width=True)

    if "atb2_results" in st.session_state:
        if st.button("🗑 Sonuçları Temizle", use_container_width=True):
            del st.session_state["atb2_results"]
            st.rerun()


# ============================================================
# Ana akış
# ============================================================

if not run and "atb2_results" not in st.session_state:
    st.info("Sol menüden parametreleri seçip **▶ Backtest'i Çalıştır** butonuna bas.")
    st.markdown("""
    ### v1'den ne değişti?

    v1'de **slot mantığı testi gizlemişti**: max 10 açık pozisyon limitine takıldığı için 11.000+ sinyal atlandı, 3 eşik aynı sonuç verdi.

    v2'de **slot kalktı**. Şimdi her sinyal alınıyor, sadece **aynı hissede tek pozisyon** kuralı var. Bu, **skor sisteminin ham kalitesini** ölçer.

    ### Hangi soruya cevap arıyoruz?

    > "Aktif Trendler skoru gerçekten **iyi hisseleri seçebiliyor mu**?"

    **Birincil cevap: Beklenti %** — trade başına ortalama getiri.
    - Pozitif ve **büyük** ise → skor iyiyi seçiyor
    - Pozitif ama küçük (≤ %1) → marjinal, komisyonu karşılamaz
    - Negatif → skor sistemi gürültü

    ### Karşılaştırma mantığı

    | Eşik | Beklenen |
    |------|---|
    | **≥ 70** | Çok trade, orta kalite |
    | **≥ 80** | Daha az ama daha kaliteli |
    | **≥ 90** | En az trade, en yüksek kalite — beklenti en yüksek olmalı |

    **Eğer skor sistemi anlamlıysa**, eşik arttıkça beklenti yükselmeli (sayı düşse bile).
    **Eğer skor anlamsızsa**, üç eşik benzer beklenti verir — yüksek skor = "iyi hisse" değil.
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

    st.info(f"📥 {len(tickers)} hisse · son {period_months} ay hazırlanıyor...")

    bench = fetch_benchmark(days_back)

    # Veri çek
    all_data = {}
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
                            if len(sub) > 60:
                                all_data[t] = sub
                        except (KeyError, AttributeError):
                            continue
            else:
                if "Close" in d.columns and len(batch) == 1:
                    sub = d.dropna(subset=["Close"])
                    if len(sub) > 60:
                        all_data[batch[0]] = sub
        except Exception:
            continue
        progress.progress((i + 1) / n_batches, text=f"Veri... ({i+1}/{n_batches})")
    progress.empty()

    if not all_data:
        st.error("Veri indirilemedi.")
        st.stop()

    # İndikatörler
    st.info(f"📊 {len(all_data)} hisse için indikatör hesaplanıyor...")
    progress2 = st.progress(0.0, text="İndikatörler...")
    prepared = {}
    total = len(all_data)
    for idx, (ticker, df) in enumerate(all_data.items()):
        try:
            prepared[ticker] = prepare_indicators(df)
        except Exception:
            pass
        progress2.progress((idx + 1) / total, text=f"İndikatörler... ({idx+1}/{total})")
    progress2.empty()

    # Simülasyon tarihleri
    if bench.empty:
        st.error("Benchmark indirilemedi.")
        st.stop()

    sim_start = pd.Timestamp(datetime.now() - timedelta(days=period_months * 30))
    trade_dates = bench.index[bench.index >= sim_start]

    if len(trade_dates) < 10:
        st.error(f"Yetersiz işlem günü ({len(trade_dates)})")
        st.stop()

    bench_in_period = bench[bench.index.isin(trade_dates)]
    bench_return = (bench_in_period["Close"].iloc[-1] / bench_in_period["Close"].iloc[0] - 1) * 100
    bench_norm = bench_in_period["Close"] / bench_in_period["Close"].iloc[0] * 100

    # 3 eşik için simülasyon
    st.info(f"🎬 3 eşik için simülasyon ({len(trade_dates)} işlem günü)...")
    progress3 = st.progress(0.0, text="Simülasyon...")
    thresholds = [70, 80, 90]
    results = {}

    for idx, th in enumerate(thresholds):
        trades = simulate_pure(prepared, threshold=th, trade_dates=trade_dates)
        equity = compute_equal_weight_equity(trades, prepared, trade_dates)
        results[th] = {
            "trades": trades,
            "equity": equity,
        }
        progress3.progress((idx + 1) / len(thresholds), text=f"Eşik {th} bitti")
    progress3.empty()

    st.session_state["atb2_results"] = {
        "results": results,
        "bench_return": bench_return,
        "bench_curve": bench_norm,
        "period_months": period_months,
        "universe_label": f"Tüm BIST ({len(prepared)} hisse)" if universe_choice == "all" else f"İlk {len(prepared)} hisse",
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_trade_days": len(trade_dates),
    }
    st.success(f"✅ Tamamlandı. {len(prepared)} hisse, {len(trade_dates)} işlem günü.")


# === Görüntüleme ===
r = st.session_state["atb2_results"]
results = r["results"]
bench_return = r["bench_return"]

st.caption(
    f"📌 Son çalıştırma: {r['run_time']} · {r['universe_label']} · "
    f"{r['period_months']} ay · {r['n_trade_days']} işlem günü"
)

# --- Ana tablo ---
st.divider()
st.subheader("📊 Eşik Karşılaştırma")

summary_rows = []
for th in [70, 80, 90]:
    trades = results[th]["trades"]
    eq_final = float(results[th]["equity"].iloc[-1]) if len(results[th]["equity"]) > 0 else 100.0
    s = summarize(trades, eq_final, bench_return)
    summary_rows.append({
        "Eşik": f"Skor ≥ {th}",
        "Trade": s["n_trades"],
        "Kazanma %": s["win_rate"],
        "Ort. Kazanç %": s["avg_win"],
        "Ort. Kayıp %": s["avg_loss"],
        "K/Z Oranı": s["win_loss_ratio"],
        "Beklenti %": s["expectancy"],
        "En İyi %": s["best"],
        "En Kötü %": s["worst"],
        "Ort. Gün": s["avg_days"],
        "Portföy %": s["portfolio_return"],
        "Alfa %": s["alpha"],
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


styled = (
    summary_df.style
    .apply(lambda s: highlight_best(s, True),
           subset=["Kazanma %", "Ort. Kazanç %", "K/Z Oranı", "Beklenti %", "En İyi %", "Portföy %", "Alfa %"])
    .apply(lambda s: highlight_best(s, False), subset=["Ort. Kayıp %", "En Kötü %"])
    .format({
        "Kazanma %": lambda x: f"{x:.1f}" if pd.notna(x) else "-",
        "Ort. Kazanç %": lambda x: f"{x:+.2f}" if pd.notna(x) else "-",
        "Ort. Kayıp %": lambda x: f"{x:+.2f}" if pd.notna(x) else "-",
        "K/Z Oranı": lambda x: f"{x:.2f}" if pd.notna(x) else "-",
        "Beklenti %": lambda x: f"{x:+.2f}" if pd.notna(x) else "-",
        "En İyi %": lambda x: f"{x:+.2f}" if pd.notna(x) else "-",
        "En Kötü %": lambda x: f"{x:+.2f}" if pd.notna(x) else "-",
        "Ort. Gün": lambda x: f"{x:.0f}" if pd.notna(x) else "-",
        "Portföy %": "{:+.2f}",
        "Alfa %": "{:+.2f}",
    })
)
st.dataframe(styled, use_container_width=True, hide_index=True)
st.caption(f"📈 BIST-100 aynı dönemde: **{bench_return:+.2f}%** · Yeşil = o kolonda en iyi")


# --- Yorum ---
st.divider()
st.subheader("🧠 Hızlı Yorum")

comments = []

# Eşik trend kontrolü
exps = {th: summarize(results[th]["trades"],
                      float(results[th]["equity"].iloc[-1]) if len(results[th]["equity"]) > 0 else 100,
                      bench_return)["expectancy"] for th in [70, 80, 90]}

if all(pd.notna(v) for v in exps.values()):
    if exps[90] > exps[80] > exps[70]:
        comments.append("✅ **Beklenti eşikle birlikte artıyor** — skor sistemi anlamlı, yüksek skor = iyi hisse")
    elif exps[70] > exps[80] > exps[90]:
        comments.append("⚠ **Beklenti eşikle ters orantılı** — yüksek skor 'kötü' anlamına geliyor olabilir, formül ters çalışıyor olabilir")
    else:
        comments.append("⚠ Beklenti eşikle düzgün artmıyor — skor sistemi gürültülü, monoton sıralama yok")

# En iyi eşik
best_exp_th = max([70, 80, 90], key=lambda t: exps[t] if pd.notna(exps[t]) else -999)
if pd.notna(exps[best_exp_th]):
    if exps[best_exp_th] > 5:
        comments.append(f"🏆 En yüksek beklenti: **Skor ≥ {best_exp_th}** = %{exps[best_exp_th]:+.2f} per trade — güçlü")
    elif exps[best_exp_th] > 1:
        comments.append(f"🥉 En yüksek beklenti: **Skor ≥ {best_exp_th}** = %{exps[best_exp_th]:+.2f} per trade — marjinal")
    elif exps[best_exp_th] > 0:
        comments.append(f"⚠ En yüksek beklenti: **Skor ≥ {best_exp_th}** = %{exps[best_exp_th]:+.2f} per trade — komisyonu karşılamaz")
    else:
        comments.append(f"❌ Tüm eşiklerde beklenti negatif — skor sistemi para kaybettiriyor")

# Trade sayısı
for th in [70, 80, 90]:
    s = summarize(results[th]["trades"],
                  float(results[th]["equity"].iloc[-1]) if len(results[th]["equity"]) > 0 else 100,
                  bench_return)
    if s["n_trades"] < 20:
        comments.append(f"⚠ Eşik {th}: sadece {s['n_trades']} trade — istatistik güveni düşük, daha uzun dönem öner")

# Portföy
for th in [70, 80, 90]:
    s = summarize(results[th]["trades"],
                  float(results[th]["equity"].iloc[-1]) if len(results[th]["equity"]) > 0 else 100,
                  bench_return)
    if s["alpha"] > 0 and s["n_trades"] >= 20:
        comments.append(f"📈 Eşik {th}: hipotetik eşit-ağırlık portföy BIST-100'ü yendi (alfa {s['alpha']:+.2f}%)")

for c in comments:
    st.markdown(c)


# --- Equity curve ---
st.divider()
st.subheader("📈 Hipotetik Eşit-Ağırlık Portföy Eğrisi")

curve_data = {}
for th in [70, 80, 90]:
    eq = results[th]["equity"]
    if len(eq) > 0:
        curve_data[f"Eşik {th}"] = eq

if "bench_curve" in r and len(r["bench_curve"]) > 0:
    curve_data["BIST-100"] = r["bench_curve"]

if curve_data:
    chart_df = pd.DataFrame(curve_data)
    st.line_chart(chart_df)
    st.caption("100 = başlangıç. Her gün açık olan tüm trade'lerin eşit-ağırlık ortalama getirisi. Slot/sermaye limiti yok.")


# --- Trade detayı ---
st.divider()
st.subheader("📋 Trade Detayları")

selected_th = st.selectbox(
    "Hangi eşik?",
    options=[70, 80, 90],
    format_func=lambda x: f"Skor ≥ {x}",
    index=2,
    key="atb2_threshold"
)

trades = results[selected_th]["trades"]
if trades:
    detail_df = pd.DataFrame(trades)
    detail_df["entry_date"] = pd.to_datetime(detail_df["entry_date"]).dt.strftime("%Y-%m-%d")
    detail_df["exit_date"] = pd.to_datetime(detail_df["exit_date"]).dt.strftime("%Y-%m-%d")
    cols = ["ticker", "entry_date", "entry_price", "entry_score", "exit_date", "exit_price", "pnl_pct", "days"]
    detail_df = detail_df[[c for c in cols if c in detail_df.columns]]
    detail_df.columns = ["Sembol", "Giriş", "Giriş ₺", "Giriş Skoru", "Çıkış", "Çıkış ₺", "P&L %", "Gün"]
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

    styled_d = (
        detail_df.style
        .map(color_pnl, subset=["P&L %"])
        .format({
            "Giriş ₺": "₺{:.2f}",
            "Çıkış ₺": "₺{:.2f}",
            "Giriş Skoru": lambda x: f"{x:.0f}" if pd.notna(x) else "-",
            "P&L %": "{:+.2f}",
        })
    )
    st.dataframe(styled_d, use_container_width=True, hide_index=True, height=500)

    csv = detail_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇ Eşik {selected_th} trade'leri CSV",
        csv,
        file_name=f"atb2_th{selected_th}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
else:
    st.info(f"Eşik {selected_th} için trade yok.")
