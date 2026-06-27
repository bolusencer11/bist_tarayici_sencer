"""
BIST Stratejisi Backtest
EMA10↑EMA30 + haftalık değişim < %15 stratejisinin geçmiş performansı.

Not: F/K ve PD/DD filtreleri tarihsel veri olmadığı için backtest'e dahil değildir.
Sadece teknik sinyalin performansı ölçülür.
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path

st.set_page_config(page_title="BIST Backtest", page_icon="📊", layout="wide")
st.title("📊 BIST Stratejisi Backtest")
st.caption("EMA10↑EMA30 crossover + haftalık değişim < %15 stratejisinin geçmiş performansı")


# ---------- Ticker listesi ----------
def load_tickers():
    """Önce bist_tickers.txt dosyasını okumayı dene, yoksa gömülü listeyi kullan."""
    p = Path(__file__).parent.parent / "bist_tickers.txt"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return DEFAULT_TICKERS


DEFAULT_TICKERS = [
    # Fallback - bist_tickers.txt yoksa kullanılır
    "AKBNK.IS", "GARAN.IS", "ISCTR.IS", "YKBNK.IS", "HALKB.IS", "VAKBN.IS",
    "SAHOL.IS", "KCHOL.IS", "EREGL.IS", "KRDMD.IS", "FROTO.IS", "TOASO.IS",
    "THYAO.IS", "PGSUS.IS", "TUPRS.IS", "BIMAS.IS", "MGROS.IS", "ASELS.IS",
    "TCELL.IS", "TTKOM.IS", "SISE.IS", "PETKM.IS", "ARCLK.IS", "EKGYO.IS",
    "KOZAL.IS", "KOZAA.IS", "HEKTS.IS", "SASA.IS", "ULKER.IS"
]


# ---------- Sidebar parametreler ----------
with st.sidebar:
    st.header("⚙ Parametreler")
    years_back = st.slider("Geriye gidilecek yıl", 1, 5, 3)
    hold_days = st.number_input("Tutma süresi (iş günü)", 1, 30, 5)

    st.divider()
    st.subheader("Strateji varyasyonları")
    use_trend_filter = st.checkbox(
        "Fiyat > EMA200 ek filtresi",
        value=False,
        help="Sadece uzun vadeli yükseliş trendindeki hisselerde sinyal kabul et"
    )
    use_volume_filter = st.checkbox(
        "Min. günlük hacim 1M TL filtresi",
        value=False,
        help="İlikit hisseleri ele"
    )
    rsi_filter = st.checkbox(
        "RSI 40-65 arası ek filtresi",
        value=False,
        help="Aşırı satım dipinden değil, sağlıklı momentum başlangıcından al"
    )

    st.divider()
    run = st.button("▶ Backtest Çalıştır", type="primary", use_container_width=True)


# ---------- Yardımcı fonksiyonlar ----------
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_batch(tickers_tuple, start, end):
    """Tek seferde toplu indirme. Cache 24 saat."""
    data = yf.download(
        list(tickers_tuple),
        start=start,
        end=end,
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
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_signals(df, trend_filter=False, volume_thresh=0, rsi_band=False):
    df = df.copy()
    df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA30"] = df["Close"].ewm(span=30, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["weekly_change"] = df["Close"].pct_change(5) * 100
    df["RSI"] = compute_rsi(df["Close"])

    # Ana koşul: EMA10 EMA30'u yukarı kesti (önceki gün altta, bugün üstte)
    crossover = (df["EMA10"] > df["EMA30"]) & (df["EMA10"].shift(1) <= df["EMA30"].shift(1))
    signal = crossover & (df["weekly_change"] < 15)

    if trend_filter:
        signal = signal & (df["Close"] > df["EMA200"])
    if volume_thresh > 0:
        df["vol_tl"] = df["Close"] * df["Volume"]
        df["vol_ma20"] = df["vol_tl"].rolling(20).mean()
        signal = signal & (df["vol_ma20"] > volume_thresh)
    if rsi_band:
        signal = signal & df["RSI"].between(40, 65)

    df["signal"] = signal
    return df


def simulate_trades(df, ticker, hold_days=5):
    trades = []
    sig_dates = df.index[df["signal"]]
    for sig_date in sig_dates:
        idx = df.index.get_loc(sig_date)
        if idx + 1 + hold_days >= len(df):
            continue
        buy_price = df["Open"].iloc[idx + 1]
        sell_price = df["Close"].iloc[idx + 1 + hold_days]
        if pd.isna(buy_price) or pd.isna(sell_price) or buy_price <= 0:
            continue
        ret = (sell_price - buy_price) / buy_price * 100
        trades.append({
            "ticker": ticker.replace(".IS", ""),
            "signal_date": sig_date,
            "buy_date": df.index[idx + 1],
            "sell_date": df.index[idx + 1 + hold_days],
            "buy_price": round(buy_price, 2),
            "sell_price": round(sell_price, 2),
            "return_pct": round(ret, 2),
            "weekly_change_at_signal": round(df["weekly_change"].iloc[idx], 2),
            "rsi_at_signal": round(df["RSI"].iloc[idx], 1) if not pd.isna(df["RSI"].iloc[idx]) else None,
        })
    return trades


# ---------- Backtest çalıştır ----------
if run:
    tickers = load_tickers()
    st.info(f"📥 {len(tickers)} hisse için {years_back} yıllık veri indiriliyor...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365 + 60)  # EMA200 ısınma payı

    # Toplu indirme - batch'lere böl
    all_data = {}
    batch_size = 50
    n_batches = (len(tickers) + batch_size - 1) // batch_size
    progress = st.progress(0.0, text="Veri indiriliyor...")

    for i in range(n_batches):
        batch = tickers[i * batch_size : (i + 1) * batch_size]
        try:
            d = fetch_batch(tuple(batch), start_date, end_date)
            if len(batch) == 1:
                # Tek ticker farklı yapıda gelir
                t = batch[0]
                if not d.empty and "Close" in d.columns:
                    sub = d.dropna(subset=["Close"])
                    if len(sub) > 220:
                        all_data[t] = sub
            else:
                for t in batch:
                    try:
                        if t in d.columns.get_level_values(0):
                            sub = d[t].dropna(subset=["Close"])
                            if len(sub) > 220:  # EMA200 + 20 minimum
                                all_data[t] = sub
                    except (KeyError, AttributeError):
                        continue
        except Exception as e:
            st.warning(f"Batch {i+1} hatası: {e}")
        progress.progress((i + 1) / n_batches, text=f"Veri indiriliyor... ({i+1}/{n_batches} batch)")

    progress.empty()
    st.success(f"✅ {len(all_data)} hissenin verisi indirildi (yetersiz verili olanlar atlandı)")

    if len(all_data) == 0:
        st.error("Hiç veri indirilemedi. İnternet bağlantısını veya ticker listesini kontrol et.")
        st.stop()

    # Sinyal ve trade hesaplama
    progress = st.progress(0.0, text="Sinyaller hesaplanıyor...")
    all_trades = []

    for i, (ticker, df) in enumerate(all_data.items()):
        try:
            df_s = compute_signals(
                df,
                trend_filter=use_trend_filter,
                volume_thresh=1_000_000 if use_volume_filter else 0,
                rsi_band=rsi_filter,
            )
            trades = simulate_trades(df_s, ticker, hold_days=hold_days)
            all_trades.extend(trades)
        except Exception:
            continue
        if (i + 1) % 20 == 0:
            progress.progress((i + 1) / len(all_data), text=f"Sinyaller... ({i+1}/{len(all_data)})")

    progress.empty()

    if not all_trades:
        st.warning("Bu filtrelerle hiç sinyal üretilmedi. Filtreleri gevşet.")
        st.stop()

    trades_df = pd.DataFrame(all_trades)

    # Benchmark - XU100 buy & hold
    try:
        bench = yf.download("XU100.IS", start=start_date, end=end_date, progress=False, auto_adjust=True)
        if not bench.empty:
            bench_return = float((bench["Close"].iloc[-1] / bench["Close"].iloc[0] - 1) * 100)
            # Yıllık ortalama (basit)
            bench_cagr = ((1 + bench_return / 100) ** (1 / years_back) - 1) * 100
        else:
            bench_return, bench_cagr = None, None
    except Exception:
        bench_return, bench_cagr = None, None

    # ---------- Sonuçlar ----------
    st.divider()
    st.header("📈 Sonuçlar")

    # Üst satır - özet metrikler
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam sinyal", f"{len(trades_df):,}")
    c2.metric(f"Ort. getiri / trade ({hold_days}g)", f"{trades_df['return_pct'].mean():.2f}%")
    c3.metric("Medyan getiri / trade", f"{trades_df['return_pct'].median():.2f}%")
    c4.metric("Kazanma oranı", f"{(trades_df['return_pct'] > 0).mean() * 100:.1f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("En iyi trade", f"{trades_df['return_pct'].max():.2f}%")
    c6.metric("En kötü trade", f"{trades_df['return_pct'].min():.2f}%")
    c7.metric("Std sapma", f"{trades_df['return_pct'].std():.2f}%")
    sharpe_like = trades_df["return_pct"].mean() / trades_df["return_pct"].std() if trades_df["return_pct"].std() > 0 else 0
    c8.metric("Risk-ayarlı (μ/σ)", f"{sharpe_like:.3f}")

    # Benchmark satırı
    if bench_return is not None:
        st.subheader("📊 Benchmark karşılaştırması")
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric(f"XU100 {years_back}y toplam getiri", f"{bench_return:.1f}%")
        bc2.metric("XU100 yıllık (CAGR)", f"{bench_cagr:.1f}%")
        bc3.metric(f"XU100 ort. {hold_days}g getiri",
                   f"{((1 + bench_cagr/100) ** (hold_days/252) - 1) * 100:.2f}%",
                   help="Yıllık getiriden tutma süresine pro-rata")

        avg_trade = trades_df["return_pct"].mean()
        bench_per_period = ((1 + bench_cagr / 100) ** (hold_days / 252) - 1) * 100
        edge = avg_trade - bench_per_period
        if edge > 0:
            st.success(f"✅ Strateji, XU100'ün {hold_days} günlük pro-rata getirisinden ortalama **{edge:.2f} puan** daha iyi.")
        else:
            st.error(f"❌ Strateji, XU100'ün {hold_days} günlük pro-rata getirisinden ortalama **{abs(edge):.2f} puan** daha kötü. Yani benchmark'tan zayıf.")

    # Dağılım
    st.subheader("Getiri dağılımı")
    fig = px.histogram(
        trades_df, x="return_pct", nbins=80,
        labels={"return_pct": f"{hold_days} günlük getiri (%)"},
        color_discrete_sequence=["#4C9AFF"]
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.add_vline(
        x=trades_df["return_pct"].mean(),
        line_dash="dash", line_color="green",
        annotation_text=f"Ort: {trades_df['return_pct'].mean():.2f}%",
        annotation_position="top right",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Sinyal anındaki haftalık değişime göre breakdown
    st.subheader("Sinyal anındaki haftalık değişime göre getiri")
    trades_df["wc_bucket"] = pd.cut(
        trades_df["weekly_change_at_signal"],
        bins=[-100, -10, -5, 0, 5, 10, 15],
        labels=["<-10%", "-10 to -5%", "-5 to 0%", "0 to 5%", "5 to 10%", "10 to 15%"],
    )
    bucket_stats = (
        trades_df.groupby("wc_bucket", observed=True)["return_pct"]
        .agg(["mean", "median", "count"])
        .round(2)
    )
    bucket_stats.columns = ["Ort. getiri %", "Medyan getiri %", "Trade sayısı"]
    st.dataframe(bucket_stats, use_container_width=True)
    st.caption("💡 Hangi haftalık değişim aralığında strateji daha iyi/kötü çalışıyor görmek için.")

    # RSI bucket
    if trades_df["rsi_at_signal"].notna().sum() > 50:
        st.subheader("Sinyal anındaki RSI'ya göre getiri")
        trades_df["rsi_bucket"] = pd.cut(
            trades_df["rsi_at_signal"],
            bins=[0, 30, 40, 50, 60, 70, 100],
            labels=["<30", "30-40", "40-50", "50-60", "60-70", ">70"],
        )
        rsi_stats = (
            trades_df.groupby("rsi_bucket", observed=True)["return_pct"]
            .agg(["mean", "median", "count"])
            .round(2)
        )
        rsi_stats.columns = ["Ort. getiri %", "Medyan getiri %", "Trade sayısı"]
        st.dataframe(rsi_stats, use_container_width=True)

    # Hisse bazlı top/bottom
    ticker_stats = trades_df.groupby("ticker")["return_pct"].agg(["mean", "count"]).round(2)
    ticker_stats = ticker_stats[ticker_stats["count"] >= 3].sort_values("mean", ascending=False)
    ticker_stats.columns = ["Ort. getiri %", "Trade sayısı"]

    st.subheader("Hisse bazlı performans (min 3 trade)")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**🟢 En iyi 15 hisse**")
        st.dataframe(ticker_stats.head(15), use_container_width=True)
    with col_b:
        st.write("**🔴 En kötü 15 hisse**")
        st.dataframe(ticker_stats.tail(15).iloc[::-1], use_container_width=True)

    # Yıllık dağılım
    trades_df["year"] = pd.to_datetime(trades_df["buy_date"]).dt.year
    yearly = trades_df.groupby("year")["return_pct"].agg(["mean", "count", lambda x: (x > 0).mean() * 100]).round(2)
    yearly.columns = ["Ort. getiri %", "Sinyal sayısı", "Kazanma oranı %"]
    st.subheader("Yıllık breakdown")
    st.dataframe(yearly, use_container_width=True)

    # Equity curve (eşit ağırlık, sıralı)
    st.subheader("Eşit-ağırlıklı sıralı trade equity eğrisi")
    st.caption("⚠ Teorik: her trade'i sırayla, eşit sermayeyle aldığını varsayar. Aynı anda birden fazla pozisyon tutulamayacağı için gerçek hayatta bu eğri ulaşılmaz, sadece referans.")
    trades_sorted = trades_df.sort_values("buy_date").reset_index(drop=True)
    trades_sorted["cum_pct"] = (1 + trades_sorted["return_pct"] / 100).cumprod() * 100 - 100
    fig_eq = px.line(
        trades_sorted, x="buy_date", y="cum_pct",
        labels={"buy_date": "Tarih", "cum_pct": "Kümülatif getiri (%)"},
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # CSV indirme
    st.subheader("📥 Ham veri")
    csv = trades_df.drop(columns=["wc_bucket", "rsi_bucket", "year"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button("Tüm trade'leri CSV olarak indir", csv, "backtest_trades.csv", "text/csv")
    with st.expander("Trade örneği (ilk 50, getiri sıralı)"):
        st.dataframe(
            trades_df.sort_values("return_pct", ascending=False).head(50).drop(
                columns=["wc_bucket", "rsi_bucket", "year"], errors="ignore"
            ),
            use_container_width=True,
            hide_index=True,
        )

else:
    st.info("Sol menüden parametreleri ayarlayıp **▶ Backtest Çalıştır** butonuna bas.")
    st.markdown("""
    ### Bu backtest ne ölçüyor
    - **Sinyal:** Senin tarayıcındaki ile aynı — EMA10'un EMA30'u yukarı kesmesi + son 5 günlük değişim < %15
    - **İşlem:** Sinyal günü kapanışında tespit, ertesi gün **açılışta alım**, N gün sonra **kapanışta satım**
    - **Veri:** yfinance üzerinden günlük OHLCV, bedelsiz/temettü düzeltmeli

    ### Önemli kısıtlar
    - **F/K ve PD/DD filtreleri dahil DEĞİL** — bu çarpanların geçmiş snapshot'larına ücretsiz erişim yok
    - Komisyon ve slipaj hariç (gerçek hayatta her trade'den ~%0.2-0.3 düş)
    - "Aynı anda çoklu pozisyon" sınırı yok — her sinyal bağımsız değerlendiriliyor
    - Hayatta kalma yanlılığı (survivorship bias) olabilir — borsadan çıkarılan hisseler listede yok

    ### Sol menüdeki ek filtreler
    Senin orijinal script'inde olmayan ama 1 günlük performans analizimizde öneri olarak çıkan üç filtre:
    - **EMA200 trend filtresi** — sadece uzun vadeli yükseliş trendindeki hisseler
    - **Hacim filtresi** — ilikit hisseleri eler
    - **RSI 40-65 bandı** — aşırı satım dipinden değil, sağlıklı momentumdan al

    Bunların etkisini ham strateji ile karşılaştırarak görebilirsin.
    """)
