import streamlit as st
import json
import requests
import base64
from tradingview_screener import Query, col
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="BIST USTEL1030 Tarayıcı (mburha YTD)", page_icon="📈", layout="wide")
st.title("📈 BIST USTEL1030 Tarayıcı (mburha YTD)")
st.caption("EMA10↑EMA30 · F/K ≤ 50 · PD/DD ≤ 25 · Haftalık Değişim < %15")


def save_to_github(df):
    token = st.secrets["GITHUB_TOKEN"]
    repo = st.secrets["GITHUB_REPO"]
    path = "results/bist_results.json"
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    content = {
        "tarih": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "hisseler": df.to_dict(orient='records')
    }
    headers = {"Authorization": f"token {token}"}
    r = requests.get(url, headers=headers)
    sha = r.json().get("sha") if r.status_code == 200 else None
    encoded = base64.b64encode(
        json.dumps(content, ensure_ascii=False, indent=2).encode()
    ).decode()
    payload = {
        "message": f"Update {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "content": encoded
    }
    if sha:
        payload["sha"] = sha
    requests.put(url, headers=headers, json=payload)


def tavsiye_uret(rsi, macd, macd_signal, ema10, ema50, ema200, fiyat):
    puanlar = 0
    yorumlar = []

    # RSI değerlendirmesi
    if rsi is None:
        rsi_durum = "Veri yok"
    elif rsi < 30:
        rsi_durum = f"{rsi:.0f} — Aşırı Satım"
        puanlar += 2
        yorumlar.append("RSI aşırı satışlarda")
    elif rsi > 70:
        rsi_durum = f"{rsi:.0f} — Aşırı Alım"
        puanlar -= 2
        yorumlar.append("RSI aşırı alımda")
    else:
        rsi_durum = f"{rsi:.0f} — Nötr"

    # MACD değerlendirmesi
    if macd is None or macd_signal is None:
        macd_durum = "Veri yok"
    elif macd > macd_signal:
        macd_durum = "Pozitif ✅"
        puanlar += 1
        yorumlar.append("MACD pozitif")
    else:
        macd_durum = "Negatif ❌"
        puanlar -= 1
        yorumlar.append("MACD negatif")

    # EMA değerlendirmesi
    ema_satirlar = []
    if ema10 and fiyat:
        ema_satirlar.append("EMA10 ✅" if fiyat > ema10 else "EMA10 ❌")
        if fiyat > ema10:
            puanlar += 1
    if ema50 and fiyat:
        ema_satirlar.append("EMA50 ✅" if fiyat > ema50 else "EMA50 ❌")
        if fiyat > ema50:
            puanlar += 1
    if ema200 and fiyat:
        ema_satirlar.append("EMA200 ✅" if fiyat > ema200 else "EMA200 ❌")
        if fiyat > ema200:
            puanlar += 1
    ema_durum = " / ".join(ema_satirlar) if ema_satirlar else "Veri yok"

    # Tavsiye
    if puanlar >= 3:
        tavsiye = "AL ✅"
    elif puanlar >= 1:
        tavsiye = "İZLE 🟡"
    else:
        tavsiye = "UZAK DUR ❌"

    yorum = ", ".join(yorumlar) if yorumlar else "Nötr görünüm"
    return rsi_durum, macd_durum, ema_durum, tavsiye, yorum


# Session state başlangıç
if "df_result" not in st.session_state:
    st.session_state.df_result = None
if "count_result" not in st.session_state:
    st.session_state.count_result = 0
if "df_technical" not in st.session_state:
    st.session_state.df_technical = None


if st.button("▶ Tara", type="primary"):
    with st.spinner("Taranıyor..."):
        try:
            count, df = (
                Query()
                .set_markets('turkey')
                .select(
                    'name', 'close', 'change|1W',
                    'price_earnings_ttm', 'price_book_ratio',
                    'RSI', 'MACD.macd', 'MACD.signal',
                    'EMA10', 'EMA50', 'EMA200'
                )
                .where(
                    col('EMA10').crosses_above(col('EMA30')),
                    col('change|1W') < 15,
                    col('price_earnings_ttm').between(0, 50),
                    col('price_book_ratio').between(0, 25),
                )
                .limit(200)
                .get_scanner_data()
            )

            df = df.rename(columns={
                'name': 'Sembol',
                'close': 'Fiyat',
                'change|1W': 'Haftalık %',
                'price_earnings_ttm': 'F/K',
                'price_book_ratio': 'PD/DD',
                'RSI': 'RSI',
                'MACD.macd': 'MACD',
                'MACD.signal': 'MACD_Signal',
                'EMA10': 'EMA10',
                'EMA50': 'EMA50',
                'EMA200': 'EMA200',
            })

            df = df.reset_index(drop=True)
            df['Score'] = 0
            n = len(df)
            if n >= 5:
                top_n = 5
            elif n >= 3:
                top_n = 3
            elif n >= 2:
                top_n = 2
            elif n >= 1:
                top_n = 1
            else:
                top_n = 0

            if top_n > 0:
                df.loc[df['Haftalık %'].dropna().nsmallest(top_n).index, 'Score'] += 3
                df.loc[df['F/K'].dropna().nsmallest(top_n).index, 'Score'] += 4
                df.loc[df['PD/DD'].dropna().nsmallest(top_n).index, 'Score'] += 3

            df = df.sort_values('Score', ascending=False).reset_index(drop=True)
            st.session_state.df_result = df
            st.session_state.count_result = count

            top_df = df.head(top_n).copy()
            teknik_rows = []
            for _, row in top_df.iterrows():
                rsi_d, macd_d, ema_d, tav, yorum = tavsiye_uret(
                    row.get('RSI'), row.get('MACD'), row.get('MACD_Signal'),
                    row.get('EMA10'), row.get('EMA50'), row.get('EMA200'),
                    row.get('Fiyat')
                )
                teknik_rows.append({
                    'Sembol': row['Sembol'],
                    'RSI': rsi_d,
                    'MACD': macd_d,
                    'EMA Durumu': ema_d,
                    'Tavsiye': tav,
                    'Yorum': yorum
                })
            st.session_state.df_technical = pd.DataFrame(teknik_rows)

        except Exception as e:
            st.error(f"Hata: {e}")


if st.session_state.df_result is not None:
    df = st.session_state.df_result
    n = len(df)
    top_n = min(5, n) if n > 0 else 0

    st.success(f"✅ {st.session_state.count_result} hisse bulundu")
    st.dataframe(
        df[['Sembol', 'Fiyat', 'Haftalık %', 'F/K', 'PD/DD', 'Score']].style
        .background_gradient(subset=['PD/DD'], cmap='RdYlGn_r')
        .background_gradient(subset=['F/K'], cmap='RdYlGn_r')
        .background_gradient(subset=['Score'], cmap='RdYlGn')
        .format({
            'Fiyat': '₺{:.2f}',
            'Haftalık %': '{:.2f}%',
            'F/K': '{:.1f}',
            'PD/DD': '{:.2f}',
            'Score': '{:.0f}'
        }),
        use_container_width=True,
        hide_index=True
    )

    if st.session_state.df_technical is not None and not st.session_state.df_technical.empty:
        st.divider()
        st.subheader(f"📊 En İyi {top_n} Teknik Analiz")
        st.dataframe(st.session_state.df_technical, use_container_width=True, hide_index=True)

    if st.button("📤 GitHub'a Kaydet"):
        save_to_github(df)
        st.success("✅ GitHub'a kaydedildi!")
