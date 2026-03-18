import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import instaloader
import re

# ==========================================
# 1. SETUP MODEL & SCRAPER
# ==========================================
st.set_page_config(page_title="Fluency AI - Bot Radar", page_icon="🤖", layout="wide")


@st.cache_resource
def load_ai_engine():
    model = tf.keras.models.load_model("fluency_fake_follower_model_v1.keras")
    scaler = joblib.load("fluency_scaler_v1.pkl")
    return model, scaler


# Inisiasi Instaloader (Scraper)
L = instaloader.Instaloader()

try:
    model, scaler = load_ai_engine()
    system_ready = True
except Exception as e:
    st.error(f"Gagal memuat Model AI. Error: {e}")
    system_ready = False


# ==========================================
# 2. FUNGSI EKSTRAKSI FITUR (Text to Numbers)
# ==========================================
def hitung_rasio_angka(teks):
    if not teks:
        return 0.0
    jumlah_angka = sum(c.isdigit() for c in teks)
    return jumlah_angka / len(teks)


def ekstrak_fitur(profile):
    """Mengubah data profil mentah menjadi 11 angka yang dimengerti AI"""
    # 1. profile_pic (1 jika ada URL, 0 jika default)
    punya_foto = 1 if "default" not in profile.profile_pic_url else 0

    # 2. nums/length username
    rasio_angka_uname = hitung_rasio_angka(profile.username)

    # 3. fullname words
    nama_lengkap = profile.full_name if profile.full_name else ""
    jumlah_kata_nama = len(nama_lengkap.split())

    # 4. nums/length fullname
    rasio_angka_nama = hitung_rasio_angka(nama_lengkap)

    # 5. name == username
    nama_sama = 1 if profile.username.lower() == nama_lengkap.lower() else 0

    # 6. description length
    bio_length = len(profile.biography) if profile.biography else 0

    # 7. external URL
    ada_link = 1 if profile.external_url else 0

    # 8. private
    is_private = 1 if profile.is_private else 0

    # Menggabungkan 11 fitur sesuai urutan saat training
    return np.array(
        [
            [
                punya_foto,
                rasio_angka_uname,
                jumlah_kata_nama,
                rasio_angka_nama,
                nama_sama,
                bio_length,
                ada_link,
                is_private,
                profile.mediacount,
                profile.followers,
                profile.followees,
            ]
        ]
    )


# ==========================================
# 3. UI STREAMLIT
# ==========================================
st.title("🤖 Fleuncy AI: Live Fake Follower Radar")
st.markdown("Masukkan *username* Instagram untuk dianalisis langsung oleh AI.")
st.markdown("---")

if system_ready:
    col1, col2 = st.columns([1, 1])

    with col1:
        username_input = st.text_input(
            "Username Instagram (Contoh: cristiano / jessnolimit)", ""
        )
        analyze_btn = st.button("Scrape & Analisis! 🚀", use_container_width=True)

    with col2:
        if analyze_btn and username_input:
            # Bersihkan input jika user memasukkan URL penuh
            clean_username = (
                username_input.replace("https://www.instagram.com/", "")
                .replace("/", "")
                .strip()
            )

            with st.spinner(
                f"Menyusup ke Instagram untuk mengambil data @{clean_username}..."
            ):
                try:
                    # PROSES SCRAPING
                    profile = instaloader.Profile.from_username(
                        L.context, clean_username
                    )

                    st.success("Data berhasil ditarik!")

                    # Tampilkan metrik dasar
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Followers", f"{profile.followers:,}")
                    c2.metric("Following", f"{profile.followees:,}")
                    c3.metric("Posts", f"{profile.mediacount:,}")

                    # PROSES PREDIKSI AI
                    st.markdown("### 🎯 Hasil Analisis AI")
                    data_input = ekstrak_fitur(profile)

                    # Scale data & Predict
                    data_scaled = scaler.transform(data_input)
                    probability = model.predict(data_scaled)[0][0]
                    bot_percentage = probability * 100

                    # Visualisasi Hasil
                    if probability > 0.6:
                        st.error(f"## {bot_percentage:.1f}% Indikasi Bot")
                        st.error(
                            "🚨 **BAHAYA:** Pola profil ini identik dengan akun Spam/Fake!"
                        )
                    elif probability > 0.4:
                        st.warning(f"## {bot_percentage:.1f}% Indikasi Bot")
                        st.warning(
                            "⚠️ **WASPADA:** Akun ini memiliki perilaku statistik yang mencurigakan."
                        )
                    else:
                        st.success(f"## {bot_percentage:.1f}% Indikasi Bot")
                        st.success(
                            "✅ **AMAN:** AI mengklasifikasikan ini sebagai manusia organik."
                        )

                except instaloader.exceptions.ProfileNotExistsException:
                    st.error("❌ Username tidak ditemukan di Instagram.")
                except instaloader.exceptions.ConnectionException:
                    st.error(
                        "❌ Koneksi diblokir oleh Instagram (Rate Limit). Coba lagi nanti."
                    )
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan: {e}")
