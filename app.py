import streamlit as st
from rembg import remove
from PIL import Image, ImageOps
import io
import replicate
import os

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Dama Studio", page_icon="🎨", layout="wide")

# CSS Süsleme
st.markdown("""
<style>
    .stApp { background-color: #ffffff; color: #000000; }
    h1 { font-family: 'Helvetica', sans-serif; font-weight: 700; color: #000000; }
    .stButton>button { background-color: #000000; color: white; border-radius: 4px; border: none; padding: 10px 24px; }
    .stButton>button:hover { background-color: #333333; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("DAMA STUDIO v2 🎨")
st.markdown("### El Yapımı Ürünler İçin Yapay Zeka Fotoğraf Stüdyosu")

# --- API ANAHTARI ---
if 'REPLICATE_API_TOKEN' in st.secrets:
    replicate_api = st.secrets['REPLICATE_API_TOKEN']
    os.environ["REPLICATE_API_TOKEN"] = replicate_api
else:
    replicate_api = st.text_input("API Anahtarını Girin", type="password")
    if replicate_api:
        os.environ["REPLICATE_API_TOKEN"] = replicate_api

# --- MENÜ ---
with st.sidebar:
    st.header("📸 Stüdyo Ayarları")
    product_type = st.text_input("Ürün Tipi", value="Seramik Vazo")
    
    scene = st.selectbox(
        "Hangi Ortam?",
        (
            "Mermer Masa & Gün Işığı",
            "Ahşap Konsol & Loş Işık",
            "Beton Zemin & Modern",
            "Düz Beyaz Sonsuz Fon"
        )
    )
    
    # GÜÇLÜ PROMPT STRATEJİSİ
    prompts = {
        "Mermer Masa & Gün Işığı": "placed on a white marble table, bright kitchen background, morning sunlight coming from window, soft shadows, 8k, photorealistic, architectural digest style",
        "Ahşap Konsol & Loş Işık": "placed on a wooden table, cozy warm lighting, blurred living room background, cinematic lighting, 8k, photorealistic",
        "Beton Zemin & Modern": "placed on a grey concrete surface, minimalist style, indoor plant shadows, soft studio lighting, 8k, photorealistic",
        "Düz Beyaz Sonsuz Fon": "placed on a pure white seamless infinity curve background, professional product photography, soft shadow, commercial lighting"
    }
    
    selected_prompt = prompts[scene]

# --- ANA EKRAN ---
uploaded_file = st.file_uploader("Fotoğraf Yükle", type=["jpg", "png", "jpeg"])

if uploaded_file and replicate_api:
    col1, col2 = st.columns(2)
    
    # 1. Orijinal Resmi Aç
    image = Image.open(uploaded_file).convert("RGB")
    
    # Resmi yeniden boyutlandır (Hız ve Kalite için ideal boyut: 1024x1024)
    # Çok büyük resimler işlem süresini uzatır ve hata verebilir.
    image.thumbnail((1024, 1024))
    
    with col1:
        st.caption("Orijinal Fotoğraf")
        st.image(image, use_container_width=True)

    if st.button("✨ Sihirli Dokunuşu Yap (Render)"):
        with st.spinner("Yapay zeka önce maske çıkarıyor, sonra sahneyi boyuyor..."):
            try:
                # ADIM 1: MASKE OLUŞTURMA (En Kritik Kısım)
                # Rembg ile arkaplanı siliyoruz
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                image_bytes = buf.getvalue()
                
                # Arkaplanı temizle (Sadece ürün kalsın)
                no_bg_image = remove(image_bytes)
                pil_no_bg = Image.open(io.BytesIO(no_bg_image)).convert("RGBA")
                
                # Maskeyi Çıkar: Sadece Alpha kanalını al
                # Alpha kanalında; Ürün=Beyaz, Arkaplan=Siyah olur.
                mask = pil_no_bg.split()[-1]
                
                # TERS ÇEVİR: Inpainting için Maske; Değişecek yer BEYAZ, Korunacak yer SİYAH olmalı.
                # Yani Arkaplanı Beyaz, Ürünü Siyah yapıyoruz.
                inverted_mask = ImageOps.invert(mask)
                
                # Dosyaları kaydet (Replicate'e göndermek için)
                image.save("temp_original.jpg")
                inverted_mask.save("temp_mask.png")

                # ADIM 2: REPLICATE (INPAINTING)
                # Orijinal resmi veriyoruz + Nereyi değiştireceğini maske ile söylüyoruz.
                
                output = replicate.run(
                    "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                    input={
                        "prompt": f"Professional product photo of a {product_type}, {selected_prompt}",
                        "negative_prompt": "text, watermark, low quality, distorted, bad anatomy, floating object",
                        "image": open("temp_original.jpg", "rb"),
                        "mask": open("temp_mask.png", "rb"),
                        "strength": 0.99, # 1.0 = Maskeli alanı tamamen yeniden yarat
                        "num_inference_steps": 50,
                        "guidance_scale": 15 # Komuta ne kadar sadık kalsın (Yüksek iyidir)
                    }
                )

                with col2:
                    st.success("İşlem Başarılı!")
                    st.image(output[0], use_container_width=True)
                    
            except Exception as e:
                st.error(f"Hata: {str(e)}")
