import streamlit as st
from rembg import remove
from PIL import Image, ImageOps
import io
import replicate
import os

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Dama Studio", page_icon="🎨", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #ffffff; color: #000000; }
    h1 { font-family: 'Helvetica', sans-serif; font-weight: 700; color: #000000; }
    .stButton>button { background-color: #000000; color: white; border-radius: 4px; border: none; padding: 10px 24px; }
    .stButton>button:hover { background-color: #333333; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("DAMA STUDIO (Auto-Update) 🎨")
st.markdown("### El Yapımı Ürünler İçin Hassas Koruma Modu")

# --- API ANAHTARI ---
if 'REPLICATE_API_TOKEN' in st.secrets:
    replicate_api = st.secrets['REPLICATE_API_TOKEN']
    os.environ["REPLICATE_API_TOKEN"] = replicate_api
else:
    replicate_api = st.text_input("API Anahtarını Girin", type="password")
    if replicate_api:
        os.environ["REPLICATE_API_TOKEN"] = replicate_api

# --- FONKSİYON: GÖRSELİ KARE YAP (TOMBUL VAZO SORUNU ÇÖZÜMÜ) ---
def make_square(im, min_size=512, fill_color=(255, 255, 255, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im.resize((512, 512)) # AI için standart boyut

# --- MENÜ ---
with st.sidebar:
    st.header("📸 Stüdyo Ayarları")
    scene = st.selectbox(
        "Hangi Ortam?",
        (
            "Mermer Masa & Gün Işığı",
            "Ahşap Konsol & Loş Işık",
            "Beton Zemin & Modern",
            "Düz Beyaz Sonsuz Fon"
        )
    )
    
    prompts = {
        "Mermer Masa & Gün Işığı": "high quality photo of a vase on a white marble table, bright modern kitchen background, morning window light, soft shadows, 4k, photorealistic, 8k uhd",
        "Ahşap Konsol & Loş Işık": "high quality photo of a vase on a rustic wooden table, cozy warm lighting, blurred living room background, cinematic lighting, 4k, 8k uhd",
        "Beton Zemin & Modern": "high quality photo of a vase on a grey concrete pedestal, minimalist architectural style, indoor plant shadows, soft studio lighting, 4k",
        "Düz Beyaz Sonsuz Fon": "high quality photo of a vase on a pure white seamless infinity curve background, professional product photography, soft ground shadow"
    }
    selected_prompt = prompts[scene]

# --- ANA EKRAN ---
uploaded_file = st.file_uploader("Fotoğraf Yükle", type=["jpg", "png", "jpeg"])

if uploaded_file and replicate_api:
    col1, col2, col3 = st.columns(3)
    
    # 1. Orijinal Resmi Hazırla
    original = Image.open(uploaded_file).convert("RGBA")
    
    # KARE YAP: Vazonun şeklini korumak için kenarlara boşluk ekle
    original_square = make_square(original)
    
    with col1:
        st.caption("1. Orijinal (Kare Format)")
        st.image(original_square, use_container_width=True)

    if st.button("✨ Sihirli Dokunuşu Yap"):
        with st.spinner("Modelin en güncel versiyonu bulunuyor ve çalıştırılıyor..."):
            try:
                # ADIM 1: MASKE OLUŞTURMA
                buf = io.BytesIO()
                original_square.save(buf, format="PNG")
                image_bytes = buf.getvalue()
                
                # Arkaplanı temizle
                no_bg_image = remove(image_bytes)
                pil_no_bg = Image.open(io.BytesIO(no_bg_image)).convert("RGBA")
                
                # Maskeyi Çıkar ve Ters Çevir
                mask = pil_no_bg.split()[-1]
                inverted_mask = ImageOps.invert(mask)
                
                with col2:
                    st.caption("2. Koruma Maskesi")
                    st.image(inverted_mask, use_container_width=True)

                # Dosyaları kaydet
                original_square.convert("RGB").save("temp_orig.jpg")
                inverted_mask.save("temp_mask.png")

                # ADIM 2: DİNAMİK MODEL SEÇİMİ (MİMARIN ZEKASI)
                # Sabit ID yerine, Replicate'e sorup en güncelini alıyoruz.
                model = replicate.models.get("stability-ai/stable-diffusion-inpainting")
                latest_version = model.versions.list()[0] # Listenin başındaki en yenisidir.
                
                output = latest_version.predict(
                    prompt=selected_prompt,
                    image=open("temp_orig.jpg", "rb"),
                    mask=open("temp_mask.png", "rb"),
                    num_inference_steps=40,
                    guidance_scale=7.5
                )

                with col3:
                    st.caption("3. Sonuç")
                    st.image(output[0], use_container_width=True)
                    st.success("İşlem Başarılı!")
                    
            except Exception as e:
                st.error(f"Bir hata oluştu: {str(e)}")
