import streamlit as st
from rembg import remove
from PIL import Image
import io
import replicate
import os

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="Dama Studio", page_icon="🎨", layout="wide")

# CSS: Siyah Beyaz Jo Malone Tarzı
st.markdown("""
<style>
    .stApp { background-color: #ffffff; color: #000000; }
    h1 { font-family: 'Helvetica', sans-serif; font-weight: 700; color: #000000; }
    .stButton>button { background-color: #000000; color: white; border-radius: 4px; border: none; padding: 10px 24px; }
    .stButton>button:hover { background-color: #333333; color: white; }
    div[data-testid="stFileUploader"] { border: 1px dashed #000; }
</style>
""", unsafe_allow_html=True)

st.title("DAMA STUDIO 🎨")
st.markdown("### El Yapımı Ürünler İçin Yapay Zeka Fotoğraf Stüdyosu")

# --- API ANAHTARI KONTROLÜ ---
if 'REPLICATE_API_TOKEN' in st.secrets:
    replicate_api = st.secrets['REPLICATE_API_TOKEN']
else:
    # Eğer secrets yoksa (Local test için) manuel giriş
    replicate_api = st.text_input("API Anahtarını Girin", type="password")
    if replicate_api:
        os.environ["REPLICATE_API_TOKEN"] = replicate_api

# --- SOL MENÜ (AYARLAR) ---
with st.sidebar:
    st.header("📸 Stüdyo Ayarları")
    st.info("Bu panel, yüklediğiniz ürünün etrafını yapay zeka ile yeniden tasarlar.")
    
    product_type = st.text_input("Ürün Tipi Nedir?", value="Seramik Vazo")
    
    scene = st.selectbox(
        "Hangi Ortam?",
        (
            "Minimalist Mermer Masa (Gün Işığı)",
            "Rustik Ahşap Konsol (Loş Işık)",
            "Beton Zemin & Botanik (Modern)",
            "Bembeyaz Sonsuz Fon (E-Ticaret)"
        )
    )
    
    # Mimarın Özel Promptları
    prompts = {
        "Minimalist Mermer Masa (Gün Işığı)": "placed on a white carrara marble table, soft morning window light from left, luxury bright kitchen background, bokeh, 8k resolution, photorealistic, architectural digest style",
        "Rustik Ahşap Konsol (Loş Işık)": "placed on a rustic dark oak wooden console, warm cozy lighting, cinematic shadows, blurred interior background, 8k resolution, photorealistic",
        "Beton Zemin & Botanik (Modern)": "placed on a grey concrete pedestal, minimalist architectural style, indoor green plant shadows, soft studio lighting, 8k, photorealistic",
        "Bembeyaz Sonsuz Fon (E-Ticaret)": "placed on a pure white seamless infinity curve studio background, professional product photography, soft ground shadow, commercial lighting, 8k"
    }
    
    selected_prompt_suffix = prompts[scene]

# --- ANA EKRAN ---
uploaded_file = st.file_uploader("Ham Ürün Fotoğrafını Yükle (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    # Resmi Göster
    image = Image.open(uploaded_file)
    with col1:
        st.caption("Orijinal Fotoğraf")
        st.image(image, use_container_width=True)

    # Buton ve İşlem
    if st.button("✨ Sihirli Dokunuşu Yap (Render)"):
        if not replicate_api:
            st.error("Lütfen API anahtarını girin veya sistem yöneticisine başvurun.")
        else:
            with st.spinner("Yapay zeka stüdyoyu hazırlıyor... (Yaklaşık 15-20 sn)"):
                try:
                    # 1. Arka Planı Temizle (Rembg)
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    image_bytes = buf.getvalue()
                    output_image_bytes = remove(image_bytes)
                    
                    # Temizlenmiş resmi kaydet (Replicate'e göndermek için)
                    cleaned_image = Image.open(io.BytesIO(output_image_bytes)).convert("RGBA")
                    cleaned_path = "temp_cleaned.png"
                    cleaned_image.save(cleaned_path)

                    # 2. Replicate'e Gönder (Inpainting)
                    # Model: stability-ai/sdxl
                    output = replicate.run(
                        "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                        input={
                            "prompt": f"Professional product photography of a {product_type}, {selected_prompt_suffix}",
                            "image": open(cleaned_path, "rb"),
                            "mask": open(cleaned_path, "rb"), # Şeffaf alanı maske olarak kullan
                            "strength": 0.95, # Arka planı tamamen değiştir
                            "num_inference_steps": 40,
                            "guidance_scale": 7.5
                        }
                    )

                    # 3. Sonucu Göster
                    with col2:
                        st.caption("Dama Studio Sonuç")
                        st.image(output[0], use_container_width=True)
                        st.success("İşlem Başarılı!")
                        
                except Exception as e:
                    st.error(f"Bir hata oluştu: {str(e)}")
