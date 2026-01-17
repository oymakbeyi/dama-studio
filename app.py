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

st.title("DAMA STUDIO (Final Fix) 🎨")
st.markdown("### El Yapımı Ürünler İçin Hassas Koruma Modu")

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
    
    # Promptlar
    prompts = {
        "Mermer Masa & Gün Işığı": "high quality photo of a vase placed on a white marble table, bright modern kitchen background, morning window light, soft shadows, 4k, photorealistic, 8k uhd",
        "Ahşap Konsol & Loş Işık": "high quality photo of a vase placed on a rustic wooden table, cozy warm lighting, blurred living room background, cinematic lighting, 4k, 8k uhd",
        "Beton Zemin & Modern": "high quality photo of a vase placed on a grey concrete pedestal, minimalist architectural style, indoor plant shadows, soft studio lighting, 4k",
        "Düz Beyaz Sonsuz Fon": "high quality photo of a vase placed on a pure white seamless infinity curve background, professional product photography, soft ground shadow"
    }
    
    selected_prompt = prompts[scene]

# --- ANA EKRAN ---
uploaded_file = st.file_uploader("Fotoğraf Yükle", type=["jpg", "png", "jpeg"])

if uploaded_file and replicate_api:
    col1, col2, col3 = st.columns(3)
    
    # 1. Orijinal Resmi Aç
    image = Image.open(uploaded_file).convert("RGB")
    
    # Oran Koruma: Thumbnail ile akıllı küçültme (Asla şişmanlatmaz)
    image.thumbnail((512, 512)) 
    
    with col1:
        st.caption("1. Orijinal")
        st.image(image, use_container_width=True)

    if st.button("✨ Sihirli Dokunuşu Yap"):
        with st.spinner("Model yükleniyor ve sahne kuruluyor..."):
            try:
                # ADIM 1: MASKE OLUŞTURMA
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                image_bytes = buf.getvalue()
                
                # Arkaplanı temizle
                no_bg_image = remove(image_bytes)
                pil_no_bg = Image.open(io.BytesIO(no_bg_image)).convert("RGBA")
                
                # Maskeyi Çıkar
                mask = pil_no_bg.split()[-1]
                
                # TERS ÇEVİR (Invert) -> Siyah = Koru, Beyaz = Değiştir
                inverted_mask = ImageOps.invert(mask)
                
                with col2:
                    st.caption("2. Koruma Maskesi")
                    st.image(inverted_mask, use_container_width=True)

                # Dosyaları kaydet
                image.save("temp_orig.jpg")
                inverted_mask.save("temp_mask.png")

                # ADIM 2: REPLICATE ÇAĞRISI (GÜVENLİ YOL)
                # Direkt model ID'si yerine model ismini kullanıyoruz.
                # 'stability-ai/stable-diffusion-inpainting' yerine daha stabil olan v1-5 inpainting kullanıyoruz.
                
                output = replicate.run(
                    "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd595c",
                    input={
                        "prompt": selected_prompt,
                        "image": open("temp_orig.jpg", "rb"),
                        "mask": open("temp_mask.png", "rb"),
                        "num_inference_steps": 40,
                        "guidance_scale": 7.5
                    }
                )

                with col3:
                    st.caption("3. Sonuç")
                    st.image(output[0], use_container_width=True)
                    st.success("İşlem Başarılı!")
                    
            except Exception as e:
                # Eğer yukarıdaki ID patlarsa diye yedek plan:
                st.warning("Birincil model yanıt vermedi, yedek modele geçiliyor...")
                try:
                     output = replicate.run(
                        "andreasjansson/stable-diffusion-inpainting:e490d072a34a94a11e9711ed5a6ba621c3fab39a999401510e05695d03115b0f",
                        input={
                            "prompt": selected_prompt,
                            "image": open("temp_orig.jpg", "rb"),
                            "mask": open("temp_mask.png", "rb"),
                        }
                    )
                     with col3:
                        st.caption("3. Sonuç (Yedek)")
                        st.image(output[0], use_container_width=True)
                        st.success("İşlem Başarılı!")
                except Exception as final_e:
                    st.error(f"Hata: {str(final_e)}")
