import streamlit as st
import replicate
from rembg import remove
from PIL import Image, ImageOps
import numpy as np
import io
import base64

# Page Configuration
st.set_page_config(
    page_title="Dama Studio - AI Background Replacement",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS for Dama Design Aesthetic
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    h1 {
        color: #000000;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 300;
        letter-spacing: 2px;
    }
    .stButton>button {
        background-color: #000000;
        color: #ffffff;
        border: none;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: 300;
        letter-spacing: 1px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #333333;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("DAMA STUDIO")
st.markdown("### AI Background Replacement for Product Photography")
st.markdown("---")

# Predefined High-Quality Prompts
SCENE_PROMPTS = {
    "🏠 Marble Kitchen": "Elegant ceramic piece on a luxurious white marble kitchen countertop, natural daylight from window, soft shadows, kitchen environment with subtle blurred background, 8k resolution, photorealistic, commercial product photography, soft lighting",
    "🪵 Wooden Console": "Beautiful ceramic product on a warm wooden console table in a contemporary living room, minimalist interior design, soft ambient lighting, cozy atmosphere, depth of field, 8k resolution, photorealistic, professional styling",
    "🏗️ Concrete Modern": "Ceramic design piece on polished concrete surface, modern industrial aesthetic, clean lines, dramatic side lighting, architectural photography style, 8k resolution, photorealistic, high-end editorial look",
    "⚪ White Studio": "Professional product photography of ceramic piece on pure white seamless background, studio lighting, soft shadows, commercial e-commerce style, 8k resolution, photorealistic, crisp and clean",
    "🌿 Natural Wood": "Ceramic vase on rustic reclaimed wood surface, natural organic texture, warm earth tones, soft natural window light, botanical styling, 8k resolution, photorealistic, artisanal aesthetic",
    "✨ Luxury Dark": "Premium ceramic piece on black marble surface, dramatic lighting, luxury boutique atmosphere, elegant sophisticated styling, gold accents in background, 8k resolution, photorealistic, high-end editorial"
}

PRODUCT_TYPES = ["Vase", "Bowl", "Plate", "Cup", "Sculpture", "Decorative Object"]

def make_square_with_padding(image, target_size=1024):
    """
    Add white padding to make image square while preserving aspect ratio.
    This prevents the 'squashed/fat product' issue.
    """
    width, height = image.size
    max_dim = max(width, height)
    new_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    paste_x = (max_dim - width) // 2
    paste_y = (max_dim - height) // 2
    new_image.paste(image, (paste_x, paste_y))
    
    if max_dim != target_size:
        new_image = new_image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    return new_image

def remove_background_and_create_mask(image):
    """
    Remove background using rembg and create an inverted mask.
    White = Background to replace, Black = Product to keep
    """
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        output = remove(img_byte_arr)
        output_image = Image.open(io.BytesIO(output)).convert("RGBA")
        alpha = output_image.split()[3]
        inverted_mask = ImageOps.invert(alpha)
        
        return output_image, inverted_mask
    except Exception as e:
        st.error(f"Background removal error: {str(e)}")
        return None, None

def image_to_base64(image):
    """Convert PIL Image to base64 string for API."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def run_inpainting(image, mask, prompt, api_token):
    """
    Run Stable Diffusion Inpainting using Replicate API.
    Uses multiple fallback models for reliability.
    """
    # List of working inpainting models (in priority order)
    models = [
        "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
        "lucataco/sdxl-inpainting:f2a36c58e62d36ad7502f096d9e3e7fb64192f13be3b1f5e023cc9f3e5b4e21a",
        "adirik/sdxl-inpainting:9cff4c80e82e6e0e7f5fa00e12c34f021ed365462b5ec2aaef3c82d0a2d1b87e"
    ]
    
    try:
        image_b64 = image_to_base64(image)
        mask_b64 = image_to_base64(mask)
        
        image_uri = f"data:image/png;base64,{image_b64}"
        mask_uri = f"data:image/png;base64,{mask_b64}"
        
        replicate_client = replicate.Client(api_token=api_token)
        
        # Try each model until one works
        last_error = None
        for model in models:
            try:
                st.info(f"🔄 Trying model: {model.split(':')[0]}...")
                
                # Different models may have different input parameters
                if "sdxl" in model.lower():
                    # SDXL-based models
                    output = replicate_client.run(
                        model,
                        input={
                            "image": image_uri,
                            "mask": mask_uri,
                            "prompt": prompt,
                            "num_outputs": 1,
                            "guidance_scale": 7.5,
                            "num_inference_steps": 30,
                            "scheduler": "K_EULER"
                        }
                    )
                else:
                    # Standard SD 1.5 models
                    output = replicate_client.run(
                        model,
                        input={
                            "image": image_uri,
                            "mask": mask_uri,
                            "prompt": prompt,
                            "num_outputs": 1,
                            "guidance_scale": 7.5,
                            "num_inference_steps": 25
                        }
                    )
                
                # If we get here, it worked!
                if output:
                    # Handle different output formats
                    if isinstance(output, list) and len(output) > 0:
                        return output[0]
                    elif isinstance(output, str):
                        return output
                    
            except Exception as model_error:
                last_error = model_error
                st.warning(f"⚠️ Model failed: {str(model_error)[:100]}")
                continue
        
        # If all models failed
        st.error(f"❌ All models failed. Last error: {str(last_error)}")
        return None
        
    except Exception as e:
        st.error(f"AI Generation error: {str(e)}")
        st.error(f"Details: {e}")
        return None

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Secure API token handling - uses Streamlit secrets
    default_token = st.secrets.get("REPLICATE_API_TOKEN", "")
    api_token = st.text_input(
        "Replicate API Token",
        value=default_token,
        type="password",
        help="Get your token from replicate.com/account"
    )
    
    if not api_token:
        st.warning("⚠️ Please add your API token")
        st.markdown("""
        **For Streamlit Cloud:**
        1. Go to App Settings
        2. Click 'Secrets'
        3. Add: `REPLICATE_API_TOKEN = "your_token_here"`
        """)
    
    st.markdown("---")
    
    st.markdown("**Product Type**")
    product_type = st.selectbox(
        "Product Type",
        PRODUCT_TYPES,
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("**Background Scene**")
    selected_scene = st.selectbox(
        "Background Scene",
        list(SCENE_PROMPTS.keys()),
        index=0,
        label_visibility="collapsed"
    )
    
    with st.expander("📝 View Full Prompt"):
        st.caption(SCENE_PROMPTS[selected_scene])
    
    st.markdown("---")
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. Enter your Replicate API token
    2. Upload product image (JPG/PNG)
    3. Select product type and scene
    4. Click 'Generate New Background'
    5. Download your result
    """)

# Main Content Area
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['png', 'jpg', 'jpeg'],
        label_visibility="collapsed"
    )

if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file).convert("RGB")
        
        with col1:
            st.image(original_image, caption="Original Image", use_container_width=True)
            st.markdown(f"**Size:** {original_image.size[0]} x {original_image.size[1]}px")
        
        square_image = make_square_with_padding(original_image, target_size=1024)
        
        with st.spinner("Removing background..."):
            output_image, mask = remove_background_and_create_mask(square_image)
        
        if output_image and mask:
            with col2:
                st.markdown("#### 🎭 Mask (Debug)")
                st.image(mask, caption="Inverted Mask", use_container_width=True)
                st.markdown("*White = Background to replace*")
        
        # CRITICAL: Show button OUTSIDE the column context
        if output_image and mask:
            st.markdown("---")
            st.markdown("---")
            
            # Big centered button
            _, center_col, _ = st.columns([1, 2, 1])
            with center_col:
                st.markdown("### 🎨 Ready to Generate!")
                generate_clicked = st.button(
                    "✨ GENERATE NEW BACKGROUND", 
                    use_container_width=True,
                    type="primary",
                    key="generate_btn"
                )
            
            st.markdown("---")
            st.markdown("---")
            
            if generate_clicked:
                if not api_token:
                    st.error("⚠️ Please enter your Replicate API token in the sidebar")
                else:
                    prompt = SCENE_PROMPTS[selected_scene]
                    
                    # Show what we're generating
                    st.success(f"🎨 **Generating Scene:** {selected_scene}")
                    st.info(f"📝 **Using Prompt:** {prompt[:100]}...")
                    
                    with st.spinner(f"⏳ Creating your scene... This may take 30-60 seconds"):
                        result_url = run_inpainting(square_image, mask, prompt, api_token)
                    
                    if result_url:
                        st.balloons()
                        st.markdown("---")
                        st.markdown("# ✨ YOUR RESULT IS READY!")
                        st.markdown("---")
                        
                        # Show result in a large display
                        result_col1, result_col2, result_col3 = st.columns([0.5, 3, 0.5])
                        with result_col2:
                            st.image(result_url, caption=f"🎨 {selected_scene}", use_container_width=True)
                            
                            # Download instructions
                            st.markdown("### 📥 Download Your Image:")
                            st.markdown(f"**[Click here to download]({result_url})** or right-click the image above")
                            st.success("✅ Generation complete!")
                            
                            # Show comparison
                            st.markdown("---")
                            st.markdown("### 📊 Before & After")
                            comp_col1, comp_col2 = st.columns(2)
                            with comp_col1:
                                st.image(original_image, caption="Original", use_container_width=True)
                            with comp_col2:
                                st.image(result_url, caption="AI Generated", use_container_width=True)
                    else:
                        st.error("❌ Failed to generate image. Please try again or check your API token.")
        else:
            st.error("Failed to process image. Please try a different image.")
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.exception(e)
else:
    with col1:
        st.info("👆 Upload an image to get started")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 14px;'>"
    "Dama Studio © 2024 | Powered by AI"
    "</div>",
    unsafe_allow_html=True
)
