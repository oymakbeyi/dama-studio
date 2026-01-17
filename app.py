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
    Remove background using rembg and create mask.
    BLACK = areas to keep (product)
    WHITE = areas to inpaint (background)
    """
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        output = remove(img_byte_arr)
        output_image = Image.open(io.BytesIO(output)).convert("RGBA")
        
        # Extract alpha channel
        alpha = output_image.split()[3]
        
        # Create mask: Black where product is (alpha > 0), White where background is (alpha = 0)
        # This is the CORRECT format for most inpainting models
        mask = Image.new('L', alpha.size, 255)  # Start with all white
        mask.paste(0, mask=alpha)  # Put black where product is (alpha > 0)
        
        return output_image, mask
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
    Run FLUX Inpainting - State-of-the-art 2025 model.
    """
    try:
        replicate_client = replicate.Client(api_token=api_token)
        
        # Convert to base64
        image_b64 = image_to_base64(image)
        mask_b64 = image_to_base64(mask)
        
        image_uri = f"data:image/png;base64,{image_b64}"
        mask_uri = f"data:image/png;base64,{mask_b64}"
        
        # Try FLUX models in order of quality
        models = [
            {
                "id": "zsxkib/flux-dev-inpainting",
                "name": "FLUX Dev Inpainting (High Quality)",
                "params": {
                    "image": image_uri,
                    "mask": mask_uri,
                    "prompt": prompt,
                    "strength": 0.85,
                    "num_inference_steps": 28,
                    "guidance_scale": 3.5,
                    "output_format": "png",
                    "output_quality": 95
                }
            },
            {
                "id": "zsxkib/flux-schnell-inpainting",
                "name": "FLUX Schnell (Fast)",
                "params": {
                    "image": image_uri,
                    "mask": mask_uri,
                    "prompt": prompt,
                    "strength": 0.85,
                    "num_inference_steps": 4,
                    "output_format": "png",
                    "output_quality": 90
                }
            }
        ]
        
        for model_info in models:
            try:
                st.info(f"🎨 Using: {model_info['name']}...")
                
                output = replicate_client.run(
                    model_info['id'],
                    input=model_info['params']
                )
                
                # Handle output
                if output:
                    # FLUX models return FileOutput or string
                    if hasattr(output, 'url'):
                        st.success(f"✅ Success with {model_info['name']}")
                        return output.url
                    elif isinstance(output, str):
                        st.success(f"✅ Success with {model_info['name']}")
                        return output
                    elif hasattr(output, '__str__'):
                        result_str = str(output)
                        st.success(f"✅ Success with {model_info['name']}")
                        return result_str
                        
            except Exception as e:
                error_msg = str(e)
                st.warning(f"⚠️ {model_info['name']} failed: {error_msg[:150]}")
                continue
        
        st.error("❌ All FLUX models failed. Please check your API token or try again later.")
        return None
        
    except Exception as e:
        st.error(f"Critical error: {str(e)}")
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
col1, col2 = st.columns(2)

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
                st.markdown("#### 🎭 Mask Preview")
                st.image(mask, caption="Background Detection", use_container_width=True)
                st.caption("⚫ Black = Product (keep) | ⚪ White = Background (replace)")
            
            # BIG GENERATE BUTTON - Very visible!
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("## 🎨 Step 3: Generate New Background")
            st.markdown(f"**Selected Scene:** {selected_scene}")
            
            generate_clicked = st.button(
                "🚀 GENERATE NEW BACKGROUND NOW!", 
                use_container_width=True,
                type="primary",
                key="generate_btn",
                help=f"Click to place your product in: {selected_scene}"
            )
            st.markdown("---")
            
            if generate_clicked:
                if not api_token:
                    st.error("⚠️ Please enter your Replicate API token in the sidebar")
                else:
                    prompt = SCENE_PROMPTS[selected_scene]
                    
                    # Clear status display
                    status_container = st.container()
                    with status_container:
                        st.success(f"🎨 **Generating Scene:** {selected_scene}")
                        st.info(f"📝 **AI Prompt:** {prompt[:150]}...")
                    
                    with st.spinner(f"⏳ AI is working... This takes 30-60 seconds"):
                        result_url = run_inpainting(square_image, mask, prompt, api_token)
                    
                    if result_url:
                        st.balloons()
                        st.markdown("---")
                        st.markdown("# ✨ SUCCESS! Your New Product Photo")
                        st.markdown("---")
                        
                        # Convert result_url to proper format for st.image()
                        try:
                            # If it's a FileOutput object, get the URL string
                            if hasattr(result_url, 'url'):
                                image_url = result_url.url
                            elif hasattr(result_url, '__str__'):
                                image_url = str(result_url)
                            else:
                                image_url = result_url
                            
                            # Display the result
                            st.image(image_url, caption=f"🎨 {selected_scene}", use_container_width=True)
                            
                            # Download section
                            st.markdown("### 📥 Download Options:")
                            col_dl1, col_dl2 = st.columns(2)
                            with col_dl1:
                                st.markdown(f"**[⬇️ Download High-Res Image]({image_url})**")
                            with col_dl2:
                                st.info("💡 Tip: Right-click → Save Image As...")
                            
                            # Before & After comparison
                            st.markdown("---")
                            st.markdown("### 📊 Before & After Comparison")
                            comp_col1, comp_col2 = st.columns(2)
                            with comp_col1:
                                st.image(original_image, caption="📸 Original Upload", use_container_width=True)
                            with comp_col2:
                                st.image(image_url, caption=f"✨ AI Generated: {selected_scene}", use_container_width=True)
                            
                            st.success("✅ Generation complete! Try different scenes by changing the selection in the sidebar.")
                            
                        except Exception as display_error:
                            st.error(f"Error displaying result: {str(display_error)}")
                            st.info(f"Result URL: {result_url}")
                            st.markdown(f"**[Click here to view/download]({result_url})**")
                    else:
                        st.error("❌ Generation failed. The AI models might be temporarily unavailable. Please try again in a moment.")
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
