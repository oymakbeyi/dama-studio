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
    "Marble Kitchen Table": "Elegant ceramic piece on a luxurious white marble kitchen countertop, natural daylight from window, soft shadows, kitchen environment with subtle blurred background, 8k resolution, photorealistic, commercial product photography, soft lighting",
    "Wooden Living Room Console": "Beautiful ceramic product on a warm wooden console table in a contemporary living room, minimalist interior design, soft ambient lighting, cozy atmosphere, depth of field, 8k resolution, photorealistic, professional styling",
    "Modern Concrete Setting": "Ceramic design piece on polished concrete surface, modern industrial aesthetic, clean lines, dramatic side lighting, architectural photography style, 8k resolution, photorealistic, high-end editorial look",
    "White E-commerce Background": "Professional product photography of ceramic piece on pure white seamless background, studio lighting, soft shadows, commercial e-commerce style, 8k resolution, photorealistic, crisp and clean"
}

PRODUCT_TYPES = ["Vase", "Bowl", "Plate", "Cup", "Sculpture", "Decorative Object"]

def make_square_with_padding(image, target_size=1024):
    """
    Add white padding to make image square while preserving aspect ratio.
    This prevents the 'squashed/fat product' issue.
    """
    # Get original dimensions
    width, height = image.size
    
    # Determine the maximum dimension
    max_dim = max(width, height)
    
    # Create a new square image with white background
    new_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    
    # Calculate position to paste the original image (centered)
    paste_x = (max_dim - width) // 2
    paste_y = (max_dim - height) // 2
    
    # Paste the original image onto the white square
    new_image.paste(image, (paste_x, paste_y))
    
    # Resize to target size if needed
    if max_dim != target_size:
        new_image = new_image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    return new_image

def remove_background_and_create_mask(image):
    """
    Remove background using rembg and create an inverted mask.
    White = Background to replace, Black = Product to keep
    """
    try:
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Remove background
        output = remove(img_byte_arr)
        
        # Convert to PIL Image
        output_image = Image.open(io.BytesIO(output)).convert("RGBA")
        
        # Extract alpha channel as mask
        alpha = output_image.split()[3]
        
        # Invert mask: White = Background, Black = Product
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
    Uses stable model version hash.
    """
    try:
        # Convert images to base64
        image_b64 = image_to_base64(image)
        mask_b64 = image_to_base64(mask)
        
        # Create data URIs
        image_uri = f"data:image/png;base64,{image_b64}"
        mask_uri = f"data:image/png;base64,{mask_b64}"
        
        # Set API token
        replicate_client = replicate.Client(api_token=api_token)
        
        # Use specific stable model version
        model_version = "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd595c"
        
        # Run prediction using replicate.run()
        output = replicate_client.run(
            model_version,
            input={
                "image": image_uri,
                "mask": mask_uri,
                "prompt": prompt,
                "num_outputs": 1,
                "guidance_scale": 7.5,
                "num_inference_steps": 25
            }
        )
        
        # Output is a list of URLs or file objects
        if output and len(output) > 0:
            return output[0]
        return None
        
    except Exception as e:
        st.error(f"AI Generation error: {str(e)}")
        return None

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # API Token Input
    api_token = st.text_input(
        "Replicate API Token",
        type="password",
        help="Get your token from replicate.com/account"
    )
    
    st.markdown("---")
    
    # Product Type Selection
    product_type = st.selectbox(
        "Product Type",
        PRODUCT_TYPES,
        index=0
    )
    
    # Scene Selection
    selected_scene = st.selectbox(
        "Background Scene",
        list(SCENE_PROMPTS.keys()),
        index=0
    )
    
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
        # Load original image
        original_image = Image.open(uploaded_file).convert("RGB")
        
        with col1:
            st.image(original_image, caption="Original Image", use_container_width=True)
            st.markdown(f"**Size:** {original_image.size[0]} x {original_image.size[1]}px")
        
        # CRITICAL: Make image square with padding
        square_image = make_square_with_padding(original_image, target_size=1024)
        
        # Process image: Remove background and create mask
        with st.spinner("Removing background..."):
            output_image, mask = remove_background_and_create_mask(square_image)
        
        if output_image and mask:
            with col2:
                st.markdown("#### 🎭 Mask (Debug)")
                st.image(mask, caption="Inverted Mask", use_container_width=True)
                st.markdown("*White = Background to replace*")
            
            # Generate Button
            if st.button("✨ Generate New Background", use_container_width=True):
                if not api_token:
                    st.error("⚠️ Please enter your Replicate API token in the sidebar")
                else:
                    # Get the prompt
                    prompt = SCENE_PROMPTS[selected_scene]
                    
                    with st.spinner(f"Generating {selected_scene}... This may take 30-60 seconds"):
                        result_url = run_inpainting(
                            square_image,
                            mask,
                            prompt,
                            api_token
                        )
                    
                    if result_url:
                        with col3:
                            st.markdown("#### ✨ Result")
                            st.image(result_url, caption="AI Generated Background", use_container_width=True)
                            
                            # Download button
                            st.markdown(f"[⬇️ Download Image]({result_url})")
                            st.success("✅ Generation complete!")
                    else:
                        st.error("Failed to generate image. Please check your API token and try again.")
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
