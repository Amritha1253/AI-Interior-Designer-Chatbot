import streamlit as st
from PIL import Image
from io import BytesIO
import torch
from diffusers import StableDiffusionImg2ImgPipeline

# Load Stable Diffusion model (first time will download)
@st.cache_resource
def load_pipeline():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")
    return pipe

pipe = load_pipeline()

st.set_page_config(page_title="Free AI Interior Designer", layout="wide")
st.title("🏡 AI Interior Designer Chatbot (Free, Local)")
st.write("Upload a blank room photo, describe your style, and get AI-generated designs without any API key.")

uploaded_image = st.file_uploader("Upload a blank room photo", type=["jpg", "jpeg", "png", "webp"])
style_prompt = st.text_area("Describe your style preferences", placeholder="Example: Modern living room with warm colors, wooden furniture...")

def generate_design(prompt, uploaded_file):
    try:
        init_image = Image.open(uploaded_file).convert("RGB")
        init_image = init_image.resize((512, 512))

        # Run Stable Diffusion img2img
        images = pipe(prompt=prompt, image=init_image, strength=0.8, guidance_scale=7.5).images
        return images[0]
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

if st.button("Generate Design") and uploaded_image and style_prompt:
    st.write("🎨 Generating design locally...")
    design_image = generate_design(style_prompt, uploaded_image)
    if design_image:
        st.image(design_image, caption="AI Generated Interior Design", use_column_width=True)
else:
    st.info("Upload an image and describe your preferences, then click Generate.")
