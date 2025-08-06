import streamlit as st
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image
import os
import requests
from dotenv import load_dotenv


def load_model():
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")
    if not os.path.exists(st.session_state.dict_path):
        with st.spinner("Downloading model weights..."):
            try:
                load_dotenv()
                url = os.getenv("DOWNLOAD_LINK")
                if not url:
                    raise ValueError("DOWNLOAD_LINK not set in .env file")
            except:
                url = st.secrets['DOWNLOAD_LINK']

            response = requests.get(url, stream=True)
            with open(st.session_state.dict_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    try:
        state_dict = torch.load(st.session_state.dict_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        processor = AutoProcessor.from_pretrained("microsoft/git-base")
    except Exception as e:
        st.error(f"Failed to load model weights. Please check the download link or the model file. Error: {e}")
        os.remove(st.session_state.dict_path)
        return None, None
    
    if torch.cuda.is_available():
        model.to("cuda")
    elif torch.backends.mps.is_available():
        model.to("mps")
    print(f"Model loaded on {model.device}")
    return model, processor


if 'dict_path' not in st.session_state:
    st.session_state.dict_path = "captioning-model.pth"
if 'model' not in st.session_state:
    st.session_state.model, st.session_state.processor = load_model()


st.title("GTA Scene Caption Generator")
st.write("Upload an image of a GTA scene to generate a caption.")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

with col2:
    st.write("Unsure? Click Generate Caption to try this example")
    st.image("example/image.png", caption="Example Image", use_container_width=True)

generate_button = st.button("Generate Caption")

if generate_button:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
    else:
        image = Image.open("example/image.png").convert("RGB")

    inputs = st.session_state.processor(images=image, return_tensors="pt").to(st.session_state.model.device)
    
    with st.spinner("Generating caption..."):
        outputs = st.session_state.model.generate(
            **inputs,
            max_length=128,
            do_sample=True,
            top_p=0.5,
        )

    caption = st.session_state.processor.batch_decode(outputs, skip_special_tokens=True)[0]
    st.write(f"**Generated Caption:** {caption}")