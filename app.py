import streamlit as st
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image

@st.cache_resource("model")
def load_model():
    model = AutoModelForCausalLM.from_pretrained("captioning-model/")
    processor = AutoProcessor.from_pretrained("microsoft/git-base")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, processor

if 'model' not in st.session_state:
    st.session_state.model, st.session_state.processor = load_model()

st.title("GTA Scene Caption Generator")
st.write("Upload an image of a GTA scene to generate a caption.")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

with col2:
    st.write("Unsure? Click Generate Caption to try this example")
    st.image("example/image.png", caption="Example Image", use_column_width=True)

generate_button = st.button("Generate Caption")

if generate_button:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
    else:
        image = Image.open("example/image.png").convert("RGB")

    inputs = st.session_state.processor(images=image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    with st.spinner("Generating caption..."):
        outputs = st.session_state.model.generate(**inputs, max_length=50)
    
    caption = st.session_state.processor.batch_decode(outputs, skip_special_tokens=True)[0]
    st.write(f"**Generated Caption:** {caption}")