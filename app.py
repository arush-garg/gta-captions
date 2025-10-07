import streamlit as st
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image


def load_model():
    with st.spinner("Loading model..."):
        model = AutoModelForCausalLM.from_pretrained(
            "Techno03/gta-captioner",
            low_cpu_mem_usage=False
        )

        processor = AutoProcessor.from_pretrained("microsoft/git-large")
    
    if torch.cuda.is_available():
        model.to("cuda")
    elif torch.backends.mps.is_available():
        model.to("mps")
    print(f"Model loaded on {model.device}")
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
    st.image("example/image2.png", caption="Example Image", use_container_width=True)

generate_button = st.button("Generate Caption")

if generate_button:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
    else:
        image = Image.open("example/image2.png").convert("RGB")

    inputs = st.session_state.processor(images=image, return_tensors="pt").to(st.session_state.model.device)
    
    with st.spinner("Generating caption..."):
        outputs = st.session_state.model.generate(
            **inputs,
            max_length=128,
            do_sample=True,
            top_p=0.5,
            # num_beams=3 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else 1,
        )

    caption = st.session_state.processor.batch_decode(outputs, skip_special_tokens=True)[0]
    st.write(f"#### {caption}")