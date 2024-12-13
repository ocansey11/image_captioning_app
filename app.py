import streamlit as st
from PIL import Image
import torch
import requests
from transformers import MllamaForConditionalGeneration, AutoProcessor
from huggingface_hub import login

# Hugging Face token
hf_token = "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"

# Authenticate with Hugging Face Hub
login(token=hf_token)

# Define model ID and processor
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
st.title("Image Captioning with Llama-3.2-11B-Vision-Instruct")

# Display a spinner while loading the model
with st.spinner('Loading model and processor...'):
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_auth_token=hf_token,  # Pass the token here
    )
    processor = AutoProcessor.from_pretrained(model_id, use_auth_token=hf_token)  # Pass the token here

st.success('Model loaded successfully!')

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("Generating caption...")

    # Prepare the input for the model
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)[0]
    
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    # Generate caption
    output = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(output[0])

    st.write(f"Caption: {caption}")
