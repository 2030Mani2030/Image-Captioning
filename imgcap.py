import requests
from io import BytesIO
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel

# Load the CLIP model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# Function to preprocess the image from URL
def preprocess_image_from_url(image_url):
    response = requests.get(image_url)
    response.raise_for_status()  # Raise an exception if the request was unsuccessful
    image = Image.open(BytesIO(response.content))
    image = image.resize((224, 224))
    image = transforms.ToTensor()(image).unsqueeze(0)
    image = image.to(device)
    return image

# Function to preprocess the uploaded image
def preprocess_uploaded_image(uploaded_file):
    image = Image.open(uploaded_file)
    image = image.resize((224, 224))
    image = transforms.ToTensor()(image).unsqueeze(0)
    image = image.to(device)
    return image

# Function to generate a caption for the image
def generate_caption(image):
    inputs = processor(text=None, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probabilities = logits_per_image.softmax(dim=1)
        caption_index = torch.argmax(probabilities)
    caption = processor.tokenizer.decode(caption_index)
    return caption

# Streamlit app code
def main():
    st.set_page_config(layout="wide")

    # Set theme mode (light/dark)
    theme_mode = st.sidebar.radio("Theme Mode", ("Light", "Dark"))
    if theme_mode == "Dark":
        st.markdown(
            """
            <style>
            .stApp {
                color: white;
                background-color: #121212;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # Generate caption from image URL
    st.subheader("Generate Caption from Image URL")
    image_url = st.text_input("Enter the URL of an image", key="url_input")
    if st.button("Generate Caption", key="url_button"):
        if image_url:
            try:
                image = preprocess_image_from_url(image_url)
                caption = generate_caption(image)
                st.success(f"Caption: {caption}")
            except Exception as e:
                st.error("Error occurred during processing. Please try again.")
        else:
            st.warning("Please enter the URL of an image.")

    # Generate caption from uploaded image
    st.subheader("Generate Caption from Uploaded Image")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="file_input")
    if st.button("Generate Caption", key="file_button"):
        if uploaded_file is not None:
            try:
                image = preprocess_uploaded_image(uploaded_file)
                caption = generate_caption(image)
                st.success(f"Caption: {caption}")
            except Exception as e:
                st.error("Error occurred during processing. Please try again.")
        else:
            st.warning("Please upload an image.")

if __name__ == "__main__":
    main()
