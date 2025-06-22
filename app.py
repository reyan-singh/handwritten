# app.py

import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from cgan_mnist.models import Generator

# Constants
latent_dim = 100
num_classes = 10
img_shape = (1, 28, 28)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Generator
generator = Generator(latent_dim, num_classes, img_shape).to(device)
generator.load_state_dict(torch.load("cgan_mnist/checkpoints/generator_epoch_50.pth", map_location=device))
generator.eval()

# Digit generator
def generate_digit(digit):
    z = torch.randn(1, latent_dim).to(device)
    labels = torch.tensor([digit]).to(device)
    with torch.no_grad():
        gen_img = generator(z, labels).cpu()
    return gen_img

# Streamlit UI
st.set_page_config(page_title="Handwritten Digit Generator", layout="centered")
st.markdown(
    """
    <h1 style='text-align: center; color: #2c3e50;'>Handwritten Digit Generator</h1>
    <p style='text-align: center; font-size:16px;'>Generate MNIST-style handwritten digits using a trained Conditional GAN (CGAN).</p>
    <hr style='margin-top: 20px; margin-bottom: 20px;'>
    """,
    unsafe_allow_html=True
)

st.markdown("#### Select a digit to generate:")
digit = st.selectbox("Digit (0–9)", list(range(10)))

st.markdown("---")

if st.button("Generate"):
    with st.spinner("Generating image..."):
        img = generate_digit(digit)
        st.markdown("#### Generated Image:")
        fig, ax = plt.subplots()
        ax.imshow(img.squeeze(), cmap="gray")
        ax.axis("off")
        st.pyplot(fig)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 14px;'>Built with PyTorch and Streamlit</p>",
    unsafe_allow_html=True
)
