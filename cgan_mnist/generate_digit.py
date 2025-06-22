# generate_digit.py
import torch
from models import Generator
import matplotlib.pyplot as plt

# Params
noise_dim = 100
num_classes = 10
img_size = 28
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
G = Generator(noise_dim, num_classes, img_size).to(device)
G.load_state_dict(torch.load("checkpoints/generator_epoch_50.pth", map_location=device))
G.eval()

# Choose digit
digit = int(input("Enter digit (0-9): "))
z = torch.randn(1, noise_dim).to(device)
label = torch.tensor([digit]).to(device)

with torch.no_grad():
    generated = G(z, label).cpu().squeeze()

plt.title(f"Generated Digit: {digit}")
plt.imshow(generated, cmap='gray')
plt.axis('off')
plt.show()
