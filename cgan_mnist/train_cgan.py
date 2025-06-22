import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import Generator, Discriminator
import os

# Hyperparameters
lr = 0.0002
batch_size = 128
img_size = 28
noise_dim = 100
epochs = 50
num_classes = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Models
G = Generator(noise_dim, num_classes, img_size).to(device)
D = Discriminator(num_classes, img_size).to(device)

# Loss and optimizers
criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

os.makedirs("checkpoints", exist_ok=True)

# Training Loop
for epoch in range(epochs):
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        batch_size = imgs.size(0)

        real = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        z = torch.randn(batch_size, noise_dim).to(device)
        gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
        gen_imgs = G(z, gen_labels)

        real_loss = criterion(D(imgs, labels), real)
        fake_loss = criterion(D(gen_imgs.detach(), gen_labels), fake)
        d_loss = real_loss + fake_loss

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # Train Generator
        z = torch.randn(batch_size, noise_dim).to(device)
        gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
        gen_imgs = G(z, gen_labels)
        g_loss = criterion(D(gen_imgs, gen_labels), real)

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

    print(f"Epoch [{epoch+1}/{epochs}] - D Loss: {d_loss.item():.4f} - G Loss: {g_loss.item():.4f}")

    # Save model every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(G.state_dict(), f"checkpoints/generator_epoch_{epoch+1}.pth")

print("✅ Training Complete!")
