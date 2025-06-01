import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# ------------------------------
# Settings
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
image_size = 64
latent_dim = 100
epochs = 50
lr = 0.0002
save_dir = './dcgan_outputs'
os.makedirs(save_dir, exist_ok=True)

# ------------------------------
# Transform and Dataset
# ------------------------------
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize to [-1, 1]
])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ------------------------------
# Generator
# ------------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, x):
        return self.main(x)

# ------------------------------
# Discriminator
# ------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # Probability of real
        )

    def forward(self, x):
        return self.main(x).view(-1)

# ------------------------------
# Initialize models
# ------------------------------
G = Generator().to(device)
D = Discriminator().to(device)

# ------------------------------
# Loss and Optimizers
# ------------------------------
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# ------------------------------
# Fixed noise for visual tracking
# ------------------------------
fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

# ------------------------------
# Training Loop
# ------------------------------
print("Starting training...")

for epoch in range(1, epochs+1):
    loop = tqdm(dataloader, desc=f"Epoch [{epoch}/{epochs}]", leave=False)
    for i, (real_imgs, _) in enumerate(loop):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # Real and fake labels
        real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
        fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)

        # ---------------------
        # Train Discriminator
        # ---------------------
        D.zero_grad()
        outputs_real = D(real_imgs)
        loss_real = criterion(outputs_real, real_labels)

        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_imgs = G(noise)
        outputs_fake = D(fake_imgs.detach())
        loss_fake = criterion(outputs_fake, fake_labels)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # ---------------------
        # Train Generator
        # ---------------------
        G.zero_grad()
        outputs_fake = D(fake_imgs)
        loss_G = criterion(outputs_fake, real_labels)  # Generator wants D to output 1
        loss_G.backward()
        optimizer_G.step()

        # Update tqdm bar
        loop.set_postfix({
            "Loss_D": loss_D.item(),
            "Loss_G": loss_G.item()
        })

    # Save generated images
    if epoch % 5 == 0 or epoch == epochs:
        with torch.no_grad():
            fake = G(fixed_noise).detach().cpu()
        save_image(fake, f"{save_dir}/epoch_{epoch}.png", normalize=True, nrow=8)

print("Training complete. Generated images saved in:", save_dir)
