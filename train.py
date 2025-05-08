import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.encoder import Encoder
from models.siamese import SiameseNet
from data.pair_dataset import PairDataset
from utils.loss import contrastive_loss
import config
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

# Define transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# Load dataset and dataloader
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
pair_dataset = PairDataset(dataset)
dataloader = DataLoader(pair_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder()
model = SiameseNet(encoder).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

# List to store the loss for plotting
losses = []

# Training loop
for epoch in range(config.EPOCHS):
    model.train()
    total_loss = 0
    for img1, img2, label in dataloader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        out1, out2 = model(img1, img2)
        loss = contrastive_loss(out1, out2, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{config.EPOCHS}, Loss: {avg_loss:.4f}")

    # Save the model every epoch (or choose a specific frequency)
    torch.save(model.state_dict(), f"siamese_mnist_epoch_{epoch+1}.pth")

# Plot the training loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, config.EPOCHS + 1), losses, marker='o')
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
