import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.encoder import Encoder
from models.siamese import SiameseNet
from data.pair_dataset import PairDataset

# Load the trained model (Load the most recent model or any saved model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder()
model = SiameseNet(encoder).to(device)
model.load_state_dict(torch.load("siamese_mnist_epoch_10.pth"))  # Load model from a specific epoch
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# Load dataset
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
pair_dataset = PairDataset(dataset)
dataloader = DataLoader(pair_dataset, batch_size=1, shuffle=True)

# Function to plot image pair and their similarity
def plot_image_pair(img1, img2, label, dist):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img1.squeeze(), cmap='gray')
    axes[0].set_title("Image 1")
    axes[0].axis('off')
    
    axes[1].imshow(img2.squeeze(), cmap='gray')
    axes[1].set_title("Image 2")
    axes[1].axis('off')
    
    plt.suptitle(f"Label: {label.item()} | Distance: {dist:.4f}")
    plt.show()

# Evaluate a few pairs
for i, (img1, img2, label) in enumerate(dataloader):
    if i == 5:  # Limit to 5 pairs
        break

    img1, img2 = img1.to(device), img2.to(device)
    out1, out2 = model(img1, img2)
    
    # Calculate Euclidean distance between the outputs
    dist = torch.nn.functional.pairwise_distance(out1, out2)
    
    plot_image_pair(img1.cpu(), img2.cpu(), label, dist.item())
