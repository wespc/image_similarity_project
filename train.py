import os
from datetime import datetime
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from models.siamese import SimpleSiameseNetwork, EnhancedSiameseNetwork
from utils.loss import contrastive_loss
from data.pair_dataset import SiameseDataset
from config import get_config
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mnist', 'cifar'], default='mnist')
    args = parser.parse_args()

    config = get_config(args.dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        model = SimpleSiameseNetwork(config['in_channels']).to(device)
    else:
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        model = EnhancedSiameseNetwork(config['in_channels']).to(device)

    siamese_dataset = SiameseDataset(dataset)
    loader = DataLoader(siamese_dataset, batch_size=config['batch_size'], shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    model.train()

    loss_history = []

    for epoch in range(config['epochs']):
        total_loss = 0
        for img1, img2, label in loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            out1, out2 = model(img1, img2)
            loss = contrastive_loss(out1, out2, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f"checkpoints/siamese_{args.dataset}_epoch_{epoch+1}.pt")
    

    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"loss_curve_{args.dataset}.png")
    print("Training completed and loss curve saved.")