import argparse
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from data.pair_dataset import SiameseDataset
from models.siamese import SimpleSiameseNetwork, EnhancedSiameseNetwork
from config import get_config
import matplotlib.pyplot as plt
import os


def infer_dataset_from_path(path):
    path_lower = path.lower()
    if 'mnist' in path_lower:
        return 'mnist'
    elif 'cifar' in path_lower:
        return 'cifar'
    else:
        raise ValueError("Cannot infer dataset from model filename. Please include 'mnist' or 'cifar' in the filename.")


def prepare_image_for_plot(img_tensor):
    img = img_tensor.squeeze().cpu()
    if img.dim() == 2:
        return img.numpy(), 'gray'
    elif img.dim() == 3:
        return img.permute(1, 2, 0).numpy(), None
    else:
        raise ValueError("Unsupported image shape")


def visualize_predictions(model, dataloader, device, num_samples=5):
    model.eval()
    samples_shown = 0
    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            out1, out2 = model(img1, img2)
            euclidean_distance = torch.norm(out1 - out2, dim=1)
            for i in range(len(label)):
                if samples_shown >= num_samples:
                    return
                img1_np, cmap1 = prepare_image_for_plot(img1[i])
                img2_np, cmap2 = prepare_image_for_plot(img2[i])

                plt.figure(figsize=(4, 2))
                plt.subplot(1, 2, 1)
                plt.imshow(img1_np, cmap=cmap1)
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(img2_np, cmap=cmap2)
                plt.axis('off')
                plt.suptitle(
                    f"Label: {'Same' if label[i]==0 else 'Different'}, "
                    f"Distance: {euclidean_distance[i]:.2f}"
                )
                plt.show()
                samples_shown += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--num-samples', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = infer_dataset_from_path(args.model_path)
    config = get_config(dataset_name)

    if dataset_name == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        raw_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        model = SimpleSiameseNetwork(config['in_channels']).to(device)
    else:
        transform = transforms.Compose([transforms.ToTensor()])
        raw_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        model = EnhancedSiameseNetwork(config['in_channels']).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    siamese_dataset = SiameseDataset(raw_dataset)
    loader = DataLoader(siamese_dataset, batch_size=config['batch_size'], shuffle=True)

    visualize_predictions(model, loader, device, num_samples=args.num_samples)