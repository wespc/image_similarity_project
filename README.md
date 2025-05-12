# Image Similarity with Siamese Network

This project trains a Siamese neural network to compute similarity between pairs of images using either the MNIST or CIFAR-10 datasets.

---

## 📁 Project Structure

```
image_similarity_project/
├── train.py                 # Train the Siamese model
├── evaluate.py              # Evaluate and visualize predictions
├── config.py                # Dataset-specific settings
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── models/
│   ├── siamese.py           # Simple and enhanced Siamese networks
│   └── encoder.py           # Optional encoder module
├── utils/
│   └── loss.py              # Contrastive loss function
├── data/
│   └── pair_dataset.py      # Dataset wrapper for paired image inputs
```

---

## ⚙️ Environment Setup

We strongly recommend using a virtual environment.

```bash
# Create and activate virtual environment (Mac/Linux)
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\\Scripts\\activate
```

---

## 📦 Install Dependencies

> ✅ Includes CPU-only compatible versions for M1 Macs

```bash
pip install -r requirements.txt
```

If `matplotlib` or others fail from `pytorch.org`, you can try:

```bash
pip install matplotlib
```

---

## 🚀 Training the Model

Run training using either MNIST or CIFAR-10:

```bash
python train.py --dataset mnist
python train.py --dataset cifar
```

- Models will be saved as: `model_<dataset>_epoch_<n>.pt`
- Loss curves will be saved as: `loss_curve_<dataset>.png`

---

## 📊 Evaluating the Model

The evaluation script automatically infers dataset from the model filename.

```bash
python evaluate.py --model-path checkpoints/siamese_mnist_epoch_10.pt --num-samples 5
python evaluate.py --model-path checkpoints/siamese_cifar_epoch_10.pt --num-samples 5
```

---

## 📌 Key Features

- Siamese architecture for image pair similarity
- `SimpleSiameseNetwork` for MNIST, `EnhancedSiameseNetwork` for CIFAR
- Automatic inference of dataset from model path
- Configurable learning rates, batch size, and architecture via `config.py`
- Visualization of predicted similarity between image pairs

---

## 🧠 Author Notes

- Ensure the model filename contains either `mnist` or `cifar` for evaluation.
- Use larger models and data augmentation for more robust CIFAR performance.