# Image Similarity with Siamese Network

This project trains a **Siamese Neural Network** to learn image similarity using the MNIST dataset. It is designed to work smoothly on **macOS with Apple Silicon (M1/M2)**, and focuses on clean modularity, visualization, and reproducibility.

---

## ✅ Supported Environment

* macOS 11.0 or higher (Apple M1 / M2 chip)
* Python 3.10 or higher
* Virtual environment recommended (e.g. `venv` or `conda`)

---

## 🔧 Installation Steps

### 1. Clone the project

```bash
git clone https://github.com/yourusername/image_similarity_project.git
cd image_similarity_project
```

### 2. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # For macOS/Linux
```

### 3. Install dependencies using `requirements.txt`

Use the following command **to avoid compatibility errors on macOS M1**:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

> ⚠️ Do **not** use `--index-url`, as it replaces the default PyPI and will cause packages like `matplotlib` to fail. Use `--extra-index-url` instead.

---

## 🚀 How to Run the Project

### Step 1: Train the model

```bash
python train.py
```

This will:

* Train the Siamese network
* Save model weights at each epoch
* Plot the training loss curve

### Step 2: Evaluate and visualize

```bash
python evaluate.py
```

This will:

* Load the trained model weights
* Show a few pairs of images
* Display predicted similarity (distance) and ground truth label

---

## 📁 Project Structure

```
image_similarity_project/
│
├── config.py                  # Training configuration
├── train.py                   # Training loop with loss plotting and model saving
├── evaluate.py                # Visualize image pairs and model predictions
│
├── models/
│   ├── encoder.py             # CNN encoder definition
│   └── siamese.py             # Siamese network wrapper
│
├── data/
│   └── pair_dataset.py        # Dataset for generating image pairs
│
├── utils/
│   └── loss.py                # Contrastive loss function
│
├── requirements.txt           # Required packages (for Apple Silicon)
└── README.md                  # Project instructions
```

---

## 📦 requirements.txt (for Apple M1/M2)

```txt
torch
torchvision
torchaudio
matplotlib
numpy
tqdm
```

To install:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

---

## ✨ Features

* Siamese Network with contrastive loss
* Support for visualizing training loss
* Evaluate similarity predictions with example image pairs
* Fully compatible with macOS M1/M2 chip

---

## 📬 Contact

If you have questions or ideas, feel free to open an issue or pull request!
