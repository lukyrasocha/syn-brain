import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from PIL import Image
from scipy.linalg import sqrtm

# Feature Extractor (VGG-16)
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True)
        self.model = nn.Sequential(*list(vgg.features.children())[:30])  # conv layers only
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.model(x)

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # for VGG
                         std=[0.229, 0.224, 0.225])
])



# Feature Extraction
def extract_features(images, model, device):
    model = model.to(device)
    features = []
    loader = DataLoader(images, batch_size=16)
    for batch in loader:
        batch = batch.to(device)
        feat = model(batch).mean([2, 3])  # Global Average Pooling
        features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)

# FID
def compute_stats(feats):
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma

def compute_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

# KID
def polynomial_kernel(x, y, degree=3, gamma=None, coef0=1):
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    return (gamma * x @ y.T + coef0) ** degree

def compute_kid(x, y, subset_size=10, n_subsets=20):
    n = min(len(x), len(y))
    scores = []
    for _ in range(n_subsets):
        i = np.random.choice(len(x), subset_size, replace=False)
        j = np.random.choice(len(y), subset_size, replace=False)
        x_subset = x[i]
        y_subset = y[j]
        k_xx = polynomial_kernel(x_subset, x_subset).mean()
        k_yy = polynomial_kernel(y_subset, y_subset).mean()
        k_xy = polynomial_kernel(x_subset, y_subset).mean()
        scores.append(k_xx + k_yy - 2 * k_xy)
    return np.mean(scores)

# Helper function to load images
def load_image_tensor_dataset(folder, transform, limit=None):
    from torch.utils.data import Dataset
    class ImageFolderDataset(Dataset):
        def __init__(self, folder, transform):
            self.paths = sorted([
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith('.jpg') or f.lower().endswith('.png')
            ])
            self.transform = transform
            if limit:
                self.paths = self.paths[:limit]

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert('RGB')
            return self.transform(img)

    return ImageFolderDataset(folder, transform)

def evaluate_run(real_images_tensor, gen_images_tensor, model, device):
    
    # feature vectors from extractor
    print("Extracting features...")
    real_feats = extract_features(real_images_tensor, model, device)
    gen_feats = extract_features(gen_images_tensor, model, device)
    
    # fid stats
    print("Computing FID stats...")
    mu_real, sigma_real = compute_stats(real_feats)
    mu_gen, sigma_gen = compute_stats(gen_feats)
    
    # fid and kid
    print("Computing FID and KID...")
    fid = compute_fid(mu_real, sigma_real, mu_gen, sigma_gen)
    kid = compute_kid(real_feats, gen_feats)
    
    return fid, kid


if __name__ == "__main__":
    fid, kid = evaluate_run(
        real_images_tensor=load_image_tensor_dataset("images/dummy/real", preprocess),
        gen_images_tensor=load_image_tensor_dataset("images/dummy/generated_bad", preprocess),
        model=VGGFeatureExtractor(),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"FID: {fid}")
    print(f"KID: {kid}")