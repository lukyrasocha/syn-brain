import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from PIL import Image
from scipy.linalg import sqrtm
import clip
import argparse
import json

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

clip_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],  # for CLIP
                         std=[0.26862954, 0.26130258, 0.27577711])
])

# Feature Extraction
def extract_features(images, model, device):
    model = model.to(device)
    features = []
    loader = DataLoader(images, batch_size=16)
    for batch in loader:
        batch = batch.to(device)
        feat = model(batch).mean([2, 3])
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

# Clip Score
def compute_clip(captions, images_tensor, clip_model, device):
    clip_model = clip_model.to(device)
    clip_model.eval()

    # truncate captions
    truncated_captions = [caption[:300] for caption in captions]  # 300 chars
    with torch.no_grad():
        text_tokens = clip.tokenize(truncated_captions).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    loader = DataLoader(images_tensor, batch_size=16)
    image_features_list = []
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features_list.append(image_features)
    image_features = torch.cat(image_features_list, dim=0)

    similarities = (text_features * image_features).sum(dim=-1)
    return similarities.mean().item()

# load images
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

# extract and save real features
def extract_real_features(real_folder, model, device, save_path):
    real_images_tensor = load_image_tensor_dataset(real_folder, preprocess)
    real_feats = extract_features(real_images_tensor, model, device)
    mu_real, sigma_real = compute_stats(real_feats)
    np.savez(save_path, mu=mu_real, sigma=sigma_real, real_feats=real_feats)

# evaluate run
def evaluate_run(gen_folder, captions_path, real_stats_path, model, clip_model, device, score_type="all"):
    results = {}

    if score_type in ["all", "fid", "kid", "no_clip"]:
        stats = np.load(real_stats_path)
        mu_real = stats['mu']
        sigma_real = stats['sigma']
        real_feats = stats['real_feats'] if 'real_feats' in stats else None

        gen_images_tensor_vgg = load_image_tensor_dataset(gen_folder, preprocess)
        gen_feats = extract_features(gen_images_tensor_vgg, model, device)
        mu_gen, sigma_gen = compute_stats(gen_feats)

        if score_type in ["all", "fid", "no_clip"]:
            results['fid'] = compute_fid(mu_real, sigma_real, mu_gen, sigma_gen)

        if score_type in ["all", "kid", "no_clip"]:
            if real_feats is not None:
                results['kid'] = compute_kid(real_feats, gen_feats)
            else:
                print("Warning: real_feats not saved, cannot compute KID properly.")
                results['kid'] = None

    if score_type in ["all", "clip"]:
        gen_images_tensor_clip = load_image_tensor_dataset(gen_folder, clip_preprocess)
        with open(captions_path, 'r') as f:
            json_lines = [json.loads(line) for line in f]
            json_lines.sort(key=lambda x: x['file_name'])
            captions = [entry['text'] for entry in json_lines]
        results['clip_score'] = compute_clip(captions, gen_images_tensor_clip, clip_model, device)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract_real", action="store_true", help="Extract real features and save")
    parser.add_argument("--real_folder", type=str, default="data/raw/Test_All_Images", help="Folder with real images")
    parser.add_argument("--gen_folder", type=str, default="data/synthetic_raw", help="Folder with generated images")
    parser.add_argument("--captions_path", type=str, default="", help="Path to captions.txt")
    parser.add_argument("--real_stats_path", type=str, default="data/metrics/real_stats.npz", help="Path to real stats npz file")
    parser.add_argument("--score_type", type=str, default="all", choices=["all", "fid", "kid", "clip", "no_clip"], help="Which score to compute")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGGFeatureExtractor()
    clip_model, _ = clip.load("ViT-B/32", device=device)

    if args.extract_real:
        extract_real_features(args.real_folder, model, device, args.real_stats_path)
    else:
        results = evaluate_run(
            gen_folder=args.gen_folder,
            captions_path=args.captions_path,
            real_stats_path=args.real_stats_path,
            model=model,
            clip_model=clip_model,
            device=device,
            score_type=args.score_type
        )
        for key, value in results.items():
            print(f"{key.upper()}: {value}")
            
        
        norm_path = os.path.normpath(args.gen_folder)
        parts = norm_path.split(os.sep)
        run, stage = parts[-2], parts[-1]
        
        results_folder = os.path.dirname(args.real_stats_path)
        os.makedirs(results_folder, exist_ok=True)
        results_json_path = os.path.join(results_folder, f"metrics_{run}-{stage}.json")

        full_results = {
            k: float(v) if isinstance(v, np.floating) else v
            for k, v in {
                "fid": results.get("fid", "not_calculated"),
                "kid": results.get("kid", "not_calculated"),
                "clip": results.get("clip_score", "not_calculated")
            }.items()
        }

        with open(results_json_path, 'w') as f:
            json.dump(full_results, f, indent=4)

        print(f"Metrics saved to {results_json_path}")
        
        

