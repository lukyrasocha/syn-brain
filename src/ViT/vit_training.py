import numpy as np
import os
import random
import json
import torch
from torch import nn
import torch.nn.functional as F
import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from vit import ViT
import argparse

class CustomDataset(Dataset):
    def __init__(self, data_dir, image_label_pairs, transform=None):
        self.data_dir = data_dir
        self.image_label_pairs = image_label_pairs
        self.transform = transform

    def __len__(self):
        return len(self.image_label_pairs)

    def __getitem__(self, idx):
        img_dict = self.image_label_pairs[idx]
        img_name = img_dict['image']
        label = img_dict['label']
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def calculate_mean_std(dataset):
    mean = np.zeros(3)
    std = np.zeros(3)
    for image, _ in dataset:
        mean += image.mean(dim=(1, 2)).numpy()
        std += image.std(dim=(1, 2)).numpy()
    mean /= len(dataset)
    std /= len(dataset)
    return mean, std

def prepare_training_dataloader(data_dir, batch_size, min_tumor_ratio=0.25, seed=42):
    # Set the random seed for reproducibility
    random.seed(seed)

    # Get the list of all image files with their labels
    image_files = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg'):
            if filename.startswith('notumor'):
                label = 0
            else:
                label = 1
            image_files.append({'image': filename, 'label': label})

    # Shuffle the list of image files
    random.shuffle(image_files)

    # Separate notumor and other images
    notumor_images = [img for img in image_files if img['label'] == 0]
    other_images = [img for img in image_files if img['label'] == 1]

    # Calculate the current ratio of notumor images
    total_images = len(image_files)
    notumor_ratio = len(notumor_images) / total_images

    # If the notumor ratio is below the minimum required, remove random other images
    if notumor_ratio < min_tumor_ratio:
        num_other_images_to_remove = int((min_tumor_ratio * total_images) - len(notumor_images))
        random.shuffle(other_images)
        other_images = other_images[num_other_images_to_remove:]

    # Combine the adjusted lists
    adjusted_image_files = notumor_images + other_images
    random.shuffle(adjusted_image_files)

    # Define transformations without normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor()
    ])

    # Create the dataset
    train_dataset = CustomDataset(data_dir, adjusted_image_files, transform=transform)

    # Calculate mean and std
    mean, std = calculate_mean_std(train_dataset)

    # Define transformations with normalization
    transform_with_norm = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Update the dataset with normalization
    train_dataset = CustomDataset(data_dir, adjusted_image_files, transform=transform_with_norm)

    # Create the data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, mean, std

def prepare_dataloaders(batch_size, split_file='data/ViT_training/validation.json', mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    # Load the split JSON file
    with open(split_file, 'r') as f:
        split_dict = json.load(f)

    # Define the data directory
    data_dir = 'data/raw/Test_All_Images'

    # Define transformations with normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Create datasets for validation and test
    val_dataset = CustomDataset(data_dir, split_dict['validation'], transform=transform)
    test_dataset = CustomDataset(data_dir, split_dict['test'], transform=transform)

    # Create data loaders for validation and test
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return val_loader, test_loader

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main(args):
    loss_function = nn.CrossEntropyLoss()

    # Prepare the training data loader and calculate mean and std
    train_data_dir = args.train_data_dir
    split_file = args.split_file
    train_loader, mean, std = prepare_training_dataloader(train_data_dir, batch_size=args.batch_size, min_tumor_ratio=args.min_tumor_ratio)

    # Prepare the validation and test data loaders using the calculated mean and std
    val_loader, test_loader = prepare_dataloaders(batch_size=args.batch_size, split_file=split_file, mean=mean, std=std)

    model = ViT(image_size=args.image_size, patch_size=args.patch_size, channels=args.channels,
                embed_dim=args.embed_dim, num_heads=args.num_heads, num_layers=args.num_layers,
                pos_enc=args.pos_enc, pool=args.pool, dropout=args.dropout, fc_dim=args.fc_dim,
                num_classes=args.num_classes)

    if torch.cuda.is_available():
        model = model.to('cuda')

    opt = torch.optim.AdamW(lr=args.lr, params=model.parameters(), weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / args.warmup_steps, 1.0))

    # training loop
    best_val_loss = 1e10
    for e in range(args.num_epochs):
        print(f'\n epoch {e}')
        model.train()
        train_loss = 0
        for image, label in tqdm.tqdm(train_loader):
            if torch.cuda.is_available():
                image, label = image.to('cuda'), label.to('cuda')
            opt.zero_grad()
            out = model(image)
            loss = loss_function(out, label)
            loss.backward()
            train_loss += loss.item()
            # if the total gradient vector has a length > 1, we clip it back down to 1.
            if args.gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
            opt.step()
            sch.step()

        train_loss /= len(train_loader)

        val_loss = 0
        with torch.no_grad():
            model.eval()
            tot, cor = 0.0, 0.0
            for image, label in val_loader:
                if torch.cuda.is_available():
                    image, label = image.to('cuda'), label.to('cuda')
                out = model(image)
                loss = loss_function(out, label)
                val_loss += loss.item()
                out = out.argmax(dim=1)
                tot += float(image.size(0))
                cor += float((label == out).sum().item())
            acc = cor / tot
            val_loss /= len(val_loader)
            print(f'-- train loss {train_loss:.3f} -- validation accuracy {acc:.3f} -- validation loss: {val_loss:.3f}')
            if val_loss <= best_val_loss:
                torch.save(model.state_dict(), 'model.pth')
                best_val_loss = val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Vision Transformer model")
    parser.add_argument('--train_data_dir', type=str, default='data/raw/Train_All_Images', help='Directory containing training images')
    parser.add_argument('--split_file', type=str, default='data/ViT_training/validation.json', help='Path to the JSON file containing validation and test splits')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--min_tumor_ratio', type=float, default=0.25, help='Minimum ratio of tumor images in the training set')
    parser.add_argument('--image_size', type=tuple, default=(224, 224), help='Size of the input images')
    parser.add_argument('--patch_size', type=tuple, default=(16, 16), help='Size of the patches')
    parser.add_argument('--channels', type=int, default=3, help='Number of channels in the input images')
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--pos_enc', type=str, default='learnable', help='Type of positional encoding')
    parser.add_argument('--pool', type=str, default='cls', help='Type of pooling')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--fc_dim', type=int, default=None, help='Dimension of the fully connected layer')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=625, help='Number of warmup steps')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--gradient_clipping', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model will run on {device}")
    set_seed(seed=args.seed)
    main(args)
