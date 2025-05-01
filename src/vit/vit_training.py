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
import wandb

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

def prepare_training_dataloader(args):
    # Set the random seed for reproducibility
    set_seed(args.seed)

    # Get the list of all image files with their labels
    image_files = []
    for filename in os.listdir(args.train_data_dir):
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
    if notumor_ratio < args.min_tumor_ratio:
        num_other_images_to_remove = int((args.min_tumor_ratio * total_images) - len(notumor_images))
        random.shuffle(other_images)
        other_images = other_images[num_other_images_to_remove:]

    # print the number of tumor and no tumor images in the dataset
    print('Number of no tumor images:', len(notumor_images))
    print('Number of tumor images:', len(other_images))
    
    # Combine the adjusted lists
    adjusted_image_files = notumor_images + other_images
    random.shuffle(adjusted_image_files)

    # Define transformations without normalization
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),  # Resize images to 224x224
        transforms.ToTensor()
    ])

    # Create the dataset
    train_dataset = CustomDataset(args.train_data_dir, adjusted_image_files, transform=transform)

    # Calculate mean and std
    mean, std = calculate_mean_std(train_dataset)

    # Define transformations with normalization
    transform_with_norm = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)), 
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Update the dataset with normalization
    train_dataset = CustomDataset(args.train_data_dir, adjusted_image_files, transform=transform_with_norm)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    return train_loader, mean, std

def prepare_dataloaders(args, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    
    # Load the split JSON file
    with open(args.split_file, 'r') as f:
        split_dict = json.load(f)

    # Define transformations with normalization
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),  # Resize images to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_dataset = CustomDataset(args.val_test_data_dir, split_dict['validation'], transform=transform)
    test_dataset = CustomDataset(args.val_test_data_dir, split_dict['test'], transform=transform)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return val_loader, test_loader

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def image_wandb(img, caption, std, mean):
    if img is None:
        return None
    img = img * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
    img = torch.clamp(img, 0, 1)
    return wandb.Image(img.permute(1, 2, 0).cpu().numpy(), caption=caption)

def main(args):
    
    loss_function = nn.CrossEntropyLoss()
    train_loader, mean, std = prepare_training_dataloader(args)
    val_loader, test_loader = prepare_dataloaders(args, mean=mean, std=std)

    model = ViT(image_size=args.image_size, 
                patch_size=args.patch_size, 
                channels=args.channels,
                embed_dim=args.embed_dim, 
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                pos_enc=args.pos_enc, 
                pool=args.pool, 
                dropout=args.dropout, 
                fc_dim=args.fc_dim,
                num_classes=args.num_classes)

    if torch.cuda.is_available():
        model = model.to('cuda')

    opt = torch.optim.AdamW(lr=args.lr, params=model.parameters(), weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / args.warmup_steps, 1.0))

    wandb.init(
        project="vit_training",  # change this to your actual project name
        config=vars(args)  # log all hyperparameters from argparse
    )

    # training loop
    best_val_loss = 1e10
    for e in range(args.num_epochs):

        model.train()
        train_loss = 0
        train_tot, train_cor = 0.0, 0.0

        for image, label in train_loader:
            if torch.cuda.is_available():
                image, label = image.to('cuda'), label.to('cuda')

            opt.zero_grad()
            out = model(image)
            loss = loss_function(out, label)
            loss.backward()
            train_loss += loss.item()

            preds = out.argmax(dim=1)
            train_tot += float(image.size(0))
            train_cor += float((label == preds).sum().item())
            
            # if the total gradient vector has a length > 1, we clip it back down to 1.
            if args.gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
            opt.step()
            sch.step()


        train_loss /= len(train_loader)
        train_acc = train_cor / train_tot
        
        # Validation
        example_images = {"TP": None, "FP": None, "FN": None, "TN": None}
        val_loss = 0

        with torch.no_grad():
            model.eval()
            val_tot, val_cor = 0.0, 0.0
            TP, FP, FN, TN = 0, 0, 0, 0

            for image, label in test_loader:
                if torch.cuda.is_available():
                    image, label = image.to('cuda'), label.to('cuda')

                out = model(image)
                loss = loss_function(out, label)
                val_loss += loss.item()

                out = out.argmax(dim=1)
                val_tot += float(image.size(0))
                val_cor += float((label == out).sum().item())

                # get images for TP, TN, FP, and FN to show in wandb
                for img, true_label, pred_label in zip(image, label, preds):
                    if true_label == 1 and pred_label == 1 and example_images["TP"] is None:
                        example_images["TP"] = img.detach().cpu()
                    elif true_label == 0 and pred_label == 1 and example_images["FP"] is None:
                        example_images["FP"] = img.detach().cpu()
                    elif true_label == 1 and pred_label == 0 and example_images["FN"] is None:
                        example_images["FN"] = img.detach().cpu()
                    elif true_label == 0 and pred_label == 0 and example_images["TN"] is None:
                        example_images["TN"] = img.detach().cpu()

                    # Exit early if we already collected one of each
                    if all(v is not None for v in example_images.values()):
                        break

                    TP += ((preds == 1) & (label == 1)).sum().item()
                    FP += ((preds == 1) & (label == 0)).sum().item()
                    FN += ((preds == 0) & (label == 1)).sum().item()
                    TN += ((preds == 0) & (label == 0)).sum().item()

            val_acc = val_cor / val_tot
            val_loss /= len(val_loader)

            if val_loss <= best_val_loss:
                torch.save(model.state_dict(), args.model_dir)
                best_val_loss = val_loss


        wandb.log({
            "epoch": e,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": val_loss,
            "test_acc": val_acc,
            "learning_rate": opt.param_groups[0]["lr"],
            "val_TP": TP,
            "val_FP": FP,
            "val_FN": FN,
            "val_TN": TN,
            "image_TP": image_wandb(example_images.get("TP"), "True Positive",  mean, std),
            "image_FP": image_wandb(example_images.get("FP"), "False Positive", mean, std),
            "image_FN": image_wandb(example_images.get("FN"), "False Negative", mean, std),
            "image_TN": image_wandb(example_images.get("TN"), "True Negative",  mean, std),
        })
        
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Vision Transformer model")
    parser.add_argument('--train_data_dir', type=str, default='data/raw/Train_All_Images', help='Directory containing training images')
    parser.add_argument('--val_test_data_dir', type=str, default='data/raw/Train_All_Images', help='Directory containing validation and test images')
    parser.add_argument('--split_file', type=str, default='data/vit_training/validation.json', help='Path to the JSON file containing validation and test splits')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--min_tumor_ratio', type=float, default=0.25, help='Minimum ratio of tumor images in the training set')
    parser.add_argument('--image_size', type=int, default=512, help='Size of the input images')
    parser.add_argument('--patch_size', type=int, default=16, help='Size of the patches')
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
    parser.add_argument('--model_dir', type=str, default='models/vit/model.pth', help='Name for the model to save')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model will run on {device}")
    set_seed(seed=args.seed)
    main(args)
