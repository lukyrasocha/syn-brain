import numpy as np
import os
import random
import json  # Import the json module
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

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




# import os
# import json
# import random
# import numpy as np
# from torchvision import transforms
# from torch.utils.data import DataLoader, Dataset
# from PIL import Image

# class CustomDataset(Dataset):
#     def __init__(self, data_dir, image_label_pairs, transform=None):
#         self.data_dir = data_dir
#         self.image_label_pairs = image_label_pairs
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_label_pairs)

#     def __getitem__(self, idx):
#         img_dict = self.image_label_pairs[idx]
#         img_name = img_dict['image']
#         label = img_dict['label']
#         img_path = os.path.join(self.data_dir, img_name)
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image, label

# def calculate_mean_std(dataset):
#     mean = np.zeros(3)
#     std = np.zeros(3)
#     for image, _ in dataset:
#         mean += image.mean(dim=(1, 2)).numpy()
#         std += image.std(dim=(1, 2)).numpy()
#     mean /= len(dataset)
#     std /= len(dataset)
#     return mean, std

# def prepare_training_dataloader(data_dir, batch_size, min_tumor_ratio=0.25, seed=42):
#     # Set the random seed for reproducibility
#     random.seed(seed)

#     # Get the list of all image files with their labels
#     image_files = []
#     for filename in os.listdir(data_dir):
#         if filename.endswith('.jpg'):
#             if filename.startswith('notumor'):
#                 label = 0
#             else:
#                 label = 1
#             image_files.append({'image': filename, 'label': label})

#     # Shuffle the list of image files
#     random.shuffle(image_files)

#     # Separate notumor and other images
#     notumor_images = [img for img in image_files if img['label'] == 0]
#     other_images = [img for img in image_files if img['label'] == 1]

#     # Calculate the current ratio of notumor images
#     total_images = len(image_files)
#     notumor_ratio = len(notumor_images) / total_images

#     # If the notumor ratio is below the minimum required, remove random other images
#     if notumor_ratio < min_tumor_ratio:
#         num_other_images_to_remove = int((min_tumor_ratio * total_images) - len(notumor_images))
#         random.shuffle(other_images)
#         other_images = other_images[num_other_images_to_remove:]

#     # Combine the adjusted lists
#     adjusted_image_files = notumor_images + other_images
#     random.shuffle(adjusted_image_files)

#     # Define transformations without normalization
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),  # Resize images to 224x224
#         transforms.ToTensor()
#     ])

#     # Create the dataset
#     train_dataset = CustomDataset(data_dir, adjusted_image_files, transform=transform)

#     # Calculate mean and std
#     mean, std = calculate_mean_std(train_dataset)

#     # Define transformations with normalization
#     transform_with_norm = transforms.Compose([
#         transforms.Resize((224, 224)),  # Resize images to 224x224
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)
#     ])

#     # Update the dataset with normalization
#     train_dataset = CustomDataset(data_dir, adjusted_image_files, transform=transform_with_norm)

#     # Create the data loader
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#     return train_loader, mean, std

# def prepare_dataloaders(batch_size, split_file='data/ViT_training/validation.json', mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
#     # Load the split JSON file
#     with open(split_file, 'r') as f:
#         split_dict = json.load(f)

#     # Define the data directory
#     data_dir = 'data/raw/Test_All_Images'

#     # Define transformations with normalization
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),  # Resize images to 224x224
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)
#     ])

#     # Create datasets for validation and test
#     val_dataset = CustomDataset(data_dir, split_dict['validation'], transform=transform)
#     test_dataset = CustomDataset(data_dir, split_dict['test'], transform=transform)

#     # Create data loaders for validation and test
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     return val_loader, test_loader

# if __name__ == '__main__':
#     # Define the data directories
#     train_data_dir = 'data/raw/Train_All_Images'
#     split_file = 'data/ViT_training/validation.json'

#     # Prepare the training data loader and calculate mean and std
#     train_loader, mean, std = prepare_training_dataloader(train_data_dir, batch_size=16, min_tumor_ratio=0.25)

#     # Prepare the validation and test data loaders using the calculated mean and std
#     val_loader, test_loader = prepare_dataloaders(batch_size=16, split_file=split_file, mean=mean, std=std)

#     # Iterate through the training loader
#     for images, labels in train_loader:
#         print("Training batch:", images.shape, labels.shape)
#         break

#     # Iterate through the validation loader
#     for images, labels in val_loader:
#         print("Validation batch:", images.shape, labels.shape)
#         break

#     # Iterate through the test loader
#     for images, labels in test_loader:
#         print("Test batch:", images.shape, labels.shape)
#         break
