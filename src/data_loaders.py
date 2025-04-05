import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# root directory is data/raw/training
ROOT_DIR_TRAINING = "data/raw/Training"
# root directory is data/raw/testing
ROOT_DIR_TESTING = "data/raw/Testing"




class BrainTumorDataset(Dataset):
    """
        root_dir (str): Path to the main folder.
        transform (callable, optional): Optional transform to be applied on a sample (e.g., ToTensor).
        class_filter (str or list, optional): If provided, only images from these classes will be loaded.
                                              E.g., class_filter="glioma" or class_filter=["glioma", "meningioma"].
    """
    def __init__(self, root_dir, transform=None, class_filter=None):
        self.root_dir = root_dir
        self.transform = transform

        # Collect subfolders (classes) under root_dir
        all_classes = [
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]

        # Filter classes if class_filter is provided
        if class_filter is not None:
            if isinstance(class_filter, str):
                class_filter = [class_filter]
            classes = [c for c in all_classes if c in class_filter]
        else:
            classes = all_classes

        # Map class names to indices
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        # Gather all (image_path, label) pairs
        self.samples = []
        for cls_name in classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg')):
                    path = os.path.join(cls_dir, fname)
                    label = self.class_to_idx[cls_name]
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        # Return the raw image and label
        return image, label


if __name__ == "__main__":
    # loading only the Glioma class
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = BrainTumorDataset(
        root_dir=ROOT_DIR_TRAINING,
        transform=train_transform,
        class_filter='glioma'  # or pass a list of classes
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Suppose for testing, we want all classes
    test_dataset = BrainTumorDataset(
        root_dir=ROOT_DIR_TESTING,
        transform=train_transform,
        class_filter=None
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    
    # print somethng about the dataset 
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")
    print(f"Classes in training dataset: {train_dataset.class_to_idx.keys()}")
    print(f"Classes in testing dataset: {test_dataset.class_to_idx.keys()}")
    
    

    # Ready for being used in the VLM or Stable Diffusion finetunning 
    for images, labels in train_loader:
        pass
