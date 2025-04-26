import os
import json
import random

def create_split_json(data_dir, output_file, val_ratio=0.2, seed=42):
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

    # Calculate the number of validation images
    num_val = int(len(image_files) * val_ratio)

    # Split the image files into validation and test sets
    val_images = image_files[:num_val]
    test_images = image_files[num_val:]

    # Create the split dictionary
    split_dict = {
        'validation': val_images,
        'test': test_images
    }

    # Save the split dictionary to a JSON file
    with open(output_file, 'w') as f:
        json.dump(split_dict, f, indent=4)

# Define the data directory and output file
data_dir = 'data/raw/Test_All_Images'
output_file = 'data/ViT_training/validation.json'

# Create the split JSON file
create_split_json(data_dir, output_file)
