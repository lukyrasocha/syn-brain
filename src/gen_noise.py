import os
import numpy as np
from PIL import Image

def create_grayscale_noise_images(output_folder, num_images=1000, image_size=(512, 512)):
    os.makedirs(output_folder, exist_ok=True)

    for i in range(num_images):
        # Random noise for grayscale (single channel)
        noise = np.random.randint(0, 256, size=(image_size[1], image_size[0]), dtype=np.uint8)
        img = Image.fromarray(noise, mode='L')  # 'L' = grayscale mode
        img.save(os.path.join(output_folder, f"noise_{i:04d}.png"))

if __name__ == "__main__":
    create_grayscale_noise_images("data/noise", num_images=1311, image_size=(512, 512))