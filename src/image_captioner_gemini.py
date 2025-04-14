import os
import json
import PIL.Image
from google import genai
from tqdm import tqdm  

API_KEY = "api" 
client = genai.Client(api_key=API_KEY)


def get_class_from_filename(filename):
    return filename.split("_")[0].lower()

def generate_prompt(class_name):
    if class_name == "notumor":
        return (
            "This brain MRI shows no tumor. Analyze this brain MRI. Output format in one line: tumor: yes/no; general_description: describe the brain MRI image in general. Max 77 tokens."
        )
    else:
        return (
            f"This image has {class_name}. Analyze this brain MRI. Output format in one line: tumor: yes/no; if yes—location: brain region; size: small/medium/large; shape; intensity; orientation: axial/saggital/coronal; general description: describe the brain MRI in general and also mention any other abnormalities. Max 77 tokens"
        )

def process_images():
    results = []
    all_image_paths = []

    directories = ["data/raw/Train_All_Images", "data/raw/Test_All_Images"]

    for dir_path in directories:
        if not os.path.exists(dir_path):
            print(f"Directory '{dir_path}' does not exist. Skipping.")
            continue

        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    all_image_paths.append(full_path)
    for full_path in tqdm(all_image_paths, desc="Processing images", unit="image"):
        try:
            base_name = os.path.basename(full_path)
            class_name = get_class_from_filename(base_name)
            prompt = generate_prompt(class_name)

            image = PIL.Image.open(full_path)

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, image]
            )

            caption = response.text.strip()

            print("="*50)
            print(f"Class: {class_name}")
            print(f"Image: {base_name}")
            print("Caption:", caption)
            print("="*50)

            results.append({
                "image": base_name,
                "text": caption,
                "class": class_name,
                "path": full_path,
            })

        except Exception as e:
            print(f"Error processing {full_path}: {e}")

    output_file = "captions/captions_gemini.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Captions saved in {output_file}")

if __name__ == "__main__":
    process_images()