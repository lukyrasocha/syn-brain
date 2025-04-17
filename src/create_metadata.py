import json
import os

def create_metadata_jsonl(input_json_path, image_root_dir, metadata_name,
                          output_metadata_path=None):
    """
    Creates a metadata JSONL file from JSON captions for use with
    datasets.load_dataset("imagefolder").
    
    Args:
        input_json_path: Path to input JSON file
        image_root_dir: Directory containing image files
        metadata_name: Base name (without extension) for the output
        output_metadata_path: Either a directory (you’ll get
            {metadata_name}.jsonl there) or a full .jsonl filepath.
            Defaults to image_root_dir/metadata_name.jsonl.
    
    Returns:
        int: Number of entries written
    """
    if output_metadata_path is None:
        output_metadata_path = os.path.join(
            image_root_dir, f"{metadata_name}.jsonl"
        )
    elif os.path.isdir(output_metadata_path):
        output_metadata_path = os.path.join(
            output_metadata_path, f"{metadata_name}.jsonl"
        )

    print(f"Reading data from:   {input_json_path}")
    print(f"Image directory:     {image_root_dir}")
    print(f"Writing metadata to: {output_metadata_path}")

    try:
        os.makedirs(os.path.dirname(output_metadata_path), exist_ok=True)
        with open(input_json_path, 'r') as infile, \
             open(output_metadata_path, 'w') as outfile:

            original_data = json.load(infile)
            count = 0
            for entry in original_data:
                file_name = entry.get("image")
                text = entry.get("text")

                if not file_name or not text:
                    print(f"  Skipping entry, missing 'image' or 'text': {entry}")
                    continue

                full_image_path = os.path.join(image_root_dir, file_name)
                if not os.path.exists(full_image_path):
                    print(f"  Warning: Image not found, skipping: {full_image_path}")
                    continue

                metadata_entry = {"file_name": file_name, "text": text}
                outfile.write(json.dumps(metadata_entry) + '\n')
                count += 1

            print(f"Successfully wrote {count} entries to {output_metadata_path}")
            return count

    except FileNotFoundError:
        print(f"Error: Input JSON file not found at {input_json_path}")
        return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0


if __name__ == "__main__":
    image_root_dir = "data/raw/Train_All_Images"
    out_root_dir = "data/preprocessed_json_files"

    tasks = [
        ("data/json_files/captions_gemini.json",   "metadata_gemini"),
        ("data/json_files/captions_llava_med.json", "metadata_llava_med"),
        ("data/json_files/captions_ovis_large.json","metadata_ovis_large"),
    ]

    for input_path, meta_name in tasks:
        create_metadata_jsonl(
            input_json_path=input_path,
            image_root_dir=image_root_dir,
            metadata_name=meta_name,
            output_metadata_path=out_root_dir
        )
