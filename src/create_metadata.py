import json
import os
from pathlib import Path

def create_metadata_jsonl(input_json_path: str, base_name: str, output_dir: str):
	"""
    Creates a metadata JSONL file from JSON captions for use with
    datasets.load_dataset("imagefolder").
	"""
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	out_train = output_dir / f"{base_name}.jsonl" # train set
	out_test  = output_dir / f"{base_name}_test.jsonl"

	n_train = n_test = 0

	with open(input_json_path, "r") as f_in, \
		 open(out_train, "w") as f_train, \
		 open(out_test,  "w") as f_test:

		for entry in json.load(f_in):
			img_name = entry.get("image")
			caption  = entry.get("text")
			img_path_from_json = entry.get("path")

			if not (img_name and caption):
				print(f"skipping, missing image/text {entry}")
				continue
			if "Train_All_Images" in img_path_from_json:
				split = "train"
			elif "Test_All_Images" in img_path_from_json:
				split = "test"
			else:
				split = None
			if split is None:
				print(f"image not found in either root")
				continue

			jsonl_line = {"file_name": img_name, "text": caption}

			if split == "train":
				f_train.write(json.dumps(jsonl_line) + "\n")
				n_train += 1
			else:
				f_test.write(json.dumps(jsonl_line) + "\n")
				n_test += 1

	print(f"wrote {n_train} train entries to {out_train}")
	print(f"wrote {n_test} test  entries to {out_test}")
	return n_train, n_test


if __name__ == "__main__":
	output_dir = "data/preprocessed_json_files"

	tasks = [
		("captions/captions_gemini.json",    "metadata_gemini"),
		("captions/captions_llava_med.json", "metadata_llava_med"),
		("captions/captions_ovis_large.json","metadata_ovis_large"),
	]

	for in_json, base in tasks:
		create_metadata_jsonl(
			input_json_path=in_json,
			base_name=base,
			output_dir=output_dir
		)