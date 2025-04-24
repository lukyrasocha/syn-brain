import json
import random

SEED = 42
NUM_SAMPLES = 1000
NO_TUMOR_RATE = 0.28

random.seed(SEED)

num_no  = int(NUM_SAMPLES * NO_TUMOR_RATE)   # 280
num_yes = NUM_SAMPLES - num_no               # 720
flags   = ["no"] * num_no + ["yes"] * num_yes
random.shuffle(flags)


tumor_types_llava = ["glioma", "meningioma", "pituitary tumor"]
locations_llava   = ["pituitary region", "left sylvian fissure", "frontal region", "right frontal region"]
locations_glioma  = ["frontal lobe", "temporal lobe", "parietal lobe", "occipital lobe", "brainstem", "cerebellum"]
locations_mening  = ["parasagittal region", "sphenoid wing", "convexity", "falx"]
locations_pit     = ["pituitary region", "sella turcica", "suprasellar area"]

sizes_llava       = ["small", "medium", "large", "not specified"]
shapes_llava      = ["round", "irregular", "not specified"]
intensities_llava = ["iso-intense", "similar to CSF", "hyperintense", "hypointense", "not specified"]
orientations_llava= ["axial", "sagittal", "coronal", "not specified"]

llava_tumor_intros = [
    "The brain MRI shows a {tumor_type}.",
    "The brain MRI shows a tumor in the {location}.",
    "The brain MRI shows the presence of {tumor_type}.",
]

llava_tumor_explanations = {
    "glioma": [
        "which is a type of brain tumor. Gliomas arise from glial cells, which are supportive cells in the brain.",
        "Gliomas are a type of brain tumor originating from glial cells."
    ],
    "meningioma": [
        "which is a type of tumor that arises from the meninges, the protective layers surrounding the brain and spinal cord.",
        "Meningiomas develop from the protective coverings of the brain."
    ],
    "pituitary tumor": [
        "The pituitary gland is a small, pea-sized gland located at the base of the brain... It plays a crucial role in regulating various hormones...",
        "This type of tumor affects the pituitary gland located at the base of the brain."
    ]
}

llava_tumor_details_sparse = [
    "The tumor appears to be {size}.",
    "It has a {shape} shape.",
    "The intensity is described as {intensity}.",
    "The orientation of the image is {orientation}.",
    "The tumor is located in the {location}.",
    "It is important to note that the size, shape, intensity, and orientation of the tumor are not always visible in the image itself.",
    "The image provides information about the tumor's location, size, shape, intensity, and orientation."
]

llava_tumor_conclusions = [
    "Further evaluation and consultation with a healthcare professional are necessary to determine the nature of the tumor and the appropriate course of action.",
    "It is important to note that the presence of a tumor in the brain can have various implications and may require further evaluation and treatment.",
    "Without specific details, it is difficult to provide a more comprehensive analysis.",
    "It is important to consult a healthcare professional for a thorough evaluation and proper diagnosis.",
    "A healthcare professional would need to analyze the MRI in detail to provide a comprehensive assessment.",
]

llava_no_tumor_templates = [
    "tumor: no; general_description: The brain MRI image appears to be normal, without any visible tumor.",
    "tumor: no; general_description: The brain MRI image shows no tumor. It is a normal brain MRI, which means that there are no visible abnormalities or tumors in the brain.",
    "tumor: no; general_description: The brain MRI image shows no tumor.",
    "tumor: no; general_description: The brain MRI image appears to be normal, without any visible tumor. It is important to note that a normal brain MRI does not necessarily rule out all possible conditions..."
]

def generate_llava_caption_strict(force_tumor: str):
    data = {"file_name": ""}
    text_parts = []
    tumor = force_tumor

    if tumor == "yes":
        data["tumor"] = "yes"
        data["file_name"] = "tumor_yes"
        tumor_type = random.choice(tumor_types_llava)

        if tumor_type == "pituitary tumor":
            location = random.choice(locations_pit)
        elif tumor_type == "meningioma":
            location = random.choice(locations_mening + locations_llava[2:])
        else:
            location = random.choice(locations_glioma + locations_llava[2:])

        # Intro
        intro = random.choice(llava_tumor_intros)
        text_parts.append(intro.format(tumor_type=tumor_type, location=location))

        # Explanation
        if tumor_type in llava_tumor_explanations:
            text_parts.append(random.choice(llava_tumor_explanations[tumor_type]))

        # 0–2 sparse details
        for detail in random.sample(llava_tumor_details_sparse, k=random.randint(0,2)):
            text_parts.append(detail.format(
                size=random.choice(sizes_llava),
                shape=random.choice(shapes_llava),
                intensity=random.choice(intensities_llava),
                orientation=random.choice(orientations_llava),
                location=location
            ))

        # Conclusion
        text_parts.append(random.choice(llava_tumor_conclusions))
        data["text"] = " ".join(text_parts)

    else:
        data["tumor"] = "no"
        data["file_name"] = "tumor_no"
        data["text"] = random.choice(llava_no_tumor_templates)

    return data

# Generate and save
captions = [generate_llava_caption_strict(flag) for flag in flags]

output_data = [{"file_name": c["file_name"], "text": c["text"]} for c in captions]
with open("generated_captions/generated_llava_captions.json", "w") as f:
    for item in output_data:
        json.dump(item, f)
        f.write("\n")

print(f"Saved {len(captions)} Llava captions (280 no / 720 yes).")