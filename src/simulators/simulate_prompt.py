import json
import random
import re

SEED = 42
NUM_SAMPLES = 1000
NO_TUMOR_RATE = 0.28

random.seed(SEED)

num_no  = int(NUM_SAMPLES * NO_TUMOR_RATE)   # 280
num_yes = NUM_SAMPLES - num_no               # 720
flags   = ["no"] * num_no + ["yes"] * num_yes
random.shuffle(flags)

def get_random_elements(data_list, count=1):
    if not data_list:
        return []
    return random.sample(data_list, min(count, len(data_list)))

# ---------- GEMMA ----------

gemma_data = {
    "location": ["pituitary", "temporal lobe", "frontal lobe", "left frontal lobe"],
    "size": ["medium", "large", "small"],
    "shape": ["round", "irregular"],
    "intensity": ["hypointense", "mixed", "hyperintense", "bright", "low", "heterogeneous"],
    "orientation": ["saggital", "axial", "coronal"],
    "mri_types": ["MRI", "Coronal T1-weighted MRI", "Axial T2-weighted MRI", "Axial T1-weighted MRI", "Sagittal MRI"],
    "ct_types": ["Axial CT scan"],
    "normal_structs": ["Brain structures", "Ventricles", "brain parenchyma", "gray and white matter differentiation", "midline structures", "optic chiasm"],
    "tumor_effects": ["compressing", "displacing surrounding structures", "mass effect", "surrounding edema", "Ventricular compression"],
    "tumor_guesses": ["pituitary adenoma", "meningioma", "glioma"]
}

def generate_gemma_caption(force_tumor=None):
    tumor_flag = force_tumor if force_tumor is not None else ("no" if random.random() < NO_TUMOR_RATE else "yes")
    data = {"file_name": ""}
    cp = {}

    if tumor_flag == "no":
        data["file_name"] = "tumor_no"
        if random.random() < 0.1:
            scan_type = random.choice(gemma_data["ct_types"])
            finding = random.choice([
                "dense hemorrhage within the left basal ganglia and thalamus, likely indicating an intracerebral hemorrhage",
                "no acute intracranial findings",
                "age-related calcifications"
            ])
            desc = f"{scan_type} shows {finding}."
        else:
            mri = random.choice(gemma_data["mri_types"])
            s = get_random_elements(gemma_data["normal_structs"], 3)
            desc = (
                f"{mri} shows normal brain anatomy with clear {s[0]} and well-defined {s[1]}, "
                f"and {s[2]}."
            )
        text = f"tumor: no; general_description: {desc}"

    else:
        data["file_name"] = "tumor_yes"
        cp["location"]    = random.choice(gemma_data["location"])
        cp["size"]        = random.choice(gemma_data["size"])
        cp["shape"]       = random.choice(gemma_data["shape"])
        cp["intensity"]   = random.choice(gemma_data["intensity"])
        cp["orientation"] = random.choice(gemma_data["orientation"])

        mri = random.choice(gemma_data["mri_types"])
        parts = [f"{mri} shows a {cp['size']}, {cp['shape']} lesion in the {cp['location']}."]
        if random.random() < 0.7:
            parts.append(f"The lesion shows {cp['intensity']} intensity.")
        if random.random() < 0.5:
            parts.append(f"This likely represents a {random.choice(gemma_data['tumor_guesses'])}.")
        if random.random() < 0.6:
            parts.append(f"There is evidence of {random.choice(gemma_data['tumor_effects'])}.")
        parts.append(f"{random.choice(gemma_data['normal_structs'])} appear otherwise normal.")
        desc = " ".join(parts)

        text = (
            "tumor: yes; "
            f"location: {cp['location']}; size: {cp['size']}; shape: {cp['shape']}; "
            f"intensity: {cp['intensity']}; orientation: {cp['orientation']}; "
            f"general description: {desc}"
        )

    data["text"] = text.replace("..", ".").strip()
    return data

# ---------- GEMINI ----------

gemini_data = {
    "location": ["pituitary", "brain parenchyma", "frontal lobe", "right frontal lobe", "right occipital lobe"],
    "size": ["medium", "small", "large"],
    "shape": ["round", "irregular", "rounded"],
    "intensity": ["high", "heterogeneous", "hyperintense", "mixed", "heterogenous"],
    "orientation": ["saggital", "coronal", "axial", "sagittal"],
    "mri_types": ["Coronal T1-weighted MRI post contrast", "Axial brain MRI", "Axial T2 FLAIR sequence", "Coronal brain MRI", "Axial T2-weighted MRI", "Axial T1-weighted MRI", "Axial FLAIR MRI", "Sagittal T1-weighted MRI"],
    "normal_structs": ["brain parenchyma", "overall brain structure", "ventricles", "sulci", "gray and white matter differentiation", "midline structures", "cerebral hemispheres"],
    "other_findings": ["periventricular and subcortical white matter hyperintensities", "chronic small vessel ischemic disease", "dural enhancement", "mild ventriculomegaly"],
    "tumor_effects": ["edema", "distorted anatomical structure", "mass effect", "heterogeneous enhancement", "surrounding edema", "displaced ventricles"],
    "tumor_guesses": ["glioma", "pituitary adenoma", "similar tumor"]
}

def generate_gemini_caption(force_tumor=None):
    tumor_flag = force_tumor if force_tumor is not None else ("no" if random.random() < NO_TUMOR_RATE else "yes")
    data = {"file_name": ""}
    cp = {}

    if tumor_flag == "no":
        data["file_name"] = "tumor_no"
        mri = random.choice(gemini_data["mri_types"])
        if random.random() < 0.25:
            finding = random.choice(gemini_data["other_findings"])
            desc = f"{mri} shows {finding}."
        else:
            s = get_random_elements(gemini_data["normal_structs"], 3)
            desc = f"{mri} shows normal {s[0]}, {s[1]}, and {s[2]}."
        text = f"tumor: no; general_description: {desc}"

    else:
        data["file_name"] = "tumor_yes"
        cp["location"]    = random.choice(gemini_data["location"])
        cp["size"]        = random.choice(gemini_data["size"])
        cp["shape"]       = random.choice(gemini_data["shape"])
        cp["intensity"]   = random.choice(gemini_data["intensity"])
        cp["orientation"] = random.choice(gemini_data["orientation"])

        mri = random.choice(gemini_data["mri_types"])
        guess = random.choice(gemini_data["tumor_guesses"])
        parts = [f"{mri} shows a {cp['size']}, {cp['shape']} {cp['intensity']} lesion in the {cp['location']}, suspicious for a {guess}."]
        if random.random() < 0.4:
            parts.append(f"Associated {random.choice(gemini_data['tumor_effects'])} is noted.")
        parts.append("The rest of the brain appears normal.")
        desc = " ".join(parts)

        text = (
            "tumor: yes; "
            f"location: {cp['location']}; size: {cp['size']}; shape: {cp['shape']}; "
            f"intensity: {cp['intensity']}; orientation: {cp['orientation']}; "
            f"general description: {desc}"
        )

    data["text"] = text.replace("..", ".").strip()
    return data

# ---------- OVIS ----------

ovis_data = {
    "location": ["right parietal lobe", "left parietal lobe", "sella turcica", "left hemisphere", "left frontal lobe", "right frontal lobe", "left temporal lobe", "right temporal lobe", "pituitary region", "pituitary"],
    "size": ["small", "medium", "large"],
    "shape": ["irregular", "round"],
    "intensity": ["hyperintense", "mixed", "none"],
    "orientation": ["axial", "sagittal", "coronal"],
    "mri_types": ["Axial MRI scan", "Axial brain MRI", "brain MRI"],
    "tumor_effects": ["surrounding edema", "midline shift", "mass effect"],
    "normal_structs": ["symmetrical cerebral hemispheres", "clear ventricles", "symmetrical brain structures", "normal brain structures", "normal brain parenchyma", "ventricles"]
}

def generate_ovis_caption(force_tumor=None):
    tumor_flag = force_tumor if force_tumor is not None else ("no" if random.random() < NO_TUMOR_RATE else "yes")
    data = {"file_name": ""}
    mri = random.choice(ovis_data["mri_types"])
    s = get_random_elements(ovis_data["normal_structs"], 2)

    if tumor_flag == "no":
        data["file_name"] = "tumor_no"
        desc = f"{mri} showing {s[0]} and {s[1]}, with no visible abnormalities."
        text = f"tumor: no; general_description: {desc}"

    else:
        data["file_name"] = "tumor_yes"
        loc = random.choice(ovis_data["location"])
        size = random.choice(ovis_data["size"])
        shape= random.choice(ovis_data["shape"])
        inten= random.choices(ovis_data["intensity"], weights=[0.8,0.1,0.1])[0]
        ori  = random.choice(ovis_data["orientation"])
        eff  = random.choice(ovis_data["tumor_effects"])

        desc = (
            f"{mri} shows a {size}, {shape}, {inten} tumor in the {loc} region, "
            f"with {eff}. The image is a {ori} view. No other significant abnormalities are apparent."
        )
        text = (
            "tumor: yes; "
            f"location: {loc}; size: {size}; shape: {shape}; intensity: {inten}; "
            f"orientation: {ori}; general description: {desc}"
        )

    data["text"] = text.replace("..", ".").strip()
    return data

vlms = {
    "gemma": generate_gemma_caption,
    "gemini": generate_gemini_caption,
    "ovis":  generate_ovis_caption,
}

for name, fn in vlms.items():
    caps = [fn(flag) for flag in flags]
    out = [{"file_name": c["file_name"], "text": c["text"]} for c in caps]
    with open(f"generated_captions/generated_{name}_captions.json", "w") as f:
        for item in out:
            json.dump(item, f)
            f.write("\n")
    print(f"Saved {len(caps)} captions for {name} (280 no / 720 yes).")