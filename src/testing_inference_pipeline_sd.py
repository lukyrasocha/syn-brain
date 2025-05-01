import argparse
import os
import gc
import torch
import time
import json
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from tqdm import tqdm

def load_jsonl(file_path):
    """Load data from a JSONL file (one JSON object per line)."""
    entries = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line: 
                entries.append(json.loads(line))
    return entries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_root", required=True, help="Path to LoRA checkpoints directory")
    parser.add_argument("--out_root", required=True, help="Path to output images directory")
    parser.add_argument("--metadata_json", required=True, help="Path to metadata JSON/JSONL file containing prompts")
    parser.add_argument("--device", default="cuda", help="Torch device to run on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of diffusion steps")
    args = parser.parse_args()

    # === Load prompts from metadata file ===
    print(f"Loading metadata from: {args.metadata_json}")
    if args.metadata_json.endswith('.jsonl'):
        # Handle JSONL format (one JSON object per line)
        metadata_entries = load_jsonl(args.metadata_json)
    else:
        # Handle regular JSON format
        try:
            with open(args.metadata_json, 'r') as f:
                metadata_entries = json.load(f)
                
            # If metadata is a dict, convert to list of dicts
            if isinstance(metadata_entries, dict):
                metadata_entries = [metadata_entries]
        except json.JSONDecodeError:
            # Fallback to JSONL format if JSON parsing fails
            print("JSON parsing failed, trying JSONL format...")
            metadata_entries = load_jsonl(args.metadata_json)
    
    # Extract prompts and filenames
    prompts = [entry['text'] for entry in metadata_entries]
    filenames = [entry['file_name'] for entry in metadata_entries]
    total_images = len(prompts)
    
    print(f"Loaded {total_images} prompts from metadata file")

    # === Prepare output directory ===
    ckpts = sorted(
        [d for d in os.listdir(args.lora_root) if d.startswith("checkpoint-")],
        key=lambda d: int(d.split("-")[1])
    )
    if not ckpts:
        raise RuntimeError(f"No checkpoints found in {args.lora_root}")
    ckpt = ckpts[-1]
    ckpt_dir = os.path.join(args.lora_root, ckpt)
    print(f"Using checkpoint: {ckpt}")

    dest = os.path.join(args.out_root, ckpt)
    os.makedirs(dest, exist_ok=True)
    print(f"Images will be saved to: {dest}")

    # === Load and configure pipeline ===
    print("Loading Stable Diffusion 1.5 pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    )
    # Use faster scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True
    )
    # Load LoRA weights
    print(f"Loading LoRA weights from: {ckpt_dir}")
    pipe.load_lora_weights(ckpt_dir)

    # Memory optimizations
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("Using xformers memory efficient attention")
    except Exception:
        print("xformers not available")

    # Torch compile if available
    try:
        if hasattr(torch, "compile"):
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
            print("Using torch.compile optimization")
    except Exception:
        print("torch.compile optimization failed or unavailable")

    # Move to device
    pipe = pipe.to(args.device)

    # === Generation loop ===
    start_time = time.time()
    for batch_idx in tqdm(range(0, total_images, args.batch_size), desc="Batches"):
        batch_end = min(batch_idx + args.batch_size, total_images)
        batch_prompts = prompts[batch_idx:batch_end]
        batch_filenames = filenames[batch_idx:batch_end]
        
        # Create a generator per image to vary seeds
        seeds = [args.seed + batch_idx + i for i in range(len(batch_prompts))]
        generators = [torch.Generator(device=args.device).manual_seed(s) for s in seeds]

        # Generate images
        out = pipe(
            batch_prompts,
            height=512,
            width=512,
            num_inference_steps=args.num_inference_steps,
            generator=generators if len(generators) > 1 else generators[0],
        )

        # Save results
        for i, img in enumerate(out.images):
            filename = batch_filenames[i]
            img.save(os.path.join(dest, filename))

        # Free memory
        del out
        gc.collect()
        torch.cuda.empty_cache()

        # Progress logging
        done = batch_end
        elapsed = time.time() - start_time
        ips = done / elapsed if elapsed > 0 else 0
        eta_sec = (total_images - done) / ips if ips > 0 else 0
        print(f"Generated {done}/{total_images} images — {ips:.2f} img/s, ETA {eta_sec/60:.1f} min")

    print(f"✓ All {total_images} images saved under {dest}")

    # Cleanup
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()