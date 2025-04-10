import argparse
import torch
from diffusers import StableDiffusionPipeline

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion models.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-naruto-model",
        help="Directory containing the fine-tuned model.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a man in a blue shirt with glasses",
        help="Prompt to generate images from.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the pre-trained Stable Diffusion model
    stable_diffusion_model = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path)
    stable_diffusion_model.to(device)

    # Generate an image using the pre-trained model
    image_stable_diffusion = stable_diffusion_model(args.prompt).images[0]

    # Load the fine-tuned model from the specified directory
    fine_tuned_model = StableDiffusionPipeline.from_pretrained(args.output_dir)
    fine_tuned_model.to(device)

    # Generate an image using the fine-tuned model
    image_fine_tuned = fine_tuned_model(args.prompt).images[0]

    # Save the generated images
    image_stable_diffusion.save("image_stable_diffusion.png")
    image_fine_tuned.save("image_fine_tuned.png")

if __name__ == "__main__":
    main()
