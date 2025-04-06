import argparse
import torch
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.conversation import conv_templates, SeparatorStyle


def main(args):
    print("STARTING FUCKING SCRIPT")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = get_model_name_from_path(args.model_path)

    # Load model
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name,
        load_8bit=args.load_8bit, load_4bit=args.load_4bit,
        device=device
    )

    # Load image
    image = Image.open(args.image_file).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [im.to(device, dtype=torch.float16) for im in image_tensor]
    else:
        image_tensor = image_tensor.to(device, dtype=torch.float16)


    if model.config.mm_use_im_start_end:
        prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + args.question
    else:
        prompt = DEFAULT_IMAGE_TOKEN + '\n' + args.question
    conv_mode = "mistral_instruct"
    conv = conv_templates[conv_mode].copy()
    #prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + args.question
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    print("HELLO")
    answer = tokenizer.decode(output_ids[0, input_ids.shape[1]:])
    print(answer)
    print(f" Answer: {answer.strip()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    main(args)
