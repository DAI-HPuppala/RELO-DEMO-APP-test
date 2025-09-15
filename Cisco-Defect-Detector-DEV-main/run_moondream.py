import torch
from transformers import AutoModelForCausalLM
from PIL import Image
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run Moondream demo")
    parser.add_argument("--img-path", required=True, help="Path to your image file")
    parser.add_argument("--object", default="face", help="Object to detect and point")
    args = parser.parse_args()

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Moondream model with GPU acceleration
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-01-09",
        trust_remote_code=True,
        device_map="auto"
    )
    model.to(device)

    # Load image
    image = Image.open(args.img_path)

    # 1. Image Captioning
    print("Short caption:")
    caption = model.caption(image, length="short")["caption"]
    print(caption)

    # 2. Detailed caption
    print("\nDetailed caption:")
    for token in model.caption(image, length="normal", stream=True)["caption"]:
        print(token, end="", flush=True)
    print()

    # 3. Visual Question Answering
    print("\nVisual Question Answering:")
    answer = model.query(image, "How many people are in the image?")["answer"]
    print(answer)

    # 4. Object Detection
    print("\nObject Detection:")
    objects = model.detect(image, args.object)["objects"]
    print(f"Found {len(objects)} {args.object}(s)")

    # 5. Visual Pointing
    print("\nVisual Pointing:")
    points = model.point(image, args.object)["points"]
    print(f"Found {len(points)} {args.object}(s)")


if __name__ == "__main__":
    main()
