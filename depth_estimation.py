import torch
import cv2
import numpy as np
# import os
from PIL import Image


from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Use GPU if available
def run_midas(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Path to model weights (download if not exists)
    MODEL_PATH = "./weights/dpt_beit_large_512.pt"
    # CACHE_PATH = "~/.cache/torch/hub/checkpoints/dpt_beit_large_512.pt"
    # if not os.path.exists(MODEL_PATH):
    #     import urllib.request
    #     print("Downloading model...")
    #     urllib.request.urlretrieve(
    #         "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt",
    #         MODEL_PATH
    #     )
    # if not os.path.exists(CACHE_PATH):
    #     print("Copying to cache...")
    #     done = os.system(f"cp {MODEL_PATH} {CACHE_PATH}")
    #     print("Model Cached.") if(done == 0) else print("Failed to cache.")

    # Load model
    model = torch.hub.load(
        "isl-org/MiDaS",
        "DPT_BEiT_L_512",
        model_map_location="cuda" if torch.cuda.is_available() else "cpu"
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model.to(device)
    model.eval()

    # Load image
    img = cv2.imread("input.jpg")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = Image.fromarray(img_rgb)

    # Preprocessing (for DPT_BEiT_L_512)
    transform = Compose([
        Resize(512, interpolation=cv2.INTER_CUBIC),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    input_tensor = transform(img_rgb).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Normalize and save
    depth_map = prediction.cpu().numpy()
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_colored = depth_normalized.astype(np.uint8)

    cv2.imwrite("depth_map.png", depth_colored)
    print("Saved: depth_map.png")


if __name__ == "__main__":
    image_path = "input.jpg"
    run_midas(image_path)
