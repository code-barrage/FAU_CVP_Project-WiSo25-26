from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
IMAGE_NAME = "2.3.png"   # CHANGE THIS to your image filename
# ---------------------

def run_depth_experiment():
    # 1. Setup
    print("Loading Depth Anything V2 (Small)...")
    # device=-1 ensures CPU usage
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=-1)

    # 2. Load Image
    if not os.path.exists(IMAGE_NAME):
        print(f"Error: {IMAGE_NAME} not found.")
        return
    
    image = Image.open(IMAGE_NAME).convert("RGB")

    # 3. Inference
    print("Estimating depth...")
    result = pipe(image)
    depth_map = result["depth"]

    # 4. Visualization
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    # 'magma' is excellent for depth: Black=Far, Orange/White=Near
    ax[1].imshow(depth_map, cmap='inferno')
    ax[1].set_title("Predicted Depth Map")
    ax[1].axis('off')
    plt.tight_layout()
    output_file = f"result_depth_{IMAGE_NAME}.png"
    plt.savefig(output_file)
    print(f"Saved visualization to: {output_file}")
    plt.show()

if __name__ == "__main__":
    run_depth_experiment()