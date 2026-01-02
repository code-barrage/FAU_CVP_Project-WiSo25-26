import torch
from torchvision.models.optical_flow import raft_small
from torchvision.utils import flow_to_image
import torchvision.transforms.functional as F
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# --- CONFIGURATION ---
IMAGE_NAME = "1.9 - Copy.jpg"  # CHANGE THIS to your image filename
# ---------------------

def run_flow_experiment():
    # 1. Setup
    device = "cpu"
    print(f"Loading RAFT Small on {device}...")
    model = raft_small(pretrained=True, progress=False).to(device)
    model.eval()

    # 2. Load & Preprocess Image
    if not os.path.exists(IMAGE_NAME):
        print(f"Error: {IMAGE_NAME} not found. Please add it to this folder.")
        return

    img_raw = Image.open(IMAGE_NAME).convert("RGB")
    
    # RESIZE LOGIC: Dimensions must be divisible by 8 for RAFT
    w, h = img_raw.size
    new_w = (w // 8) * 8
    new_h = (h // 8) * 8
    img_resized = img_raw.resize((new_w, new_h))
    
    # Create Input Tensor (Batch size 1)
    img_tensor = F.to_tensor(img_resized).unsqueeze(0).to(device)

    # 3. The Experiment: Static Input (Frame 1 == Frame 2)
    print("Running inference (Static Illusion Test)...")
    with torch.no_grad():
        # We pass the SAME image twice. 
        # Ideally, flow should be 0. If it's not, the model is 'fooled'.
        list_of_flows = model(img_tensor, img_tensor)
        predicted_flow = list_of_flows[-1][0] # Final prediction

    # 4. Analysis
    # Calculate average motion magnitude (The "Hallucination Score")
    flow_magnitude = torch.norm(predicted_flow, p=2, dim=0)
    avg_mag = flow_magnitude.mean().item()
    print(f"Average Flow Magnitude: {avg_mag:.4f} (0.0 means perfect stability)")

    # 5. Visualization
    flow_img = flow_to_image(predicted_flow).permute(1, 2, 0).numpy()
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 7))
    
    ax[0].imshow(img_resized)
    ax[0].set_title("Input (Static Image)")
    ax[0].axis('off')
    
    ax[1].imshow(flow_img)
    ax[1].set_title(f"Predicted Flow (Avg Mag: {avg_mag:.2f})")
    ax[1].axis('off')
    plt.tight_layout()
    output_file = f"result_flow_{IMAGE_NAME}.png"
    plt.savefig(output_file)
    print(f"Saved visualization to: {output_file}")
    plt.show()

if __name__ == "__main__":
    run_flow_experiment()