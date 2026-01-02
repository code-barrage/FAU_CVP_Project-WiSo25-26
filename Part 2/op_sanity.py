import torch
from torchvision.models.optical_flow import raft_small
from torchvision.utils import flow_to_image
import torchvision.transforms.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
FRAME_1 = "0.4.png"  # The starting position
FRAME_2 = "0.6.png"  # The moved position
# ---------------------

def verify_real_motion():
    # 1. Setup
    device = "cpu"
    print(f"Loading RAFT Small on {device}...")
    model = raft_small(pretrained=True, progress=False).to(device)
    model.eval()

    # 2. Check Files
    if not os.path.exists(FRAME_1) or not os.path.exists(FRAME_2):
        print("Error: Please provide 'frame1.jpg' and 'frame2.jpg'")
        return

    # 3. Preprocess Function (Resize + Normalize)
    def preprocess(img_path):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        img = img.resize((new_w, new_h))
        tensor = F.to_tensor(img)
        # Normalize to [-1, 1] for RAFT
        return (tensor * 2.0) - 1.0

    print("Processing images...")
    img1 = preprocess(FRAME_1).unsqueeze(0).to(device)
    img2 = preprocess(FRAME_2).unsqueeze(0).to(device)

    # 4. Run Inference (Frame 1 -> Frame 2)
    with torch.no_grad():
        list_of_flows = model(img1, img2)
        predicted_flow = list_of_flows[-1][0] # Shape: [2, H, W]

    # 5. Scientific Metrics
    # Calculate Magnitude per pixel: sqrt(x^2 + y^2)
    flow_magnitude = torch.norm(predicted_flow, p=2, dim=0)
    avg_motion = flow_magnitude.mean().item()
    max_motion = flow_magnitude.max().item()

    print(f"-" * 30)
    print(f"Average Motion: {avg_motion:.4f} pixels")
    print(f"Max Motion:     {max_motion:.4f} pixels")
    
    if avg_motion > 0.5:
        print("✅ SUCCESS: Significant motion detected.")
    else:
        print("⚠️ WARNING: Motion is very low. Are the images too similar?")
    print(f"-" * 30)

    # 6. Visualization
    # Convert flow to standard RGB visualization
    flow_img = flow_to_image(predicted_flow).permute(1, 2, 0).numpy()
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Show Frame 1
    ax[0].imshow(Image.open(FRAME_1).resize((img1.shape[3], img1.shape[2])))
    ax[0].set_title("Frame 1 (Start)")
    ax[0].axis('off')
    
    # Show Frame 2
    ax[1].imshow(Image.open(FRAME_2).resize((img1.shape[3], img1.shape[2])))
    ax[1].set_title("Frame 2 (End)")
    ax[1].axis('off')
    
    # Show Flow
    ax[2].imshow(flow_img)
    ax[2].set_title(f"Detected Motion\n(Avg: {avg_motion:.2f} px)")
    ax[2].axis('off')

    plt.savefig("verification_result.png")
    print("Saved result to 'verification_result.png'")
    plt.show()

if __name__ == "__main__":
    verify_real_motion()