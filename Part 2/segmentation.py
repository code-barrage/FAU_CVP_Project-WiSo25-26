from ultralytics import SAM
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# --- CONFIGURATION ---
IMAGE_NAME = "triangle.jpg"  # CHANGE THIS to your image filename
# ---------------------

def run_sam_experiment():
    # 1. Setup
    print("Loading SAM 2 (Tiny)...")
    # This automatically loads 'sam2_t.pt'
    model = SAM("sam2_t.pt")

    # 2. Check Image
    if not os.path.exists(IMAGE_NAME):
        print(f"Error: {IMAGE_NAME} not found.")
        return

    # 3. Inference
    print("Running segmentation...")
    # conf=0.25 lowers the threshold, encouraging the model to 'guess' more
    results = model(IMAGE_NAME, device='cpu', conf=0.25)

    # 4. Visualization
    for result in results:
        # Ultralytics provides a nice plotter that overlays masks
        im_bgr = result.plot() 
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 10))
        plt.imshow(im_rgb)
        plt.title(f"SAM 2 Segmentation: {IMAGE_NAME}")
        plt.axis('off')
        
        output_file = f"result_sam_{IMAGE_NAME.split('.')[0]}.png"
        plt.savefig(output_file)
        print(f"Saved visualization to: {output_file}")
        plt.show()

if __name__ == "__main__":
    run_sam_experiment()