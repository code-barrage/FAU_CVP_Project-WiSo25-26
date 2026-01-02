import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import SAM
import numpy as np

# --- CONFIGURATION ---
INPUT_FOLDER = "seg_experiments"   # Folder where your 3.1.jpg, 3.2.jpg files are
OUTPUT_FOLDER = "seg_results"      # Folder where results will be saved

# MAPPING: Numbered Files -> Gestalt Theme
# Ensure your files are named exactly 3.1.jpg, 3.2.jpg, etc.
TEST_CASES = {
    "3.1.jpg":  "Closure (Kanizsa Triangle)",
    "3.2.jpg":  "Closure (Kanizsa Square)",
    "3.3.jpg":  "Emergence (Dalmatian Dog)",
    "3.4.jpg":  "Figure-Ground (Camouflage)",
    "3.5.jpg":  "Continuity (Occluded Banana)",
    "3.6.jpg":  "Bistability (Rubin's Vase)",
    "3.7.jpg":  "Grouping (Pointillism)",
    "3.8.jpg":  "Over-segmentation (Mosaic Bird)",
    "3.9.jpg":  "Emergence (Dalmatian Dog) - Alternate",
    "3.10.jpg": "Pareidolia (Cloud Face)"
}

def run_gestalt_numeric():
    # 1. Setup Folders
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Error: Folder '{INPUT_FOLDER}' not found.")
        return

    # 2. Load SAM 2 (Tiny)
    print(f"Loading SAM 2 Tiny Model...")
    model = SAM("sam2_t.pt") 

    print(f"Starting Gestalt Stress Test on {len(TEST_CASES)} images...\n")

    # 3. Iterate through Test Cases
    for filename, test_theme in TEST_CASES.items():
        image_path = os.path.join(INPUT_FOLDER, filename)
        
        # Check if file exists (handle both .jpg and .png just in case)
        if not os.path.exists(image_path):
            # Try .png if .jpg is missing
            alt_path = image_path.replace(".jpg", ".png")
            if os.path.exists(alt_path):
                image_path = alt_path
            else:
                print(f"⚠️ Skipping {filename}: File not found.")
                continue

        print(f"Processing: {filename} [{test_theme}]...")

        # 4. Run Inference
        # conf=0.25 is standard. 
        results = model(image_path, device='cpu', conf=0.25, verbose=False)

        # 5. Visualization
        for result in results:
            annotated_img = result.plot() 
            annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            # Load original
            original_img = cv2.imread(image_path)
            if original_img is None:
                print(f"Error loading image {filename}")
                continue
            original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

            # Create Figure
            fig, ax = plt.subplots(1, 2, figsize=(14, 7))
            
            # Left: Original
            ax[0].imshow(original_rgb)
            ax[0].set_title(f"Original: {filename}", fontsize=12)
            ax[0].axis('off')
            
            # Right: SAM 2 Result
            ax[1].imshow(annotated_rgb)
            ax[1].set_title(f"SAM 2 Result: {test_theme}", fontsize=12)
            ax[1].axis('off')

            # Save
            save_name = f"result_{filename.replace('.jpg', '')}.png"
            save_path = os.path.join(OUTPUT_FOLDER, save_name)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            print(f"  ✅ Saved result to: {save_path}")

    print("\nAll experiments completed.")

if __name__ == "__main__":
    run_gestalt_numeric()