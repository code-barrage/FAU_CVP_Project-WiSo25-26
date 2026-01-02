import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import Compose, Resize, Normalize, ToTensor

# 1. SETUP: Force usage of CPU
DEVICE = 'cpu'
print(f"Running on: {DEVICE} (Intel i5 12th Gen)")

# 2. LOAD MODEL (Small Version)
# We load the model architecture directly from the hub to save you manual cloning
# Note: This requires internet on first run to get the config
try:
    model = torch.hub.load('LiheYoung/Depth-Anything', 'depth_anything_vits14', pretrained=True).to(DEVICE)
except:
    print("Error loading from Hub. Ensure you have internet access for the first run.")

model.eval()

# 3. PREPARE TRANSFORM PIPELINE
transform = Compose([
    Resize((518, 518)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 4. RUN INFERENCE
def predict_depth(image_path):
    # Load Image
    raw_image = cv2.imread(image_path)
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    
    # Preprocess
    h, w = image.shape[:2]
    image_tensor = transform(cv2.resize(raw_image, (518, 518))).unsqueeze(0).to(DEVICE)

    # Infer
    with torch.no_grad():
        depth = model(image_tensor)
    
    # Resize back to original resolution
    depth = torch.nn.functional.interpolate(depth.unsqueeze(1), size=(h, w), mode='bicubic', align_corners=False).squeeze()
    depth = depth.cpu().numpy()

    # Visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Predicted Depth Map")
    plt.imshow(depth, cmap='inferno') # 'inferno' makes depth pop nicely
    plt.axis('off')
    
    # Save output for your PPT
    output_name = f"result_{image_path}"
    plt.savefig(output_name)
    print(f"Saved result to {output_name}")
    plt.show()

# Run on your images
predict_depth('my_illusion_image.jpg')