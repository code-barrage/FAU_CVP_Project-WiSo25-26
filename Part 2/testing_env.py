import sys
import torch
import numpy as np
import cv2
from PIL import Image

print(f"--- SYSTEM CHECK (ATTEMPT 2) ---")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# --- FIX: Change size to 128x128 (Divisible by 8 for RAFT) ---
dummy_img_np = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
dummy_img_pil = Image.fromarray(dummy_img_np)
cv2.imwrite("dummy_test.jpg", dummy_img_np)
print("CREATED: dummy_test.jpg (512x512)")
print("-" * 30)

# --- TEST 1: OPTICAL FLOW (RAFT) ---
print("\n[1/3] Testing RAFT (Optical Flow)...")
try:
    from torchvision.models.optical_flow import raft_small
    import torchvision.transforms.functional as F
    
    # Load Model
    model_flow = raft_small(pretrained=True, progress=False).eval()
    
    # Prepare Inputs (Batch of 2 identical images)
    img_tensor = F.to_tensor(dummy_img_pil).unsqueeze(0)
    
    # Run Inference
    model_flow(img_tensor, img_tensor)
    print("✅ RAFT Status: WORKING")
except Exception as e:
    print(f"❌ RAFT Status: FAILED ({e})")

# --- TEST 2: DEPTH ESTIMATION (Depth Anything V2) ---
print("\n[2/3] Testing Depth Anything V2...")
try:
    from transformers import pipeline
    depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=-1)
    depth_pipe(dummy_img_pil)
    print("✅ Depth Model Status: WORKING")
except Exception as e:
    print(f"❌ Depth Model Status: FAILED ({e})")

# --- TEST 3: SEGMENTATION (SAM 2) ---
print("\n[3/3] Testing SAM 2 (Tiny)...")
try:
    from ultralytics import SAM
    sam_model = SAM("sam2_t.pt")
    sam_model("dummy_test.jpg", verbose=False)
    print("✅ SAM 2 Status: WORKING")
except Exception as e:
    print(f"❌ SAM 2 Status: FAILED ({e})")

print("\n" + "="*30)