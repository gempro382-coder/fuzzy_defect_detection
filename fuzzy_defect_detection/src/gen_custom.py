import os
import cv2
import numpy as np
from dataset_generator import SurfaceFactory, DefectPainter, GenConfig

def apply_beautiful_design(img_gray):
    # apply a beautiful colormap
    color = cv2.applyColorMap(img_gray, cv2.COLORMAP_PARULA)
    # add some geometric shapes (beautiful design)
    h, w = img_gray.shape
    for _ in range(5):
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        r = np.random.randint(20, 80)
        c = (int(np.random.randint(150, 255)), int(np.random.randint(150, 255)), int(np.random.randint(150, 255)))
        cv2.circle(color, (cx, cy), r, c, 2)
    return color

def apply_cartoon(img_gray):
    color = cv2.applyColorMap(img_gray, cv2.COLORMAP_AUTUMN)
    cartoon = cv2.stylization(color, sigma_s=60, sigma_r=0.07)
    return cartoon

def generate():
    cfg = GenConfig()
    rng = np.random.default_rng(42)
    surface = SurfaceFactory(cfg, rng)
    painter = DefectPainter(rng)

    # Save to upload folder so user can easily test them or just to a custom folder
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "custom_samples")
    os.makedirs(out_dir, exist_ok=True)
    
    classes = ["good", "scratch", "crack", "dent", "corrosion"]
    
    print(f"Generating 50 images in {out_dir}...")
    for cls in classes:
        os.makedirs(os.path.join(out_dir, cls), exist_ok=True)
        for i in range(10):
            # i=0: standard mid severity
            # i=1..5: beautiful
            # i=6..9: cartoon
            style = "standard"
            if i >= 1 and i <= 5: style = "beautiful"
            elif i > 5: style = "cartoon"
            
            img_gray = surface.make(256)
            sev = 0.55 # mid severity moderate
            
            if cls != "good":
                img_gray = painter.paint(cls, img_gray, sev)
                
            img_gray = surface._sensor_effects(img_gray)
            img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)
            
            if style == "standard":
                final = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            elif style == "beautiful":
                final = apply_beautiful_design(img_gray)
            else:
                final = apply_cartoon(img_gray)
                
            out_name = f"{cls}_{style}_{i}.png"
            cv2.imwrite(os.path.join(out_dir, cls, out_name), final)
            
    print("Done generating 50 custom samples.")

if __name__ == '__main__':
    generate()
