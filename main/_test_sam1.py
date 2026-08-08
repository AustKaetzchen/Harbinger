import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def cleanupMemory (sam_model=None, masks=None):
    import gc
    import torch
    import matplotlib.pyplot as plt

    # Close all open matplotlib figures to release CPU RAM
    plt.close('all')

    # Delete large objects if passed
    if masks is not None:
        del masks
    if sam_model is not None:
        del sam_model

    # Force Python garbage collection
    gc.collect()

    # Release cached PyTorch GPU memory back to the CUDA driver
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"GPU memory cleared. Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

def loadAndPrepareImage (image_path, max_dimension=2048):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image from {image_path}. Please check the path.")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Resized image to {new_w}x{new_h} to maintain high detail while keeping memory usage under control.")
    else:
        print(f"Keeping full image resolution: {w}x{h}.")
        
    return image

def generateOptimizedMasks (image, sam_model):
    import torch
    from segment_anything import SamAutomaticMaskGenerator

    # Free up unallocated memory from model loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mask_generator = SamAutomaticMaskGenerator(
        model=sam_model,
        points_per_side=48, # Dense point grid balanced for 16GB VRAM
        points_per_batch=16, # Reduced batch size to prevent PyTorch tensor allocation spike
        pred_iou_thresh=0.82, 
        stability_score_thresh=0.88,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        crop_overlap_ratio=512/1500,
        crop_nms_thresh=0.7,
    )

    # Updated autocast syntax to eliminate deprecation warning
    with torch.amp.autocast('cuda'):
        masks = mask_generator.generate(image)

    return masks

def plotMasks (image, masks, max_background_ratio=0.7):
    if len(masks) == 0:
        print("No masks generated.")
        return

    h, w = image.shape[:2]
    total_pixels = h * w
    
    # Filter out giant background canvas masks (>70% image area) that cause double-background fill
    filtered_anns = [ann for ann in masks if (ann['area'] / total_pixels) < max_background_ratio]
    if len(filtered_anns) == 0:
        filtered_anns = masks

    # Sort remaining region masks from largest to smallest
    sorted_anns = sorted(filtered_anns, key=(lambda x: x['area']), reverse=True)
    
    # Composite clean 4-channel mask buffer
    mask_img = np.zeros((h, w, 4), dtype=np.float32)
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.random.random(3)
        mask_img[m, 0:3] = color_mask
        mask_img[m, 3] = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    
    # 1. Image with single transparent mask overlay
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[0].set_title('Original Image', fontsize=16)
    
    # 2. Pure masks rendered cleanly on dark background
    black_bg = np.zeros((h, w, 3), dtype=np.uint8)
    axes[1].imshow(black_bg)
    axes[1].imshow(mask_img)
    axes[1].axis('off')
    axes[1].set_title('Masks Only', fontsize=16)
    
    plt.tight_layout()
    plt.show()

def refineMasksWithEdgesAndColor (masks, image, low_threshold=50, high_threshold=150, color_diff_threshold=25.0):
    # Convert image to grayscale for edge detection and CIELAB for perceptual color separation
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    kernel = np.ones((2, 2), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    refined_masks = []
    
    for ann in masks:
        mask = ann['segmentation'].copy()
        
        # 1. Zero out pixels overlapping with map border outlines
        mask[dilated_edges > 0] = False
        
        if np.sum(mask) < 20:
            continue
            
        # 2. Extract LAB color values of all pixels inside the current mask
        mask_pixels = lab_image[mask].astype(np.float32)
        color_std = np.std(mask_pixels, axis=0)
        
        # Check if color variance inside mask is high (indicates merged distinct colors like red & blue)
        candidate_masks = []
        if np.linalg.norm(color_std) > 15.0:
            # Run 2-cluster K-Means inside the mask
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            compactness, kmeans_labels, centers = cv2.kmeans(mask_pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Calculate Delta E color distance between the two cluster centers
            color_dist = np.linalg.norm(centers[0] - centers[1])
            
            if color_dist >= color_diff_threshold:
                # Color difference is high: split mask into two color sub-masks
                indices = np.where(mask)
                for k in range(2):
                    sub_bin_mask = np.zeros_like(mask, dtype=bool)
                    sub_bin_mask[(indices[0][kmeans_labels.ravel() == k], indices[1][kmeans_labels.ravel() == k])] = True
                    candidate_masks.append(sub_bin_mask)
            else:
                candidate_masks.append(mask)
        else:
            candidate_masks.append(mask)
            
        # 3. Separate spatial components for each color-pure mask candidate
        for cand_mask in candidate_masks:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cand_mask.astype(np.uint8))
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < 15:
                    continue
                
                sub_mask = (labels == i)
                new_ann = ann.copy()
                new_ann['segmentation'] = sub_mask
                new_ann['area'] = int(area)
                refined_masks.append(new_ann)
                
    return refined_masks

def main ():
    import os
    from torch.hub import download_url_to_file

    image_path = 'sample01.jpg'
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    expected_size = 2564550879
    
    # Check if file is missing or incompletely downloaded
    if not os.path.exists(sam_checkpoint) or os.path.getsize(sam_checkpoint) < expected_size:
        if os.path.exists(sam_checkpoint):
            os.remove(sam_checkpoint)
        print("Downloading SAM vit_h checkpoint (2.56 GB)... Please wait.")
        checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        download_url_to_file(checkpoint_url, sam_checkpoint)
        print("Download completed successfully!")

    image = loadAndPrepareImage(image_path, max_dimension=2048)
    
    print("Loading SAM model...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device=device)

    try:
        print("Generating SAM dense masks...")
        raw_masks = generateOptimizedMasks(image, sam)
        
        print("Refining masks using map edge detection and LAB color separation...")
        refined_masks = refineMasksWithEdgesAndColor(raw_masks, image)
        
        print(f"Generated {len(refined_masks)} color and edge-aligned masks. Rendering visualization...")
        plotMasks(image, refined_masks)
    finally:
        print("Cleaning up GPU and CPU memory...")
        cleanupMemory(sam_model=sam, masks=raw_masks if 'raw_masks' in locals() else None)

if __name__ == '__main__':
    main()