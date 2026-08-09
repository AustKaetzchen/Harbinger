import os
import sys
import cv2
import easyocr
import numpy as np
import random
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
import streamlit as st
from streamlit.web import cli as stcli
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from torch.hub import download_url_to_file


def denoiseMap (img_pil, color_thresh=15, edge_thresh=20):
  img_np = np.array(img_pil)
  has_alpha = img_np.shape[2] == 4
  img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB if has_alpha else cv2.COLOR_BGR2RGB)
  
  # Uses color_thresh to determine how aggressively to merge similar colors spatially
  filtered = cv2.pyrMeanShiftFiltering(img_rgb, 7, color_thresh)
  
  pixels = filtered.reshape(-1, 3)
  
  # Uses color_thresh as the quantization step to control color binning
  quant_step = max(1, color_thresh)
  quantised_pixels = (pixels // quant_step) * quant_step
  quantised_img = quantised_pixels.reshape(img_rgb.shape)
  
  kernel = np.ones((3, 3), np.uint8)
  grad = cv2.morphologyEx(quantised_img, cv2.MORPH_GRADIENT, kernel)
  
  # Uses edge_thresh to control how distinct a boundary needs to be to be treated as a segment edge
  edge_mask = (np.max(grad, axis=2) > edge_thresh).astype(np.uint8)
  
  num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(1-edge_mask, connectivity=8)
  
  bin_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
  bin_mask[edge_mask == 1] = True
  
  for label_idx in range(1, num_labels):
    area = stats[label_idx, cv2.CC_STAT_AREA]
    if area < 10:
      bin_mask[labels == label_idx] = True
      
  _, indices = distance_transform_edt(bin_mask, return_indices=True)
  denoised_rgb = img_rgb[indices[0], indices[1]]
  
  if has_alpha:
    denoised_np = np.dstack((denoised_rgb, img_np[:, :, 3]))
  else:
    denoised_np = denoised_rgb
    
  return Image.fromarray(denoised_np), bin_mask


def draw ():
  st.set_page_config(page_title="SRG268", layout="wide")
  st.title("(SRG268) Frame Analysis")
  st.write("Upload a map image to extract text, mask legends/infoboxes, view bounding box overlays, and generate SAM masks.")
  
  uploaded_file = st.sidebar.file_uploader("Choose a map image...", type=["jpg", "jpeg", "png"])
  
  st.sidebar.subheader("Segmentation Adjustments")
  color_thresh = st.sidebar.slider("Color Similarity Threshold", min_value=1, max_value=50, value=15, help="Decrease this to preserve boundaries between similarly colored regions (e.g. drop to 5-10 for subtle boundaries).")
  edge_thresh = st.sidebar.slider("Edge Gradient Threshold", min_value=1, max_value=50, value=20, help="Decrease this to make edge detection more sensitive to low-contrast borders.")
  
  if uploaded_file is not None:
    reader_obj = loadReader()
    
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col5, col6 = st.columns(2)
    
    with col1:
      st.subheader("1. Original Map")
      st.image(uploaded_file, use_container_width=True)
      
    with st.spinner("Extracting OCR, mapping geometry, and resolving enclaves..."):
      masked_img, composite_img, preview_img, ocr_results, edge_vis_pil = processMap(uploaded_file, reader_obj, color_thresh, edge_thresh)
      
    with col2:
      st.subheader("2. Infobox Candidates")
      st.image(preview_img, use_container_width=True)
      
    with col3:
      st.subheader("3. Post-Process Edges")
      st.image(edge_vis_pil, use_container_width=True)
      
    with col4:
      st.subheader("4. OCR Labelling")
      st.image(composite_img, use_container_width=True)
      
    with col5:
      st.subheader("5. Denoised Image")
      st.image(masked_img, use_container_width=True)
      
    with col6:
      st.subheader("6. Segmentation")
      with st.spinner("Loading SAM model..."):
        sam_model = loadSamModel()
        
        # Take the final processed PIL image from Step 5 and prep for segmentation
        final_img_np = np.array(masked_img.convert("RGB"))
        
        # Keep SAM memory under control for large images uploaded in Streamlit
        h, w = final_img_np.shape[:2]
        max_dimension = 2048
        if max(h, w) > max_dimension:
          scale = max_dimension / max(h, w)
          new_h, new_w = int(h * scale), int(w * scale)
          final_img_np = cv2.resize(final_img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

      with st.spinner("Generating SAM dense masks..."):
        raw_masks = generateOptimisedMasks(final_img_np, sam_model)
        
      with st.spinner("Refining masks using map edge detection and LAB colour separation..."):
        refined_masks = refineMasksWithEdgesAndColour(raw_masks, final_img_np)
        segmentation_vis = renderMasks(final_img_np, refined_masks)
          
      st.image(segmentation_vis, use_container_width=True)
      st.success(f"Successfully generated {len(refined_masks)} masks.")
        
      # Release memory after processing the upload
      del raw_masks
      if torch.cuda.is_available():
          torch.cuda.empty_cache()


def generateOptimisedMasks (image, sam_model):
  # Free up unallocated memory from model loading
  if torch.cuda.is_available():
    torch.cuda.empty_cache()

  mask_generator = SamAutomaticMaskGenerator(
    model=sam_model,
    points_per_side=48,  # Dense point grid balanced for 16GB VRAM
    points_per_batch=16,  # Reduced batch size to prevent PyTorch tensor allocation spike
    pred_iou_thresh=0.82,
    stability_score_thresh=0.88,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    crop_overlap_ratio=512/1500,
    crop_nms_thresh=0.7,
  )

  # Updated autocast syntax to eliminate deprecation warning
  if torch.cuda.is_available():
    with torch.amp.autocast("cuda"):
      masks = mask_generator.generate(image)
  else:
    masks = mask_generator.generate(image)

  return masks


def initApp ():
  sys.argv = ["streamlit", "run", __file__]
  sys.exit(stcli.main())


def loadFont (font_size=14):
  font_paths = [
    "arial.ttf",
    "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/Library/Fonts/Arial.ttf"
  ]
  for path in font_paths:
    try:
      return ImageFont.truetype(path, font_size)
    except OSError:
      continue
  return ImageFont.load_default()


@st.cache_resource
def loadReader ():
  return easyocr.Reader(["en", "ru"])


@st.cache_resource
def loadSamModel ():
  sam_checkpoint = "sam_vit_h_4b8939.pth"
  model_type = "vit_h"
  expected_size = 2564550879

  # Check if file is missing or incompletely downloaded
  if (
    not os.path.exists(sam_checkpoint)
    or os.path.getsize(sam_checkpoint) < expected_size
  ):
    if os.path.exists(sam_checkpoint):
      os.remove(sam_checkpoint)
    st.info("Downloading SAM vit_h checkpoint (2.56 GB)... Please wait.")
    checkpoint_url = (
      "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    )
    download_url_to_file(checkpoint_url, sam_checkpoint)
    st.success("Download completed successfully!")

  sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  sam.to(device=device)
  
  return sam


def maskInfoboxes (img_pil):
  img_np = np.array(img_pil)
  img_h, img_w = img_np.shape[:2]
  total_area = img_h*img_w
  
  img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
  
  pad = 10
  img_pad = cv2.copyMakeBorder(img_rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
  
  contours_pool = []
  
  gray_pad = cv2.cvtColor(img_pad, cv2.COLOR_RGB2GRAY)
  for t1, t2 in [(30, 100), (50, 150), (100, 200)]:
    edges = cv2.Canny(gray_pad, t1, t2)
    closed = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_pool.extend(cnts)

  thresh1 = cv2.adaptiveThreshold(gray_pad, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
  thresh2 = cv2.adaptiveThreshold(gray_pad, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
  for th in [thresh1, thresh2]:
    cnts, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_pool.extend(cnts)

  for bin_size in [16, 32]:
    quantised = (img_pad.astype(np.int32)//bin_size)*bin_size
    quantised = quantised.astype(np.uint8)
    pixels = quantised.reshape(-1, 3)
    colours, counts = np.unique(pixels, axis=0, return_counts=True)
    
    top_colours = colours[np.argsort(-counts)][:20]
    
    for colour in top_colours:
      lower = np.clip(colour, 0, 255).astype(np.uint8)
      upper = np.clip(colour+bin_size-1, 0, 255).astype(np.uint8)
      colour_mask = cv2.inRange(quantised, lower, upper)
      
      colour_mask = cv2.morphologyEx(colour_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
      cnts, _ = cv2.findContours(colour_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
      contours_pool.extend(cnts)

  mask_rects_pad = np.zeros(img_pad.shape[:2], dtype=np.uint8)
  
  for cnt in contours_pool:
    x, y, w, h = cv2.boundingRect(cnt)
    bbox_area = w*h
    
    if bbox_area < (total_area*0.001) or bbox_area > (total_area*0.50):
      continue
      
    c_area = cv2.contourArea(cnt)
    if bbox_area == 0:
      continue
      
    solidity = c_area/bbox_area
    
    if solidity > 0.88:
      peri = cv2.arcLength(cnt, True)
      approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
      if len(approx) <= 8:
        cv2.rectangle(mask_rects_pad, (x, y), (x+w, y+h), 255, -1)

  mask_rects = mask_rects_pad[pad:-pad, pad:-pad]

  close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
  mask_closed = cv2.morphologyEx(mask_rects, cv2.MORPH_CLOSE, close_kernel)
  
  pad_e = 5
  mask_closed_pad = cv2.copyMakeBorder(mask_closed, pad_e, pad_e, pad_e, pad_e, cv2.BORDER_CONSTANT, value=255)
  
  holes_mask = cv2.bitwise_not(mask_closed_pad)
  hole_cnts, _ = cv2.findContours(holes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
  mask_rects_resolved_pad = cv2.copyMakeBorder(mask_rects, pad_e, pad_e, pad_e, pad_e, cv2.BORDER_CONSTANT, value=0)
  
  for enclave_cnt in hole_cnts:
    if cv2.contourArea(enclave_cnt) < (total_area*0.03):
      hull = cv2.convexHull(enclave_cnt)
      cv2.drawContours(mask_rects_resolved_pad, [hull], 0, 255, -1)
                
  mask_rects = mask_rects_resolved_pad[pad_e:-pad_e, pad_e:-pad_e]

  final_contours, _ = cv2.findContours(mask_rects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
  preview_img = (img_rgb*0.35).astype(np.uint8)
  
  dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
  final_mask = cv2.dilate(mask_rects, dilate_kernel, iterations=1)
  
  for cnt in final_contours:
    rand_colour = (random.randint(80, 255), random.randint(80, 255), random.randint(80, 255))
    overlay = preview_img.copy()
    
    cv2.drawContours(overlay, [cnt], 0, rand_colour, -1)
    cv2.addWeighted(overlay, 0.5, preview_img, 0.5, 0, preview_img)
    cv2.drawContours(preview_img, [cnt], 0, rand_colour, 2)
    
    x, y, w, h = cv2.boundingRect(cnt)
    text = "Masked UI"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    tx = x+(w-text_size[0])//2
    ty = y+(h+text_size[1])//2
    cv2.rectangle(preview_img, (tx-2, ty-text_size[1]-2), (tx+text_size[0]+2, ty+2), (0, 0, 0), -1)
    cv2.putText(preview_img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

  img_np[final_mask == 255] = (0, 0, 0, 0)
  
  return Image.fromarray(img_np), Image.fromarray(preview_img)


def postProcessMap (img_pil, reader_obj, edge_cache, edge_thresh=20):
  img_np = np.array(img_pil)
  has_alpha = img_np.shape[2] == 4
  img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB if has_alpha else cv2.COLOR_BGR2RGB)
  
  # Step 1: Secondary OCR pass for residual text
  ocr_results_post = reader_obj.readtext(img_rgb)
  text_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
  
  for bbox, text, prob in ocr_results_post:
    box_pts = np.array(bbox, dtype=np.int32)
    cv2.fillPoly(text_mask, [box_pts], True)
    
  if np.any(text_mask):
    _, indices = distance_transform_edt(text_mask, return_indices=True)
    img_rgb = img_rgb[indices[0], indices[1]]
    edge_cache = edge_cache | text_mask 
    
  # Step 2: Detect thin linear features (rivers) via morphological operations
  kernel_river = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  opened_rgb = cv2.morphologyEx(img_rgb, cv2.MORPH_OPEN, kernel_river)
  closed_rgb = cv2.morphologyEx(img_rgb, cv2.MORPH_CLOSE, kernel_river)
  
  diff_open = cv2.absdiff(img_rgb, opened_rgb)
  diff_close = cv2.absdiff(img_rgb, closed_rgb)
  
  # Using the exposed edge_thresh to preserve narrow colored features if dialed low
  river_mask = (np.max(diff_open, axis=2) > edge_thresh) | (np.max(diff_close, axis=2) > edge_thresh)
  
  if np.any(river_mask):
    _, indices = distance_transform_edt(river_mask, return_indices=True)
    img_rgb = img_rgb[indices[0], indices[1]]
    edge_cache = edge_cache | river_mask  
    
  # Step 3: Inner-segment grain smoothing via high-spatial mean shift and median filtering
  flat_rgb = cv2.pyrMeanShiftFiltering(img_rgb, 15, 40)
  flat_rgb = cv2.medianBlur(flat_rgb, 7)
  
  # Step 4: Reimpose exact boundaries & repair bleeding
  num_labels, labels = cv2.connectedComponents((~edge_cache).astype(np.uint8), connectivity=8)
  segmented_rgb = np.zeros_like(flat_rgb)
  
  # Flatten interior component colours 
  for label_idx in range(num_labels):
    mask = (labels == label_idx)
    if np.any(mask):
      avg_color = np.mean(flat_rgb[mask], axis=0).astype(np.uint8)
      segmented_rgb[mask] = avg_color
      
  # Visualisation image for edges overlay on the flat segments
  edge_vis = segmented_rgb.copy()
  edge_vis[edge_cache] = [255, 0, 255]
      
  # Resolve isolated edge boundaries with closest solid neighbour colours
  _, indices = distance_transform_edt(edge_cache, return_indices=True)
  flat_rgb = segmented_rgb[indices[0], indices[1]]
  
  if has_alpha:
    out_np = np.dstack((flat_rgb, img_np[:, :, 3]))
  else:
    out_np = flat_rgb
    
  return Image.fromarray(out_np), Image.fromarray(edge_vis)


def processMap (image_file, reader_obj, color_thresh=15, edge_thresh=20):
  img_pil = Image.open(image_file).convert("RGBA")
  
  # 1. OCR (Strictly processed first)
  img_np_rgb = np.array(img_pil.convert("RGB"))
  ocr_results = reader_obj.readtext(img_np_rgb)
  
  # 2. Masking
  img_pil, preview_pil = maskInfoboxes(img_pil)
  
  # 3. Denoising thin networks
  denoised_pil, edge_cache = denoiseMap(img_pil, color_thresh, edge_thresh)
  
  # 4. Post-processing: second OCR text removal, river filtering & inner-segment flattening
  final_pil, edge_vis_pil = postProcessMap(denoised_pil, reader_obj, edge_cache, edge_thresh)
  
  # 5. Draw OCR bounds
  overlay_img = Image.new("RGBA", final_pil.size, (0, 0, 0, 0))
  draw_obj = ImageDraw.Draw(overlay_img)
  font_obj = loadFont(14)
  
  for bbox, text, prob in ocr_results:
    box_pts = [(int(pt[0]), int(pt[1])) for pt in bbox]
    draw_obj.polygon(box_pts, fill=(255, 255, 0, 80), outline=(255, 0, 0, 220))
    
    label_text = f"{text} ({prob:.2f})"
    min_x = min(pt[0] for pt in box_pts)
    min_y = min(pt[1] for pt in box_pts)
    
    text_pos = (min_x, max(0, min_y-16))
    text_box = draw_obj.textbbox(text_pos, label_text, font=font_obj)
    
    draw_obj.rectangle(text_box, fill=(255, 255, 255, 200))
    draw_obj.text(text_pos, label_text, fill=(0, 0, 0, 255), font=font_obj)
    
  composite_img = Image.alpha_composite(final_pil, overlay_img)
  masked_img = final_pil
  
  return masked_img, composite_img, preview_pil, ocr_results, edge_vis_pil


def refineMasksWithEdgesAndColour (
  masks,
  image,
  low_threshold=50,
  high_threshold=150,
  colour_diff_threshold=25.0,
):
  # Convert image to greyscale for edge detection and CIELAB for perceptual colour separation
  grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

  blurred = cv2.GaussianBlur(grey, (3, 3), 0)
  edges = cv2.Canny(blurred, low_threshold, high_threshold)

  kernel = np.ones((2, 2), np.uint8)
  dilated_edges = cv2.dilate(edges, kernel, iterations=1)

  refined_masks = []

  for ann in masks:
    mask = ann["segmentation"].copy()

    # 1. Zero out pixels overlapping with map border outlines
    mask[dilated_edges > 0] = False

    if np.sum(mask) < 20:
      continue

    # 2. Extract LAB colour values of all pixels inside the current mask
    mask_pixels = lab_image[mask].astype(np.float32)
    colour_std = np.std(mask_pixels, axis=0)

    # Check if colour variance inside mask is high (indicates merged distinct colours like red and blue)
    candidate_masks = []
    if np.linalg.norm(colour_std) > 15.0:
      # Run 2-cluster K-Means inside the mask
      criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
      compactness, kmeans_labels, centres = cv2.kmeans(
        mask_pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
      )

      # Calculate Delta E colour distance between the two cluster centres
      colour_dist = np.linalg.norm(centres[0] - centres[1])

      if colour_dist >= colour_diff_threshold:
        # Colour difference is high: split mask into two colour sub-masks
        indices = np.where(mask)
        for k in range(2):
          sub_bin_mask = np.zeros_like(mask, dtype=bool)
          sub_bin_mask[(
            indices[0][kmeans_labels.ravel() == k],
            indices[1][kmeans_labels.ravel() == k],
          )] = True
          candidate_masks.append(sub_bin_mask)
      else:
        candidate_masks.append(mask)
    else:
      candidate_masks.append(mask)

    # 3. Separate spatial components for each colour-pure mask candidate
    for cand_mask in candidate_masks:
      num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cand_mask.astype(np.uint8)
      )
      for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 15:
          continue

        sub_mask = labels == i
        new_ann = ann.copy()
        new_ann["segmentation"] = sub_mask
        new_ann["area"] = int(area)
        refined_masks.append(new_ann)

  return refined_masks


def renderMasks (image, masks, max_background_ratio=0.7):
  if len(masks) == 0:
    return np.zeros_like(image)

  h, w = image.shape[:2]
  total_pixels = h * w

  # Filter out giant background canvas masks (>70% image area) that cause double-background fill
  filtered_anns = [
    ann for ann in masks if (ann["area"] / total_pixels) < max_background_ratio
  ]
  if len(filtered_anns) == 0:
    filtered_anns = masks

  # Sort remaining region masks from largest to smallest
  sorted_anns = sorted(filtered_anns, key=(lambda x: x["area"]), reverse=True)

  # Pure masks rendered cleanly on dark background
  mask_img = np.zeros((h, w, 3), dtype=np.uint8)
  for ann in sorted_anns:
    m = ann["segmentation"]
    colour_mask = np.random.randint(0, 256, (3,), dtype=np.uint8)
    mask_img[m] = colour_mask

  return mask_img


if __name__ == "__main__":
  is_running = st.runtime.exists()
  draw() if is_running else initApp()