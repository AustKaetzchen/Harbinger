import os
import sys
import cv2
import easyocr
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.stats import mode
import streamlit as st
from streamlit.web import cli as stcli


def buildDiversityEdgeMap (img_rgb, text_mask=None, river_mask=None, alpha_mask=None):
  lab_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
  
  kernel = np.ones((3, 3), np.uint8)
  morph_grad = cv2.morphologyEx(img_rgb, cv2.MORPH_GRADIENT, kernel)
  grad_mag = np.max(morph_grad, axis=2)
  colour_transition_edges = grad_mag > 4
  
  lum = lab_img[:, :, 0]
  lum_blur = cv2.GaussianBlur(lum, (7, 7), 0)
  dark_lines = (lum_blur-lum) > 6
  
  local_mean = cv2.blur(lab_img, (5, 5))
  local_sqr = cv2.blur(lab_img**2, (5, 5))
  local_var = np.maximum(0, local_sqr-local_mean**2)
  local_std = np.sqrt(np.sum(local_var, axis=2))
  diversity_edges = local_std > 6.0
  
  combined_edges = colour_transition_edges|dark_lines|diversity_edges
  
  if text_mask is not None:
    combined_edges[text_mask] = False
  if river_mask is not None:
    combined_edges[river_mask] = False
  if alpha_mask is not None:
    combined_edges[~alpha_mask] = False

  edge_uint8 = combined_edges.astype(np.uint8)
  
  diversity_vis = img_rgb.copy()
  diversity_vis[combined_edges] = [0, 255, 255]
  
  return edge_uint8, Image.fromarray(diversity_vis)


def denoiseMap (img_pil, colour_thresh=15, edge_thresh=20):
  img_np = np.array(img_pil)
  has_alpha = img_np.shape[2] == 4
  img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB if has_alpha else cv2.COLOR_BGR2RGB)
  alpha_channel = img_np[:, :, 3] if has_alpha else np.ones(img_rgb.shape[:2], dtype=np.uint8)*255
  
  filtered = cv2.pyrMeanShiftFiltering(img_rgb, 7, colour_thresh)
  pixels = filtered.reshape(-1, 3)
  
  quant_step = max(1, colour_thresh)
  quantised_pixels = (pixels//quant_step)*quant_step
  quantised_img = quantised_pixels.reshape(img_rgb.shape)
  
  kernel = np.ones((3, 3), np.uint8)
  grad = cv2.morphologyEx(quantised_img, cv2.MORPH_GRADIENT, kernel)
  edge_mask = (np.max(grad, axis=2) > edge_thresh).astype(np.uint8)
  
  num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(1-edge_mask, connectivity=8)
  bin_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
  bin_mask[edge_mask == 1] = True
  
  for label_idx in range(1, num_labels):
    area = stats[label_idx, cv2.CC_STAT_AREA]
    if area < 10:
      bin_mask[labels == label_idx] = True

  valid_sources = (~bin_mask)&(alpha_channel > 0)&(edge_mask == 0)
  if not np.any(valid_sources):
    valid_sources = (alpha_channel > 0)
      
  _, indices = distance_transform_edt(~valid_sources, return_indices=True)
  denoised_rgb = img_rgb[indices[0], indices[1]]
  
  if has_alpha:
    denoised_rgb[alpha_channel == 0] = [0, 0, 0]
    denoised_np = np.dstack((denoised_rgb, alpha_channel))
  else:
    denoised_np = denoised_rgb
    
  return Image.fromarray(denoised_np), bin_mask


def draw ():
  st.set_page_config(page_title="SRG268", layout="wide")
  st.title("(SRG268) Frame Analysis")
  st.write("Segments a map and attempts to create a detailed spatial ontology out ")
  
  uploaded_file = st.sidebar.file_uploader("Choose a map image...", type=["jpg", "jpeg", "png"])
  
  st.sidebar.subheader("Segmentation Adjustments")
  colour_thresh = st.sidebar.slider("Colour Similarity Threshold", min_value=1, max_value=50, value=15, help="Controls colour quantisation density.")
  edge_thresh = st.sidebar.slider("Edge Gradient Threshold", min_value=1, max_value=50, value=20, help="Controls sensitivity of border detection.")
  
  if uploaded_file is not None:
    reader_obj = loadReader()
    
    col1, col2, col3, col4 = st.columns(4)
    col5, col6, col7, col8 = st.columns(4)
    col9, col10, col11, col12 = st.columns(4)
    
    with col1:
      st.subheader("1. Original Map")
      st.image(uploaded_file, use_container_width=True)
      
    with st.spinner("Extracting OCR, mapping geometry, and resolving enclaves..."):
      masked_img, composite_img, preview_img, ocr_results, edge_vis_pil, text_mask, river_mask, total_edges = processMap(uploaded_file, reader_obj, colour_thresh, edge_thresh)
      
    with col2:
      st.subheader("2. UI Masking")
      st.image(preview_img, use_container_width=True)
      
    with col3:
      st.subheader("3. Denoised Image")
      st.image(masked_img, use_container_width=True)
      
    with col4:
      st.subheader("4. Sharpness Layer")
      st.image(edge_vis_pil, use_container_width=True)
      
    with col5:
      st.subheader("5. Semantic Features")
      st.image(composite_img, use_container_width=True)
      
    with col6:
      st.subheader("6. Denoised Edges")
      final_img_np = np.array(masked_img.convert("RGB"))
      masked_np = np.array(masked_img)
      alpha_mask = masked_np[:, :, 3] > 0 if masked_np.shape[2] == 4 else np.ones(final_img_np.shape[:2], dtype=bool)
      
      diversity_edges, diversity_vis_pil = buildDiversityEdgeMap(final_img_np, text_mask, river_mask, alpha_mask)
      st.image(diversity_vis_pil, use_container_width=True)

    with col7:
      st.subheader("7. First-pass Segmentation")
      with st.spinner("Merging regions bounded by diversity edges..."):
        refined_masks = segmentAndMergeRegions(final_img_np, diversity_edges, alpha_mask, max_colour_diff=6.0)
        segmentation_vis = renderMasks(final_img_np, refined_masks)
      st.image(segmentation_vis, use_container_width=True)

    with col8:
      st.subheader("8. First-pass Filtering")
      l1 = np.zeros(final_img_np.shape[:2], dtype=np.int32)
      for i, mask_dict in enumerate(refined_masks):
        l1[mask_dict["segmentation"]] = i+1
          
      h_diff = l1[:, :-1] != l1[:, 1:]
      v_diff = l1[:-1, :] != l1[1:, :]
      s1_edges = np.zeros_like(l1, dtype=bool)
      
      s1_edges[:, :-1][h_diff] = True
      s1_edges[:-1, :][v_diff] = True
      
      plot8_img = (final_img_np*0.35).astype(np.uint8)
      s1_edges_vis = cv2.dilate(s1_edges.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
      plot8_img[s1_edges_vis] = [255, 0, 255]
      st.image(Image.fromarray(plot8_img), use_container_width=True)

    with col9:
      st.subheader("9. Edge Restoration")
      
      real_text_mask = np.zeros(final_img_np.shape[:2], dtype=np.uint8)
      for bbox, text, prob in ocr_results:
        box_pts = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(real_text_mask, [box_pts], 255)
      
      real_text_mask = cv2.dilate(real_text_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))) > 0
      raw_edges = total_edges&(~real_text_mask)
      
      q_step = 10
      q_img = (final_img_np//q_step)*q_step
      img_hashes = q_img[:, :, 0].astype(np.int64)*65536+q_img[:, :, 1].astype(np.int64)*256+q_img[:, :, 2].astype(np.int64)
      
      edge_pixels_hash = img_hashes[raw_edges]
      if len(edge_pixels_hash) > 0:
        unique_hashes, counts = np.unique(edge_pixels_hash, return_counts=True)
        valid_hashes = unique_hashes[counts > 10]
        colour_filtered_edges = raw_edges&np.isin(img_hashes, valid_hashes)
      else:
        colour_filtered_edges = raw_edges

      num_labels_edges, labels_edges, stats_edges, _ = cv2.connectedComponentsWithStats(colour_filtered_edges.astype(np.uint8), connectivity=8)
      valid_external_edges = np.zeros_like(colour_filtered_edges)
      
      for i in range(1, num_labels_edges):
        area = stats_edges[i, cv2.CC_STAT_AREA]
        if area >= 12:
          valid_external_edges[labels_edges == i] = True
                  
      plot9_img = (final_img_np*0.35).astype(np.uint8)
      plot9_img[valid_external_edges] = [0, 255, 255]
      st.image(Image.fromarray(plot9_img), use_container_width=True)

    with col10:
      st.subheader("10. Second-pass Segmentation")
      with st.spinner("Bisecting segments with validated external features and seeding enclaves..."):
        second_pass_masks = []
        
        bisect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bisect_edges = cv2.dilate(valid_external_edges.astype(np.uint8), bisect_kernel).astype(bool)
        
        covered_by_second_pass = np.zeros(final_img_np.shape[:2], dtype=bool)

        for i in range(1, len(refined_masks)+1):
          mask_i = (l1 == i)
          mask_i_bisected = mask_i&(~bisect_edges)
          
          num_sub, sub_labels, sub_stats, _ = cv2.connectedComponentsWithStats(mask_i_bisected.astype(np.uint8), connectivity=4)
          
          for j in range(1, num_sub):
            area = sub_stats[j, cv2.CC_STAT_AREA]
            if area >= 20: 
              sub_mask = (sub_labels == j)
              second_pass_masks.append({"segmentation": sub_mask, "area": area})
              covered_by_second_pass |= sub_mask

        unassigned_enclaves = alpha_mask&(~covered_by_second_pass)&(~bisect_edges)
        num_enc, enc_labels, enc_stats, _ = cv2.connectedComponentsWithStats(unassigned_enclaves.astype(np.uint8), connectivity=4)
        
        for k in range(1, num_enc):
          area = enc_stats[k, cv2.CC_STAT_AREA]
          if area >= 20:
            enc_mask = (enc_labels == k)
            second_pass_masks.append({"segmentation": enc_mask, "area": area})

        segmentation_vis_2 = renderMasks(final_img_np, second_pass_masks)
      st.image(segmentation_vis_2, use_container_width=True)

    with col11:
      st.subheader("11. First-pass kNN Repair")
      with st.spinner("Binning bisected gap pixels to nearest second-pass neighbour..."):
        repaired_masks = repairSegmentation(final_img_np, refined_masks, second_pass_masks, alpha_mask, k_neighbors=5)
        segmentation_vis_3 = renderMasks(final_img_np, repaired_masks)
      st.image(segmentation_vis_3, use_container_width=True)

    with col12:
      st.subheader("12. Second-pass kNN Repair")
      with st.spinner("Binning all diversity edges to nearest segment..."):
        final_masks = repairEdgeGaps(final_img_np, repaired_masks, alpha_mask, k_neighbors=5)
        segmentation_vis_4 = renderMasks(final_img_np, final_masks)
      st.image(segmentation_vis_4, use_container_width=True)
      st.success(f"Generated {len(final_masks)} fully repaired masks.")


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
        cv2.drawContours(mask_rects_pad, [approx], -1, 255, -1)

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
  alpha_channel = img_np[:, :, 3] if has_alpha else np.ones(img_rgb.shape[:2], dtype=np.uint8)*255
  
  ocr_results_post = reader_obj.readtext(img_rgb)
  text_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
  
  for bbox, _, _ in ocr_results_post:
    box_pts = np.array(bbox, dtype=np.int32)
    poly_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    cv2.fillPoly(poly_mask, [box_pts], 255)
    
    dilated_poly = cv2.dilate(poly_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    text_mask = text_mask|(dilated_poly > 0)
    
  if np.any(text_mask):
    valid_sources = (~text_mask)&(alpha_channel > 0)&(~edge_cache)
    if not np.any(valid_sources):
      valid_sources = (~text_mask)&(alpha_channel > 0)
    _, indices = distance_transform_edt(~valid_sources, return_indices=True)
    img_rgb = img_rgb[indices[0], indices[1]]
    edge_cache = edge_cache|text_mask 
    
  kernel_river = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  opened_rgb = cv2.morphologyEx(img_rgb, cv2.MORPH_OPEN, kernel_river)
  closed_rgb = cv2.morphologyEx(img_rgb, cv2.MORPH_CLOSE, kernel_river)
  
  diff_open = cv2.absdiff(img_rgb, opened_rgb)
  diff_close = cv2.absdiff(img_rgb, closed_rgb)
  river_mask = (np.max(diff_open, axis=2) > edge_thresh)|(np.max(diff_close, axis=2) > edge_thresh)
  
  if np.any(river_mask):
    valid_sources = (~river_mask)&(alpha_channel > 0)&(~edge_cache)
    if not np.any(valid_sources):
      valid_sources = (~river_mask)&(alpha_channel > 0)
    _, indices = distance_transform_edt(~valid_sources, return_indices=True)
    img_rgb = img_rgb[indices[0], indices[1]]
    edge_cache = edge_cache|river_mask  
    
  flat_rgb = cv2.pyrMeanShiftFiltering(img_rgb, 15, 40)
  flat_rgb = cv2.medianBlur(flat_rgb, 7)
  
  num_labels, labels = cv2.connectedComponents((~edge_cache&(alpha_channel > 0)).astype(np.uint8), connectivity=8)
  segmented_rgb = np.zeros_like(flat_rgb)
  
  for label_idx in range(1, num_labels):
    mask = (labels == label_idx)
    if np.any(mask):
      avg_colour = np.mean(flat_rgb[mask], axis=0).astype(np.uint8)
      segmented_rgb[mask] = avg_colour
      
  edge_vis = segmented_rgb.copy()
  edge_vis[edge_cache] = [255, 0, 255]
      
  valid_sources = (~edge_cache)&(alpha_channel > 0)
  if not np.any(valid_sources):
    valid_sources = (alpha_channel > 0)
  _, indices = distance_transform_edt(~valid_sources, return_indices=True)
  flat_rgb = segmented_rgb[indices[0], indices[1]]
  
  if has_alpha:
    flat_rgb[alpha_channel == 0] = [0, 0, 0]
    edge_vis[alpha_channel == 0] = [0, 0, 0]
    out_np = np.dstack((flat_rgb, alpha_channel))
  else:
    out_np = flat_rgb
    
  return Image.fromarray(out_np), Image.fromarray(edge_vis), text_mask, river_mask, edge_cache


def processMap (image_file, reader_obj, colour_thresh=15, edge_thresh=20):
  img_pil = Image.open(image_file).convert("RGBA")
  img_np_rgb = np.array(img_pil.convert("RGB"))
  ocr_results = reader_obj.readtext(img_np_rgb)
  
  img_pil, preview_pil = maskInfoboxes(img_pil)
  denoised_pil, edge_cache = denoiseMap(img_pil, colour_thresh, edge_thresh)
  final_pil, edge_vis_pil, text_mask, river_mask, total_edges = postProcessMap(denoised_pil, reader_obj, edge_cache, edge_thresh)
  
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
  
  return masked_img, composite_img, preview_pil, ocr_results, edge_vis_pil, text_mask, river_mask, total_edges


def renderMasks (image, masks, max_background_ratio=0.7):
  if len(masks) == 0:
    return np.zeros_like(image)

  h, w = image.shape[:2]
  total_pixels = h*w
  filtered_anns = [ann for ann in masks if (ann["area"]/total_pixels) < max_background_ratio]
  if len(filtered_anns) == 0:
    filtered_anns = masks

  sorted_anns = sorted(filtered_anns, key=(lambda x: x["area"]), reverse=True)
  mask_img = np.zeros((h, w, 3), dtype=np.uint8)
  
  for ann in sorted_anns:
    m = ann["segmentation"]
    colour_mask = np.random.randint(0, 256, (3,), dtype=np.uint8)
    mask_img[m] = colour_mask

  return mask_img


def repairEdgeGaps (img_rgb, masks, alpha_mask=None, k_neighbors=5):
  if len(masks) == 0:
    return masks

  h, w = img_rgb.shape[:2]
  l3_map = np.zeros((h, w), dtype=np.int32)
  for idx, ann in enumerate(masks):
    l3_map[ann["segmentation"]] = idx+1

  if alpha_mask is not None:
    l3_map[~alpha_mask] = 0

  if alpha_mask is not None:
    unassigned_mask = alpha_mask&(l3_map == 0)
  else:
    unassigned_mask = (l3_map == 0)

  assigned_mask = l3_map > 0

  if not np.any(unassigned_mask) or not np.any(assigned_mask):
    return masks

  lab_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
  y_coords, x_coords = np.indices((h, w))
  spatial_scale = 0.2
  features = np.dstack((lab_img, y_coords*spatial_scale, x_coords*spatial_scale))

  train_features = features[assigned_mask]
  train_labels = l3_map[assigned_mask]
  query_features = features[unassigned_mask]

  tree = cKDTree(train_features)
  _, indices = tree.query(query_features, k=k_neighbors)

  if k_neighbors == 1:
    repaired_labels = train_labels[indices]
  else:
    neighbor_labels = train_labels[indices]
    mode_res, _ = mode(neighbor_labels, axis=1, keepdims=False)
    repaired_labels = np.squeeze(mode_res)

  repaired_l3_map = l3_map.copy()
  repaired_l3_map[unassigned_mask] = repaired_labels

  final_masks = []
  for idx, ann in enumerate(masks):
    lbl = idx+1
    m = (repaired_l3_map == lbl)
    if alpha_mask is not None:
      m &= alpha_mask
    area = int(np.sum(m))
    if area >= 20:
      final_masks.append({"segmentation": m, "area": area})

  return final_masks


def repairSegmentation (img_rgb, first_pass_masks, second_pass_masks, alpha_mask=None, k_neighbors=5):
  if len(second_pass_masks) == 0:
    return second_pass_masks

  h, w = img_rgb.shape[:2]
  
  l1_mask = np.zeros((h, w), dtype=bool)
  for ann in first_pass_masks:
    l1_mask |= ann["segmentation"]

  l2_map = np.zeros((h, w), dtype=np.int32)
  for idx, ann in enumerate(second_pass_masks):
    l2_map[ann["segmentation"]] = idx+1

  if alpha_mask is not None:
    l1_mask &= alpha_mask
    l2_map[~alpha_mask] = 0

  unassigned_mask = l1_mask&(l2_map == 0)
  assigned_mask = l2_map > 0

  if not np.any(unassigned_mask) or not np.any(assigned_mask):
    return second_pass_masks

  lab_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
  y_coords, x_coords = np.indices((h, w))
  spatial_scale = 0.2
  features = np.dstack((lab_img, y_coords*spatial_scale, x_coords*spatial_scale))

  train_features = features[assigned_mask]
  train_labels = l2_map[assigned_mask]
  query_features = features[unassigned_mask]

  tree = cKDTree(train_features)
  _, indices = tree.query(query_features, k=k_neighbors)

  if k_neighbors == 1:
    repaired_labels = train_labels[indices]
  else:
    neighbor_labels = train_labels[indices]
    mode_res, _ = mode(neighbor_labels, axis=1, keepdims=False)
    repaired_labels = np.squeeze(mode_res)

  repaired_l2_map = l2_map.copy()
  repaired_l2_map[unassigned_mask] = repaired_labels

  repaired_masks = []
  for idx, ann in enumerate(second_pass_masks):
    lbl = idx+1
    m = (repaired_l2_map == lbl)
    if alpha_mask is not None:
      m &= alpha_mask
    area = int(np.sum(m))
    if area >= 20:
      repaired_masks.append({"segmentation": m, "area": area})

  return repaired_masks


def segmentAndMergeRegions (img_rgb, edge_mask, alpha_mask=None, max_colour_diff=6.0):
  lab_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
  inv_edges = (1-edge_mask).astype(np.uint8)
  
  if alpha_mask is not None:
    inv_edges[~alpha_mask] = 0

  num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv_edges, connectivity=4)

  region_means = {}
  for i in range(1, num_labels):
    m = labels == i
    if alpha_mask is not None:
      m = m&alpha_mask
    if np.sum(m) > 10:
      region_means[i] = np.mean(lab_img[m], axis=0)

  merged_labels = {i: i for i in region_means}

  def findRoot (i):
    path = []
    while merged_labels[i] != i:
      path.append(i)
      i = merged_labels[i]
    for node in path:
      merged_labels[node] = i
    return i

  h_diff_pairs = (labels[:, :-1] != labels[:, 1:])&(labels[:, :-1] > 0)&(labels[:, 1:] > 0)
  pairs_h = np.column_stack((labels[:, :-1][h_diff_pairs], labels[:, 1:][h_diff_pairs]))

  v_diff_pairs = (labels[:-1, :] != labels[1:, :])&(labels[:-1, :] > 0)&(labels[1:, :] > 0)
  pairs_v = np.column_stack((labels[:-1, :][v_diff_pairs], labels[1:, :][v_diff_pairs]))

  pairs = np.vstack((pairs_h, pairs_v))
  valid_pairs = pairs[(pairs[:, 0] > 0)&(pairs[:, 1] > 0)]
  unique_pairs = np.unique(valid_pairs, axis=0) if len(valid_pairs) > 0 else np.empty((0, 2), dtype=np.int32)

  for u, v in unique_pairs:
    ru, rv = findRoot(u), findRoot(v)
    if ru != rv and ru in region_means and rv in region_means:
      dist = np.linalg.norm(region_means[ru]-region_means[rv])
      if dist < max_colour_diff:
        merged_labels[rv] = ru

  refined_masks = []
  final_groups = {}
  for i in region_means:
    root = findRoot(i)
    final_groups.setdefault(root, []).append(i)

  for root, group in final_groups.items():
    combined_mask = np.isin(labels, group)
    if alpha_mask is not None:
      combined_mask = combined_mask&alpha_mask
    area = int(np.sum(combined_mask))
    if area >= 20:
      refined_masks.append({"segmentation": combined_mask, "area": area})

  return refined_masks


if __name__ == "__main__":
  is_running = st.runtime.exists()
  draw() if is_running else initApp()