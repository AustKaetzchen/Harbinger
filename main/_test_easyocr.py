import sys
import cv2
import easyocr
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
import streamlit as st
from streamlit.web import cli as stcli

def denoiseMap (img_pil):
  img_np = np.array(img_pil)
  has_alpha = img_np.shape[2] == 4
  img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB if has_alpha else cv2.COLOR_BGR2RGB)
  
  filtered = cv2.pyrMeanShiftFiltering(img_rgb, 7, 15)
  
  pixels = filtered.reshape(-1, 3)
  quantised_pixels = (pixels//16)*16
  quantised_img = quantised_pixels.reshape(img_rgb.shape)
  
  kernel = np.ones((3, 3), np.uint8)
  grad = cv2.morphologyEx(quantised_img, cv2.MORPH_GRADIENT, kernel)
  edge_mask = (np.max(grad, axis=2) > 20).astype(np.uint8)
  
  num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(1-edge_mask, connectivity=8)
  
  bin_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
  bin_mask[edge_mask == 1] = True
  
  for label_idx in range(1, num_labels):
    area = stats[label_idx, cv2.CC_STAT_AREA]
    if area < 30:
      bin_mask[labels == label_idx] = True
      
  _, indices = distance_transform_edt(bin_mask, return_indices=True)
  denoised_rgb = img_rgb[indices[0], indices[1]]
  
  if has_alpha:
    denoised_np = np.dstack((denoised_rgb, img_np[:, :, 3]))
  else:
    denoised_np = denoised_rgb
    
  return Image.fromarray(denoised_np)

def draw ():
  st.set_page_config(page_title="Map Text Extraction Tool", layout="wide")
  st.title("Map Text Extraction Visualisation")
  st.write("Upload a map image to extract text, mask legends/infoboxes, and view semi-transparent bounding box overlays.")
  
  uploaded_file = st.sidebar.file_uploader("Choose a map image...", type=["jpg", "jpeg", "png"])
  
  if uploaded_file is not None:
    reader_obj = loadReader()
    
    img1, img2 = st.columns(2)
    img3, img4 = st.columns(2)
    
    with img1:
      st.subheader("1. Original Map")
      st.image(uploaded_file, use_container_width=True)
      
    with st.spinner("Extracting OCR, mapping geometry, and resolving enclaves..."):
      masked_img, composite_img, preview_img, ocr_results = processMap(uploaded_file, reader_obj)
      
    with img2:
      st.subheader("2. Segmentation Candidates")
      st.image(preview_img, use_container_width=True)
      
    with img3:
      st.subheader("3. Annotated & Masked Output")
      st.image(composite_img, use_container_width=True)
      
    with img4:
      st.subheader("4. Final image")
      st.image(masked_img, use_container_width=True)
      
    st.subheader("Extracted Text Results")
    if ocr_results:
      results_data = []
      for bbox, text, prob in ocr_results:
        results_data.append({"Text": text, "Confidence": f"{prob*100:.2f}%"})
      st.table(results_data)
    else:
      st.info("No text detected in the uploaded map.")

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

def postProcessMap (img_pil, reader_obj):
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
    
  # Step 2: Detect thin linear features (rivers) via morphological operations
  kernel_river = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
  opened_rgb = cv2.morphologyEx(img_rgb, cv2.MORPH_OPEN, kernel_river)
  closed_rgb = cv2.morphologyEx(img_rgb, cv2.MORPH_CLOSE, kernel_river)
  
  diff_open = cv2.absdiff(img_rgb, opened_rgb)
  diff_close = cv2.absdiff(img_rgb, closed_rgb)
  river_mask = (np.max(diff_open, axis=2) > 20) | (np.max(diff_close, axis=2) > 20)
  
  if np.any(river_mask):
    _, indices = distance_transform_edt(river_mask, return_indices=True)
    img_rgb = img_rgb[indices[0], indices[1]]
    
  # Step 3: Inner-segment grain smoothing via high-spatial mean shift and median filtering
  flat_rgb = cv2.pyrMeanShiftFiltering(img_rgb, 15, 40)
  flat_rgb = cv2.medianBlur(flat_rgb, 7)
  
  if has_alpha:
    out_np = np.dstack((flat_rgb, img_np[:, :, 3]))
  else:
    out_np = flat_rgb
    
  return Image.fromarray(out_np)

def processMap (image_file, reader_obj):
  img_pil = Image.open(image_file).convert("RGBA")
  
  # 1. OCR (Strictly processed first)
  img_np_rgb = np.array(img_pil.convert("RGB"))
  ocr_results = reader_obj.readtext(img_np_rgb)
  
  # 2. Masking
  img_pil, preview_pil = maskInfoboxes(img_pil)
  
  # 3. Denoising thin networks
  denoised_pil = denoiseMap(img_pil)
  
  # 4. Post-processing: second OCR text removal, river filtering & inner-segment flattening
  final_pil = postProcessMap(denoised_pil, reader_obj)
  
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
  
  return masked_img, composite_img, preview_pil, ocr_results

if __name__ == "__main__":
  is_running = st.runtime.exists()
  draw() if is_running else initApp()