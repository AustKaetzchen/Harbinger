import sys
import cv2
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from streamlit.web import cli as stcli

def draw ():
  st.set_page_config(page_title="Map Text Extraction Tool", layout="wide")
  st.title("Map Text Extraction Visualisation")
  st.write("Upload a map image to extract text, mask legends/infoboxes, and view semi-transparent bounding box overlays.")
  uploaded_file = st.sidebar.file_uploader("Choose a map image...", type=["jpg", "jpeg", "png"])
  if uploaded_file is not None:
    reader_obj = loadReader()
    col1, col2 = st.columns(2)
    with col1:
      st.subheader("Original Map")
      st.image(uploaded_file, use_container_width=True)
    with st.spinner("Extracting text, cropping infoboxes, and generating overlays..."):
      processed_img, ocr_results = processMap(uploaded_file, reader_obj)
    with col2:
      st.subheader("Annotated & Masked Map")
      st.image(processed_img, use_container_width=True)
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

def maskInfoboxes (img_pil, minimum_ratio=0.002, maximum_ratio=0.45):
  img_np = np.array(img_pil)
  img_h, img_w = img_np.shape[:2]
  total_area = img_w*img_h
  gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
  edges_img = cv2.Canny(gray_img, 30, 150)
  kernel_mat = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
  closed_img = cv2.morphologyEx(edges_img, cv2.MORPH_CLOSE, kernel_mat)
  found_contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  for single_contour in found_contours:
    x_pos, y_pos, box_w, box_h = cv2.boundingRect(single_contour)
    box_area = box_w*box_h
    if total_area*minimum_ratio < box_area < total_area*maximum_ratio:
      contour_area = cv2.contourArea(single_contour)
      extent_val = float(contour_area)/box_area if box_area > 0 else 0
      if extent_val > 0.4:
        img_np[y_pos:y_pos+box_h, x_pos:x_pos+box_w] = (0, 0, 0, 0)
  return Image.fromarray(img_np)

def processMap (image_file, reader_obj):
  img_pil = Image.open(image_file).convert("RGBA")
  img_np = np.array(img_pil.convert("RGB"))
  ocr_results = reader_obj.readtext(img_np)
  img_pil = maskInfoboxes(img_pil)
  overlay_img = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
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
  composite_img = Image.alpha_composite(img_pil, overlay_img)
  return composite_img, ocr_results

if __name__ == "__main__":
  is_running = st.runtime.exists()
  draw() if is_running else initApp()