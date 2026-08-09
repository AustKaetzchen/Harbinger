import sys
import cv2
import easyocr
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from streamlit.web import cli as stcli

def draw():
    st.set_page_config(page_title="Map Text Extraction Tool", layout="wide")
    st.title("Map Text Extraction Visualisation")
    st.write("Upload a map image to extract text, mask legends/infoboxes, and view semi-transparent bounding box overlays.")
    
    uploaded_file = st.sidebar.file_uploader("Choose a map image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        reader_obj = loadReader()
        
        # Display 3-column gallery
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("1. Original Map")
            st.image(uploaded_file, use_container_width=True)
            
        with st.spinner("Extracting OCR, then generating geometric box ensemble..."):
            composite_img, preview_img, ocr_results = processMap(uploaded_file, reader_obj)
            
        with col2:
            st.subheader("2. Segmentation Candidates")
            st.image(preview_img, use_container_width=True)
            
        with col3:
            st.subheader("3. Annotated & Masked Output")
            st.image(composite_img, use_container_width=True)
            
        st.subheader("Extracted Text Results")
        if ocr_results:
            results_data = []
            for bbox, text, prob in ocr_results:
                results_data.append({"Text": text, "Confidence": f"{prob*100:.2f}%"})
            st.table(results_data)
        else:
            st.info("No text detected in the uploaded map.")

def initApp():
    sys.argv = ["streamlit", "run", __file__]
    sys.exit(stcli.main())

def loadFont(font_size=14):
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
def loadReader():
    return easyocr.Reader(["en", "ru"])

def maskInfoboxes(img_pil):
    img_np = np.array(img_pil)
    img_h, img_w = img_np.shape[:2]
    total_area = img_h * img_w
    
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    
    # Pad the image by 10 pixels on all sides so UI elements touching the screen edge form closed contours
    pad = 10
    img_pad = cv2.copyMakeBorder(img_rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    contours_pool = []
    
    # --- ENSEMBLE GENERATOR 1: Canny Edge Detection ---
    gray_pad = cv2.cvtColor(img_pad, cv2.COLOR_RGB2GRAY)
    for t1, t2 in [(30, 100), (50, 150), (100, 200)]:
        edges = cv2.Canny(gray_pad, t1, t2)
        closed = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours_pool.extend(cnts)

    # --- ENSEMBLE GENERATOR 2: Adaptive Thresholding ---
    thresh1 = cv2.adaptiveThreshold(gray_pad, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    thresh2 = cv2.adaptiveThreshold(gray_pad, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    for th in [thresh1, thresh2]:
        cnts, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours_pool.extend(cnts)

    # --- ENSEMBLE GENERATOR 3: Color Quantization ---
    for bin_size in [16, 32]:
        quantized = (img_pad.astype(np.int32) // bin_size) * bin_size
        quantized = quantized.astype(np.uint8)
        pixels = quantized.reshape(-1, 3)
        colors, counts = np.unique(pixels, axis=0, return_counts=True)
        
        top_colors = colors[np.argsort(-counts)][:20]
        
        for color in top_colors:
            lower = np.clip(color, 0, 255).astype(np.uint8)
            upper = np.clip(color + bin_size - 1, 0, 255).astype(np.uint8)
            color_mask = cv2.inRange(quantized, lower, upper)
            
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            
            cnts, _ = cv2.findContours(color_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours_pool.extend(cnts)

    # --- GEOMETRIC FILTERING ---
    mask_rects_pad = np.zeros(img_pad.shape[:2], dtype=np.uint8)
    
    for cnt in contours_pool:
        x, y, w, h = cv2.boundingRect(cnt)
        bbox_area = w * h
        
        # Discard tiny text/noise (< 0.1%) and massive continents (> 50%)
        if bbox_area < (total_area * 0.001) or bbox_area > (total_area * 0.50):
            continue
            
        c_area = cv2.contourArea(cnt)
        if bbox_area == 0:
            continue
            
        # THE MAGIC RULE: If the shape's outer boundary fills >88% of its bounding box, 
        # it is mathematically an axis-aligned rectangle (an infobox).
        solidity = c_area / bbox_area
        
        if solidity > 0.88:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            if len(approx) <= 8:
                cv2.rectangle(mask_rects_pad, (x, y), (x+w, y+h), 255, -1)

    # Remove the 10-pixel padding to return the mask to the image's original dimensions
    mask_rects = mask_rects_pad[pad:-pad, pad:-pad]

    # --- MERGE & DRAW PREVIEW ---
    final_contours, _ = cv2.findContours(mask_rects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    preview_img = (img_rgb * 0.35).astype(np.uint8)
    
    # Expand the transparency mask slightly to consume the actual UI border lines
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    final_mask = cv2.dilate(mask_rects, dilate_kernel, iterations=1)
    
    for cnt in final_contours:
        # Instead of drawing a bounding box (which balloons over empty space when UI panels touch), 
        # we trace the exact polygon shape of the merged mask.
        rand_color = (random.randint(80, 255), random.randint(80, 255), random.randint(80, 255))
        overlay = preview_img.copy()
        
        cv2.drawContours(overlay, [cnt], 0, rand_color, -1)
        cv2.addWeighted(overlay, 0.5, preview_img, 0.5, 0, preview_img)
        cv2.drawContours(preview_img, [cnt], 0, rand_color, 2)
        
        # Draw a text label roughly in the center of the bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        text = "Masked UI"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        tx = x + (w - text_size[0]) // 2
        ty = y + (h + text_size[1]) // 2
        cv2.rectangle(preview_img, (tx - 2, ty - text_size[1] - 2), (tx + text_size[0] + 2, ty + 2), (0, 0, 0), -1)
        cv2.putText(preview_img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Erase from the final map output
    img_np[final_mask == 255] = (0, 0, 0, 0)
    
    return Image.fromarray(img_np), Image.fromarray(preview_img)

def processMap(image_file, reader_obj):
    img_pil = Image.open(image_file).convert("RGBA")
    
    # 1. RUN OCR FIRST (Strict processing order enforced here)
    img_np_rgb = np.array(img_pil.convert("RGB"))
    ocr_results = reader_obj.readtext(img_np_rgb)
    
    # 2. Erase the rectangles and generate the gallery preview map
    img_pil, preview_pil = maskInfoboxes(img_pil)
    
    # 3. Draw OCR bounding boxes over the output map
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
    
    return composite_img, preview_pil, ocr_results

if __name__ == "__main__":
    is_running = st.runtime.exists()
    draw() if is_running else initApp()