import sys
import cv2
import easyocr
import numpy as np
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
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Map")
            st.image(uploaded_file, use_container_width=True)
            
        with st.spinner("Segmenting map by straight lines and removing rectangles..."):
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
    
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # 1. Edge detection and grab all straight lines
    edges = cv2.Canny(gray, 30, 150)
    
    # Extract purely vertical and horizontal lines (this destroys organic terrain shapes)
    min_line = 15
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_line))
    
    h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)
    
    # Combine into a single grid of straight lines
    grid = cv2.bitwise_or(h_lines, v_lines)
    
    # Connect intersecting lines to close off box corners
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    grid = cv2.dilate(grid, dilate_kernel, iterations=2)
    
    # Draw a thin border to seal off UI elements resting exactly on the edge of the image
    cv2.rectangle(grid, (0, 0), (img_w - 1, img_h - 1), 255, 2)

    # 2. Segment the map based on straight lines
    # Inverting the grid gives us solid white segments representing the empty space inside boxes
    inverted_grid = cv2.bitwise_not(grid)
    
    # Find contours of both the bounding frames AND the solid segments inside
    contours_frames, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_segments, _ = cv2.findContours(inverted_grid, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    all_contours = list(contours_frames) + list(contours_segments)

    # 3. If those segments are a rectangle, remove them
    for cnt in all_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        # Filter out tiny pixel noise and ignore the full-screen map segment itself
        if area < 150 or area > total_area * 0.70:
            continue

        is_rect = False
        contour_area = cv2.contourArea(cnt)
        
        # Test A: Extent (For solid blobs like flags or white popups)
        # If the segment densely fills its bounding box, it's a rectangle.
        if area > 0 and (contour_area / area) > 0.85:
            is_rect = True
        else:
            # Test B: Polygon Trace (For hollow frames like the main legend border)
            # Because the segments are sourced entirely from straight horizontal/vertical 
            # lines, any 4-sided convex polygon found here is mathematically a perfect rectangle.
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            if len(approx) == 4 and cv2.isContourConvex(approx):
                is_rect = True
                
        if is_rect:
            # Mask the entire segment area
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    # Apply the mask to make removed regions transparent
    img_np[mask == 255] = (0, 0, 0, 0)
    return Image.fromarray(img_np)

def processMap(image_file, reader_obj):
    img_pil = Image.open(image_file).convert("RGBA")
    
    # Analyze text on the raw original map
    img_np = np.array(img_pil.convert("RGB"))
    ocr_results = reader_obj.readtext(img_np)
    
    # Erase the rectangles
    img_pil = maskInfoboxes(img_pil)
    
    # Draw OCR bounding boxes
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