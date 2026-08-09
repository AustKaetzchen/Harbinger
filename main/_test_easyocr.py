import cv2
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st


def drawOverlays (image, ocr_results):
  """Draw semi-transparent bounding boxes and text labels on image."""
  img_rgba = image.convert("RGBA")
  overlay = Image.new("RGBA", img_rgba.size, (255, 255, 255, 0))
  draw = ImageDraw.Draw(overlay)
  try:
    font = ImageFont.truetype("arial.ttf", 16)
  except IOError:
    font = ImageFont.load_default()
  for bbox, text, prob in ocr_results:
    pt1 = tuple(map(int, bbox[0]))
    pt2 = tuple(map(int, bbox[1]))
    pt3 = tuple(map(int, bbox[2]))
    pt4 = tuple(map(int, bbox[3]))
    draw.polygon([pt1, pt2, pt3, pt4], fill=(255, 0, 0, 80), outline=(255, 0, 0, 255))
    text_pos = (pt1[0], max(0, pt1[1]-18))
    draw.rectangle([text_pos[0], text_pos[1], text_pos[0]+len(text)*8, text_pos[1]+16], fill=(0, 0, 0, 160))
    draw.text(text_pos, text, fill=(255, 255, 255, 255), font=font)
  return Image.alpha_composite(img_rgba, overlay)


@st.cache_resource
def loadOcrReader ():
  """Initialise and cache the EasyOCR reader object."""
  return easyocr.Reader(["en"])


def main ():
  """Main function to run Streamlit application."""
  st.title("Map Text Extractor")
  st.write("Upload an arbitrary map image to extract text with semi-transparent overlays.")
  uploaded_file = st.file_uploader("Choose a map image...", type=["jpg", "jpeg", "png"])
  if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Uploaded Map Image", use_container_width=True)
    with st.spinner("Analysing image and extracting text..."):
      reader = loadOcrReader()
      img_np = np.array(input_image)
      results = reader.readtext(img_np)
    st.subheader("Map with Semi-Transparent Overlays")
    annotated_image = drawOverlays(input_image, results)
    st.image(annotated_image, caption="Resultant Image", use_container_width=True)
    st.subheader("Extracted Labels")
    if results:
      for bbox, text, confidence in results:
        st.write(f"**Label:** {text} | **Confidence:** {confidence:.2f}")
    else:
      st.info("No text detected in the uploaded map image.")


if __name__ == "__main__":
  main()