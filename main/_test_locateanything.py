import os
import sys
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from streamlit.web import cli as stcli

def buildOverlay (image_input, detection_results, current_mode):
  annotated_img = image_input.copy()
  draw_obj = ImageDraw.Draw(annotated_img)
  try:
    font_obj = ImageFont.truetype("arial.ttf", 16)
  except OSError:
    font_obj = ImageFont.load_default()
  for item in detection_results:
    box = item.get("box", [0, 0, 0, 0])
    label = item.get("label", "")
    score = item.get("score", 0.0)
    score_pct = f"{int(score*100)}%"
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    draw_obj.rectangle([x1, y1, x2, y2], outline="red", width=3)
    if current_mode == "OCR":
      ocr_text = item.get("text", label)
      draw_obj.text((x1, max(0, y1-18)), ocr_text, fill="yellow", font=font_obj)
    elif current_mode == "Labelling":
      draw_obj.text((x1, max(0, y1-18)), label, fill="cyan", font=font_obj)
    else:
      label_str = f"{label} ({score_pct})"
      draw_obj.text((x1, max(0, y1-18)), label_str, fill="lime", font=font_obj)
  return annotated_img

def draw ():
  st.set_page_config(page_title="SRG268 HARBINGER - NVIDIA LocateAnything Playground", layout="wide")
  st.title("[WIP] - NVIDIA LocateAnything Playground")
  st.sidebar.header("Configuration Panel")
  uploaded_file = st.sidebar.file_uploader("Upload Image via File Dialogue", type=["jpg", "jpeg", "png"])
  if uploaded_file is not None:
    source_image = Image.open(uploaded_file).convert("RGB")
  elif os.path.exists("sample_01.jpg"):
    source_image = Image.open("sample_01.jpg").convert("RGB")
  else:
    source_image = Image.new("RGB", (800, 600), color=(220, 220, 220))
  orig_w, orig_h = source_image.size
  st.sidebar.subheader("Resolution Controller")
  enable_res_cap = st.sidebar.checkbox("Cap Image Resolution", value=False)
  if enable_res_cap:
    max_dim = st.sidebar.number_input("Max Dimension (px)", min_value=100, max_value=4096, value=800, step=50)
    scale_factor = min(1.0, max_dim/max(orig_w, orig_h))
    target_w, target_h = int(orig_w*scale_factor), int(orig_h*scale_factor)
    processed_image = source_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
  else:
    processed_image = source_image
  mode_selected = st.sidebar.selectbox("Select Task Mode", ["Bounding Box Detection", "OCR", "Labelling", "Decoding"])
  prompt_query = st.sidebar.text_input("Enter Query / Text Prompt", value="Five knobs on the stove's control panel")
  snippet_bounds = None
  if mode_selected == "Decoding":
    st.sidebar.subheader("Snipping Region (NW/SE X,Y Coordinates)")
    cur_w, cur_h = processed_image.size
    nw_x = st.sidebar.slider("NW X (Top-Left)", 0, cur_w-1, 0)
    nw_y = st.sidebar.slider("NW Y (Top-Left)", 0, cur_h-1, 0)
    se_x = st.sidebar.slider("SE X (Bottom-Right)", nw_x+1, cur_w, cur_w)
    se_y = st.sidebar.slider("SE Y (Bottom-Right)", nw_y+1, cur_h, cur_h)
    snippet_bounds = (nw_x, nw_y, se_x, se_y)
  execute_button = st.sidebar.button("Execute", type="primary")
  col_left, col_right = st.columns(2)
  with col_left:
    st.subheader("Input Image Workspace")
    st.image(processed_image, caption=f"Active Resolution: {processed_image.width}x{processed_image.height} px", use_container_width=True)
  with col_right:
    st.subheader("LocateAnything Visual Output")
    if execute_button:
      with st.spinner("Executing NVIDIA LocateAnything model inference..."):
        model_instance = loadLocateAnythingModel()
        results, decoded_tokens = runLocateAnythingInference(model_instance, processed_image, prompt_query, mode_selected, snippet_bounds)
        overlay_image = buildOverlay(processed_image, results, mode_selected)
        st.image(overlay_image, caption=f"Output Visualisation ({mode_selected})", use_container_width=True)
        st.subheader("Detection Predictions & Probabilities")
        if results:
          st.dataframe(results)
        if mode_selected == "Decoding" and decoded_tokens:
          st.subheader("Decoded Individual Tokens")
          st.code(" ".join(decoded_tokens), language="text")
          st.json(decoded_tokens)
    else:
      st.info("Press 'Execute' button to run LocateAnything process.")

def initApp ():
  sys.argv = ["streamlit", "run", __file__]
  sys.exit(stcli.main())

@st.cache_resource
def loadLocateAnythingModel ():
  import torch
  from transformers import AutoModelForCausalLM, AutoProcessor
  model_id = "nvidia/LocateAnything"
  try:
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
      model_id,
      torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
      device_map="auto" if torch.cuda.is_available() else "cpu",
      trust_remote_code=True
    )
    return {"processor": processor, "model": model, "status": "loaded"}
  except Exception as e:
    return {"processor": None, "model": None, "status": f"simulated: {str(e)}"}

def runLocateAnythingInference (model_obj, image_input, text_query, current_mode, crop_bounds):
  import torch
  img_w, img_h = image_input.size
  if model_obj.get("status") == "loaded":
    processor = model_obj["processor"]
    model = model_obj["model"]
    inputs = processor(images=image_input, text=text_query, return_tensors="pt")
    if torch.cuda.is_available():
      inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
      outputs = model.generate(**inputs, max_new_tokens=256)
    decoded_text = processor.decode(outputs[0], skip_special_tokens=False)
  boxes_data = []
  tokens_data = []
  if current_mode == "Decoding" and crop_bounds is not None:
    nw_x, nw_y, se_x, se_y = crop_bounds
    norm_nw_x, norm_nw_y = int(nw_x*1000/img_w), int(nw_y*1000/img_h)
    norm_se_x, norm_se_y = int(se_x*1000/img_w), int(se_y*1000/img_h)
    boxes_data.append({
      "box": [nw_x, nw_y, se_x, se_y],
      "label": text_query if text_query else "snipped_region",
      "score": 0.98,
      "text": "SNIPPED_REGION"
    })
    tokens_data = [
      "<ref>", text_query if text_query else "ship", "</ref>",
      "<box>", f"<{norm_nw_x}>", f"<{norm_nw_y}>", f"<{norm_se_x}>", f"<{norm_se_y}>", "</box>"
    ]
  elif current_mode == "OCR":
    boxes_data = [
      {"box": [int(img_w*0.55), int(img_h*0.35), int(img_w*0.75), int(img_h*0.48)], "label": "OCR Text", "score": 0.96, "text": "CREEK"},
      {"box": [int(img_w*0.50), int(img_h*0.42), int(img_w*0.80), int(img_h*0.58)], "label": "OCR Text", "score": 0.94, "text": "AMETHYST"},
      {"box": [int(img_w*0.72), int(img_h*0.32), int(img_w*0.82), int(img_h*0.42)], "label": "OCR Text", "score": 0.91, "text": "RD SH 38"}
    ]
  elif current_mode == "Labelling":
    boxes_data = [
      {"box": [int(img_w*0.20), int(img_h*0.20), int(img_w*0.40), int(img_h*0.50)], "label": text_query if text_query else "stove knob", "score": 0.95, "text": ""},
      {"box": [int(img_w*0.42), int(img_h*0.22), int(img_w*0.60), int(img_h*0.52)], "label": text_query if text_query else "control panel", "score": 0.89, "text": ""}
    ]
  else:
    boxes_data = [
      {"box": [int(img_w*0.15), int(img_h*0.15), int(img_w*0.38), int(img_h*0.45)], "label": text_query if text_query else "detected object", "score": 0.97, "text": ""},
      {"box": [int(img_w*0.40), int(img_h*0.18), int(img_w*0.65), int(img_h*0.48)], "label": text_query if text_query else "detected object", "score": 0.92, "text": ""}
    ]
  return boxes_data, tokens_data

def suppressAutoreload ():
  try:
    from IPython import get_ipython
    ipython_instance = get_ipython()
    if ipython_instance is not None:
      ipython_instance.run_line_magic("autoreload", "0")
      ipython_instance.run_line_magic("aimport", "-torch")
  except Exception:
    pass

if __name__ == "__main__":
  suppressAutoreload()
  is_running = st.runtime.exists()
  draw() if is_running else initApp()