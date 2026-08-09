import os
import re
import sys
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from streamlit.web import cli as stcli

MODEL_ID = "nvidia/LocateAnything-3B"
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "LocateAnything-3B")

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
    draw_obj.rectangle([x1, y1, x2, y2], outline="red", width=2)
    display_txt = label if current_mode in ["OCR", "Labelling"] else f"{label} ({score_pct})"
    draw_obj.rectangle([x1, max(0, y1-18), x1+len(display_txt)*9, max(0, y1)], fill="black")
    text_fill = "yellow" if current_mode == "OCR" else ("cyan" if current_mode == "Labelling" else "lime")
    draw_obj.text((x1+2, max(0, y1-16)), display_txt, fill=text_fill, font=font_obj)
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
  prompt_query = st.sidebar.text_input("Enter Query / Categories (Optional)", value="")
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
    st.image(processed_image, caption=f"Active Resolution: {processed_image.width}x{processed_image.height} px", width="stretch")
  with col_right:
    st.subheader("LocateAnything Visual Output")
    if execute_button:
      with st.spinner("Executing NVIDIA LocateAnything model inference..."):
        model_instance = loadLocateAnythingModel()
        results, decoded_tokens = runLocateAnythingInference(model_instance, processed_image, prompt_query, mode_selected, snippet_bounds)
        if model_instance.get("status") != "loaded":
          st.error(f"Model Status: {model_instance.get('status')}")
        overlay_image = buildOverlay(processed_image, results, mode_selected)
        st.image(overlay_image, caption=f"Output Visualisation ({mode_selected})", width="stretch")
        st.subheader(f"Predictions ({len(results)} detected)")
        if results:
          st.dataframe(results)
        else:
          st.warning("No target bounding boxes or text regions were returned by LocateAnything.")
        if mode_selected == "Decoding" and decoded_tokens:
          st.subheader("Decoded Individual Tokens")
          st.code(" ".join(decoded_tokens), language="text")
          st.json(decoded_tokens)
    else:
      st.info("Press 'Execute' button to run LocateAnything process.")

def ensureLocalModelDownloaded (model_id, local_dir):
  config_file = os.path.join(local_dir, "config.json")
  if os.path.exists(config_file):
    return local_dir
  from filelock import FileLock
  from huggingface_hub import snapshot_download
  os.makedirs(local_dir, exist_ok=True)
  lock_file_path = os.path.join(os.path.expanduser("~"), f".{model_id.replace('/', '_')}.lock")
  with FileLock(lock_file_path):
    if not os.path.exists(config_file):
      print(f"[HARBINGER] Downloading model files directly to local path: {local_dir}")
      snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False
      )
  return local_dir

def initApp ():
  sys.argv = ["streamlit", "run", __file__]
  sys.exit(stcli.main())

@st.cache_resource
def loadLocateAnythingModel ():
  import torch
  from transformers import AutoModel, AutoTokenizer, AutoProcessor
  try:
    model_path = ensureLocalModelDownloaded(MODEL_ID, LOCAL_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(
      model_path,
      trust_remote_code=True,
      local_files_only=True,
      fix_mistral_regex=True
    )
    processor = AutoProcessor.from_pretrained(
      model_path,
      trust_remote_code=True,
      local_files_only=True
    )
    model = AutoModel.from_pretrained(
      model_path,
      dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
      device_map="auto" if torch.cuda.is_available() else "cpu",
      trust_remote_code=True,
      local_files_only=True
    ).eval()
    return {"processor": processor, "tokenizer": tokenizer, "model": model, "status": "loaded"}
  except Exception as e:
    return {"processor": None, "tokenizer": None, "model": None, "status": f"unloaded: {str(e)}"}

def parseLocateAnythingOutput (decoded_text, img_w, img_h, default_label):
  boxes = []
  pattern_box = r"(?:<ref>(.*?)</ref>)?\s*<box>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*</box>"
  matches = re.findall(pattern_box, decoded_text)
  for idx, match in enumerate(matches):
    ref_label, x1_str, y1_str, x2_str, y2_str = match
    box_x1 = int(int(x1_str)*img_w/1000)
    box_y1 = int(int(y1_str)*img_h/1000)
    box_x2 = int(int(x2_str)*img_w/1000)
    box_y2 = int(int(y2_str)*img_h/1000)
    clean_ref = ref_label.strip()
    lbl = clean_ref if clean_ref else (default_label if default_label else f"item_{idx+1}")
    boxes.append({
      "box": [box_x1, box_y1, box_x2, box_y2],
      "label": lbl,
      "score": 0.95,
      "text": lbl
    })
  token_matches = re.findall(r"(<ref>.*?</ref>|<box>.*?</box>|<.*?>|[^<\s]+)", decoded_text)
  tokens = [t.strip() for t in token_matches if t.strip()]
  return boxes, tokens

def runLocateAnythingInference (model_obj, image_input, text_query, current_mode, crop_bounds):
  import torch
  img_w, img_h = image_input.size
  if current_mode == "Decoding" and crop_bounds is not None:
    nw_x, nw_y, se_x, se_y = crop_bounds
    image_input = image_input.crop((nw_x, nw_y, se_x, se_y))
    img_w, img_h = image_input.size
  if current_mode == "OCR":
    prompt = f"Please locate the text referred as {text_query}." if text_query else "Detect all the text in box format."
  elif current_mode == "Labelling":
    prompt = f"Locate all the instances that match the following description: {text_query}." if text_query else "Detect and label all distinct features in the image."
  elif current_mode == "Bounding Box Detection":
    if text_query:
      cats = "</c>".join([c.strip() for c in text_query.split(",") if c.strip()])
      prompt = f"Locate all the instances that matches the following description: {cats}."
    else:
      prompt = "Locate all the instances in box format."
  else:
    prompt = f"Locate all instances of {text_query}." if text_query else "Detect all the objects in box format."
  decoded_text = ""
  if model_obj.get("status") == "loaded":
    processor = model_obj["processor"]
    tokenizer = model_obj["tokenizer"]
    model = model_obj["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    messages = [
      {"role": "user", "content": [
        {"type": "image", "image": image_input},
        {"type": "text", "text": prompt}
      ]}
    ]
    try:
      text_prompt = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
      images, videos = processor.process_vision_info(messages)
      inputs = processor(text=[text_prompt], images=images, videos=videos, return_tensors="pt").to(device)
      pixel_values = inputs["pixel_values"].to(dtype)
      input_ids = inputs["input_ids"]
      image_grid_hws = inputs.get("image_grid_hws", None)
      outputs = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=inputs.get("attention_mask", None),
        image_grid_hws=image_grid_hws,
        tokenizer=tokenizer,
        max_new_tokens=2048,
        use_cache=True,
        generation_mode="hybrid"
      )
      res_val = outputs[0] if isinstance(outputs, tuple) else outputs
      decoded_text = res_val if isinstance(res_val, str) else processor.decode(res_val[0], skip_special_tokens=False)
    except Exception:
      inputs = processor(images=image_input, text=prompt, return_tensors="pt").to(device)
      with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2048)
      decoded_text = processor.decode(outputs[0], skip_special_tokens=False)
  boxes_data, tokens_data = parseLocateAnythingOutput(decoded_text, img_w, img_h, text_query)
  if current_mode == "Decoding" and crop_bounds is not None and boxes_data:
    nw_x, nw_y, _, _ = crop_bounds
    for b in boxes_data:
      b["box"][0] += nw_x
      b["box"][1] += nw_y
      b["box"][2] += nw_x
      b["box"][3] += nw_y
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
  if is_running:
    draw()
  else:
    ensureLocalModelDownloaded(MODEL_ID, LOCAL_MODEL_DIR)
    initApp()