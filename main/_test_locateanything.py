import os
import re
import sys
import traceback
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
  from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor
  patchTransformersDynamicModules()
  try:
    model_path = ensureLocalModelDownloaded(MODEL_ID, LOCAL_MODEL_DIR)
    config = AutoConfig.from_pretrained(
      model_path,
      trust_remote_code=True,
      local_files_only=True
    )
    sanitiseConfig(config)
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
    if hasattr(processor, "image_processor"):
      setattr(processor.image_processor, "max_pixels", 1024*1024)
      setattr(processor.image_processor, "min_pixels", 256*28*28)
    model = AutoModel.from_pretrained(
      model_path,
      config=config,
      dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
      device_map="auto" if torch.cuda.is_available() else "cpu",
      trust_remote_code=True,
      local_files_only=True
    ).eval()
    return {"processor": processor, "tokenizer": tokenizer, "model": model, "status": "loaded"}
  except Exception as e:
    err_tb = traceback.format_exc()
    return {"processor": None, "tokenizer": None, "model": None, "status": f"unloaded: {str(e)}\nTraceback:\n{err_tb}"}

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

def patchSingleClass (target_cls):
  import inspect
  if not target_cls or not isinstance(target_cls, type):
    return
  if hasattr(target_cls, "_tied_weights_keys") and isinstance(target_cls._tied_weights_keys, list):
    twk = target_cls._tied_weights_keys
    if len(twk) == 2:
      target_cls._tied_weights_keys = {twk[0]: twk[1]}
    elif len(twk) == 1:
      target_cls._tied_weights_keys = {twk[0]: twk[0]}
    else:
      target_cls._tied_weights_keys = {k: k for k in twk}
  for base in target_cls.__mro__:
    if "_check_and_adjust_attn_implementation" in base.__dict__:
      attr = base.__dict__["_check_and_adjust_attn_implementation"]
      if getattr(attr, "_is_already_patched_for_kernels", False):
        continue
      is_class_method = isinstance(attr, classmethod)
      is_static_method = isinstance(attr, staticmethod)
      underlying = attr.__func__ if (is_class_method or is_static_method) else attr
      try:
        sig = inspect.signature(underlying)
        has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if "allow_all_kernels" not in sig.parameters and not has_var_kwargs:
          def makeWrapper (fn):
            def wrapper (*w_args, **w_kwargs):
              w_kwargs.pop("allow_all_kernels", None)
              return fn(*w_args, **w_kwargs)
            wrapper._is_already_patched_for_kernels = True
            return wrapper
          wrapped = makeWrapper(underlying)
          if is_class_method:
            new_attr = classmethod(wrapped)
          elif is_static_method:
            new_attr = staticmethod(wrapped)
          else:
            new_attr = wrapped
          setattr(new_attr, "_is_already_patched_for_kernels", True)
          setattr(base, "_check_and_adjust_attn_implementation", new_attr)
      except Exception:
        pass

def patchTransformersDynamicModules ():
  import sys
  import transformers.dynamic_module_utils
  import transformers.modeling_utils
  from transformers import PretrainedConfig, PreTrainedModel
  orig_config_getattr = getattr(PretrainedConfig, "__getattr__", None)
  def patchedConfigGetattr (self, key):
    if key == "rope_theta":
      return 10000.0
    if key == "rope_scaling":
      return {"type": "mrope", "mrope_section": [16, 24, 24]}
    if orig_config_getattr is not None:
      return orig_config_getattr(self, key)
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
  PretrainedConfig.__getattr__ = patchedConfigGetattr
  @property
  def ropeScalingProp (self):
    val = self.__dict__.get("rope_scaling", None)
    if isinstance(val, list):
      d_val = {"type": "mrope", "mrope_section": val}
      self.__dict__["rope_scaling"] = d_val
      return d_val
    if isinstance(val, dict):
      if "type" not in val:
        val["type"] = "mrope"
      return val
    return {"type": "mrope", "mrope_section": [16, 24, 24]}
  PretrainedConfig.rope_scaling = ropeScalingProp
  @property
  def allTiedWeightsKeysProp (self):
    if "_all_tied_weights_keys" in self.__dict__:
      return self.__dict__["_all_tied_weights_keys"]
    try:
      val = self.get_expanded_tied_weights_keys(all_submodels=False)
      self.__dict__["_all_tied_weights_keys"] = val
      return val
    except Exception:
      return []
  @allTiedWeightsKeysProp.setter
  def allTiedWeightsKeysProp (self, val):
    self.__dict__["_all_tied_weights_keys"] = val
  PreTrainedModel.all_tied_weights_keys = allTiedWeightsKeysProp
  orig_get_expanded = getattr(PreTrainedModel, "get_expanded_tied_weights_keys", None)
  if orig_get_expanded is not None and not getattr(orig_get_expanded, "_is_patched", False):
    def patchedGetExpanded (self, *args, **kwargs):
      if hasattr(self, "_tied_weights_keys"):
        twk = getattr(self, "_tied_weights_keys")
        if isinstance(twk, list):
          if len(twk) == 2:
            d_twk = {twk[0]: twk[1]}
          else:
            d_twk = {k: k for k in twk}
          try:
            setattr(self, "_tied_weights_keys", d_twk)
          except Exception:
            pass
      if orig_get_expanded is not None:
        try:
          return orig_get_expanded(self, *args, **kwargs)
        except Exception:
          return []
      return []
    setattr(patchedGetExpanded, "_is_patched", True)
    PreTrainedModel.get_expanded_tied_weights_keys = patchedGetExpanded
  patchSingleClass(PreTrainedModel)
  for mod in list(sys.modules.values()):
    if mod is not None and hasattr(mod, "__dict__"):
      for obj in list(mod.__dict__.values()):
        if isinstance(obj, type) and issubclass(obj, PreTrainedModel):
          patchSingleClass(obj)
  orig_init_subclass = getattr(PreTrainedModel, "__init_subclass__", None)
  @classmethod
  def patchedInitSubclass (cls, **kwargs):
    if orig_init_subclass is not None:
      orig_init_subclass(**kwargs)
    patchSingleClass(cls)
  PreTrainedModel.__init_subclass__ = patchedInitSubclass
  orig_get_module = getattr(transformers.dynamic_module_utils, "get_module_from_dynamic_module", None)
  if orig_get_module is not None and not getattr(orig_get_module, "_is_patched", False):
    def patchedGetModule (*args, **kwargs):
      mod = orig_get_module(*args, **kwargs)
      if mod is not None and hasattr(mod, "__dict__"):
        for obj in list(mod.__dict__.values()):
          if isinstance(obj, type):
            patchSingleClass(obj)
      return mod
    setattr(patchedGetModule, "_is_patched", True)
    transformers.dynamic_module_utils.get_module_from_dynamic_module = patchedGetModule
  orig_get_class = getattr(transformers.dynamic_module_utils, "get_class_from_dynamic_module", None)
  if orig_get_class is not None and not getattr(orig_get_class, "_is_patched", False):
    def patchedGetClass (*args, **kwargs):
      cls = orig_get_class(*args, **kwargs)
      patchSingleClass(cls)
      return cls
    setattr(patchedGetClass, "_is_patched", True)
    transformers.dynamic_module_utils.get_class_from_dynamic_module = patchedGetClass

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
    with torch.inference_mode():
      try:
        text_prompt = ""
        if hasattr(processor, "apply_chat_template"):
          text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        elif hasattr(tokenizer, "apply_chat_template"):
          text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if not text_prompt or "<|image_pad|>" not in text_prompt:
          text_prompt = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"
        raw_inputs = processor(text=[text_prompt], images=[image_input], padding=True, return_tensors="pt")
        inputs = {}
        for k, v in raw_inputs.items():
          if v is not None:
            if isinstance(v, torch.Tensor):
              if k == "pixel_values" and torch.is_floating_point(v):
                inputs[k] = v.to(device=device, dtype=dtype)
              else:
                inputs[k] = v.to(device=device)
            else:
              inputs[k] = v
        eos_id = getattr(tokenizer, "eos_token_id", None)
        pad_id = getattr(tokenizer, "pad_token_id", eos_id)
        outputs = model.generate(
          **inputs,
          max_new_tokens=512,
          use_cache=True,
          eos_token_id=eos_id,
          pad_token_id=pad_id
        )
        res_val = outputs[0] if isinstance(outputs, tuple) else outputs
        decoded_text = res_val if isinstance(res_val, str) else processor.decode(res_val[0], skip_special_tokens=False)
      except Exception:
        text_prompt = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"
        raw_inputs = processor(text=[text_prompt], images=[image_input], return_tensors="pt")
        inputs = {}
        for k, v in raw_inputs.items():
          if v is not None:
            if isinstance(v, torch.Tensor):
              if k == "pixel_values" and torch.is_floating_point(v):
                inputs[k] = v.to(device=device, dtype=dtype)
              else:
                inputs[k] = v.to(device=device)
            else:
              inputs[k] = v
        outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)
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

def sanitiseConfig (cfg):
  if cfg is None:
    return
  if not hasattr(cfg, "rope_theta") or getattr(cfg, "rope_theta", None) is None:
    setattr(cfg, "rope_theta", 10000.0)
  r_scale = getattr(cfg, "rope_scaling", None)
  if isinstance(r_scale, list):
    setattr(cfg, "rope_scaling", {"type": "mrope", "mrope_section": r_scale})
  elif isinstance(r_scale, dict) and "type" not in r_scale:
    r_scale["type"] = "mrope"
  elif r_scale is None:
    setattr(cfg, "rope_scaling", {"type": "mrope", "mrope_section": [16, 24, 24]})
  for key, val in list(getattr(cfg, "__dict__", {}).items()):
    if isinstance(val, list) and key == "rope_scaling":
      cfg.__dict__[key] = {"type": "mrope", "mrope_section": val}
    elif hasattr(val, "__dict__") and val is not cfg:
      sanitiseConfig(val)

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
  patchTransformersDynamicModules()
  is_running = st.runtime.exists()
  if is_running:
    draw()
  else:
    ensureLocalModelDownloaded(MODEL_ID, LOCAL_MODEL_DIR)
    initApp()