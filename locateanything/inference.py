import re
import torch
import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoTokenizer, AutoProcessor


def get_color_for_label(label):
    """Get a consistent color for a given label."""
    colors = [
        (8, 145, 178), (220, 38, 38), (22, 163, 74), (37, 99, 235),
        (217, 119, 6), (147, 51, 234),
    ]
    idx = sum(ord(c) for c in label)
    return colors[idx % len(colors)]


def parse_mixed_results(text, category_str=""):
    """Parse model output into detection results with labels.
    
    Handles both <ref>label</ref><box>... format and simple <box>... format.
    Returns list of dicts with type, coords, and label.
    """
    results = []
    expected_cats = [c.strip().lower() for c in category_str.split("</c>") if c.strip()]

    ref_box_pattern = r"(<ref>.*?</ref>)|(<box>.*?</box>)"
    current_label = None
    found_structured = False

    for m in re.finditer(ref_box_pattern, text, flags=re.IGNORECASE | re.DOTALL):
        token = m.group(0)
        if token.lower().startswith("<ref>"):
            label_raw = re.sub(r"</?ref>", "", token, flags=re.IGNORECASE).strip()
            if label_raw:
                current_label = label_raw
        else:
            content = re.sub(r"</?box>", "", token, flags=re.IGNORECASE)
            nums = re.findall(r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>", content)
            coords = [float(n) for n in nums]
            if not coords:
                continue
            label = current_label
            if label is None:
                label = expected_cats[0] if expected_cats else "object"
            if len(coords) == 4:
                results.append({"type": "box", "coords": coords, "label": label})
            elif len(coords) == 2:
                results.append({"type": "point", "coords": coords, "label": label})
            found_structured = True

    if found_structured:
        return results

    # Fallback: simple box pattern
    box_pattern = r"<box>(.*?)</box>"
    parts = re.split(box_pattern, text)
    for i in range(1, len(parts), 2):
        preceding_text = parts[i - 1].lower()
        content = parts[i]
        label = expected_cats[0] if expected_cats else "object"
        for cat in expected_cats:
            if cat in preceding_text:
                label = cat
                break
        nums = re.findall(r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>", content)
        coords = [float(n) for n in nums]
        if len(coords) == 4:
            results.append({"type": "box", "coords": coords, "label": label})
        elif len(coords) == 2:
            results.append({"type": "point", "coords": coords, "label": label})

    return results


def draw_on_frame(frame_bgr, results, draw_label=True):
    """Draw detection results on a frame (BGR numpy array).
    
    Args:
        frame_bgr: Input frame in BGR format (numpy array)
        results: List of detection results from parse_mixed_results
        draw_label: Whether to draw text labels
    
    Returns:
        Frame with drawn detections in BGR format
    """
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    img_draw = pil_img.convert("RGBA")
    overlay = Image.new("RGBA", img_draw.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    w_img, h_img = pil_img.size
    
    # Scale font size based on image dimensions (similar to image processing)
    # Base size 20 for 1000px reference, scale proportionally
    base_size = 20
    scale_factor = min(w_img, h_img) / 1000.0
    font_size = max(12, int(base_size * scale_factor))
    
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None

    parsed = []
    for res in results:
        label = res.get("label", "object")
        color = get_color_for_label(label)
        if res.get("type") == "point":
            c = res["coords"]
            cx = max(0, min(w_img, c[0] * w_img / 1000))
            cy = max(0, min(h_img, c[1] * h_img / 1000))
            parsed.append(("point", label, color, cx, cy))
            continue
        if "is_pixel" in res:
            x1, y1, bw, bh = res["coords"]
            x2, y2 = x1 + bw, y1 + bh
        else:
            c = res["coords"]
            if len(c) < 4:
                continue
            x1 = c[0] * w_img / 1000
            y1 = c[1] * h_img / 1000
            x2 = c[2] * w_img / 1000
            y2 = c[3] * h_img / 1000
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        parsed.append(("box", label, color, x1, y1, x2, y2))

    for item in parsed:
        if item[0] == "box":
            _, _, color, x1, y1, x2, y2 = item
            fill_color = color + (65,)
            draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=color, width=4)
        elif item[0] == "point":
            _, _, color, cx, cy = item
            r = 10
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline="white", width=2)

    if draw_label and font:
        for item in parsed:
            if item[0] == "box":
                _, label, color, x1, y1, x2, y2 = item
                if not label:
                    continue
                t_box = draw.textbbox((0, 0), label, font=font)
                th = t_box[3] - t_box[1]
                tw = t_box[2] - t_box[0]
                pad_x, pad_y = 7, 4
                tag_h = th + pad_y * 2
                tag_w = tw + pad_x * 2
                tag_y = y1 - tag_h - 2
                if tag_y < 0:
                    tag_y = y2 + 2
                draw.rectangle([x1, tag_y, x1 + tag_w, tag_y + tag_h], fill=color)
                draw.text((x1 + pad_x, tag_y + pad_y), label, fill="white", font=font)
            elif item[0] == "point":
                _, label, color, cx, cy = item
                if not label:
                    continue
                t_box = draw.textbbox((0, 0), label, font=font)
                th, tw = t_box[3] - t_box[1], t_box[2] - t_box[0]
                tx, ty = cx + 14, cy - th // 2
                draw.rectangle([tx - 2, ty - 2, tx + tw + 6, ty + th + 4], fill=color)
                draw.text((tx + 2, ty), label, fill="white", font=font)

    combined = Image.alpha_composite(img_draw, overlay).convert("RGB")
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)


def postprocess_detections(detections, w, h):
    """Convert normalized coordinates to pixel coordinates."""
    valid = []
    for det in detections:
        if det["type"] == "box":
            c = det["coords"]
            rx1 = max(0, min(w - 1, int(c[0] * w / 1000)))
            ry1 = max(0, min(h - 1, int(c[1] * h / 1000)))
            rx2 = max(0, min(w - 1, int(c[2] * w / 1000)))
            ry2 = max(0, min(h - 1, int(c[3] * h / 1000)))
            box_w, box_h = rx2 - rx1, ry2 - ry1
            if box_w <= 0 or box_h <= 0:
                continue
            valid.append({"type": "box", "coords": [rx1, ry1, box_w, box_h],
                          "is_pixel": True, "label": det["label"]})
        elif det["type"] == "point":
            valid.append(det)
    return valid


class LocateAnythingWorker:
    """Stateful worker that loads the model once and serves perception queries."""

    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device).eval()

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        question: str,
        generation_mode: str = "hybrid",   # "fast" (MTP) | "slow" (NTP/AR) | "hybrid"
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        verbose: bool = True,
    ) -> dict:
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]}
        ]

        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        response = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            image_grid_hws=image_grid_hws,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=verbose,
        )

        result = {"answer": response[0] if isinstance(response, tuple) else response}
        if isinstance(response, tuple) and len(response) >= 3:
            result["history"] = response[1]
            result["stats"] = response[2]
        return result

    # Convenience methods for each task

    def detect(self, image: Image.Image, categories: list[str], **kwargs) -> dict:
        """Object detection / document layout analysis."""
        cats = "</c>".join(categories)
        prompt = f"Locate all the instances that matches the following description: {cats}."
        return self.predict(image, prompt, **kwargs)

    def ground_single(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Phrase grounding — single instance."""
        prompt = f"Locate a single instance that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def ground_multi(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Phrase grounding — multiple instances."""
        prompt = f"Locate all the instances that match the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def ground_text(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Text grounding."""
        prompt = f"Please locate the text referred as {phrase}."
        return self.predict(image, prompt, **kwargs)

    def detect_text(self, image: Image.Image, **kwargs) -> dict:
        """Scene text detection."""
        prompt = "Detect all the text in box format."
        return self.predict(image, prompt, **kwargs)

    def ground_gui(self, image: Image.Image, phrase: str, output_type: str = "box", **kwargs) -> dict:
        """GUI grounding (box or point)."""
        if output_type == "point":
            prompt = f"Point to: {phrase}."
        else:
            prompt = f"Locate the region that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def point(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Pointing."""
        prompt = f"Point to: {phrase}."
        return self.predict(image, prompt, **kwargs)

    # Utility: parse model output

    @staticmethod
    def parse_boxes(answer: str, image_width: int, image_height: int) -> list[dict]:
        """Parse model output into pixel-coordinate bounding boxes.

        Coordinates in model output are normalized integers in [0, 1000].
        """
        boxes = []
        for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
            x1, y1, x2, y2 = [int(g) for g in m.groups()]
            boxes.append({
                "x1": x1 / 1000 * image_width,
                "y1": y1 / 1000 * image_height,
                "x2": x2 / 1000 * image_width,
                "y2": y2 / 1000 * image_height,
            })
        return boxes

    @staticmethod
    def parse_points(answer: str, image_width: int, image_height: int) -> list[dict]:
        """Parse model output into pixel-coordinate points."""
        points = []
        for m in re.finditer(r"<box><(\d+)><(\d+)></box>", answer):
            x, y = int(m.group(1)), int(m.group(2))
            points.append({
                "x": x / 1000 * image_width,
                "y": y / 1000 * image_height,
            })
        return points

    @staticmethod
    def parse_text_boxes(answer: str, image_width: int, image_height: int) -> list[dict]:
        """Parse model output into text boxes with associated text.
        
        Handles cases where one <ref> label is followed by multiple <box> elements.
        Each box is associated with the most recent <ref> that came before it.
        Format: <ref>text content</ref><box><x1><y1><x2><y2></box>...
        """
        text_boxes = []
        
        # Find all ref tags and their end positions
        refs = [(m.group(1), m.end()) for m in re.finditer(r"<ref>([^<]*)</ref>", answer)]
        
        # Find all box tags with their coordinates and start positions
        boxes = [(m.group(1), m.group(2), m.group(3), m.group(4), m.start())
                 for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer)]
        
        for box_coords in boxes:
            x1_str, y1_str, x2_str, y2_str, box_pos = box_coords
            
            # Find the most recent ref that came before this box
            current_label = ""
            for ref_text, ref_end_pos in reversed(refs):
                if ref_end_pos <= box_pos:
                    current_label = ref_text
                    break
            
            x1, y1, x2, y2 = [int(g) for g in [x1_str, y1_str, x2_str, y2_str]]
            text_boxes.append({
                "text": current_label,
                "x1": x1 / 1000 * image_width,
                "y1": y1 / 1000 * image_height,
                "x2": x2 / 1000 * image_width,
                "y2": y2 / 1000 * image_height,
            })
        
        return text_boxes

    def detect_video(
        self,
        video_path: str,
        categories: list[str],
        output_path: str,
        generation_mode: str = "hybrid",
        target_fps: int = None,
        max_seconds: int = None,
        **kwargs
    ) -> dict:
        """Run detection on a video file.
        
        Args:
            video_path: Path to input video file
            categories: List of category strings to detect
            output_path: Path to save output video
            generation_mode: Generation mode for inference
            target_fps: Target FPS to sample (None = process all frames)
            max_seconds: Maximum seconds of video to process (None = process all)
            **kwargs: Additional arguments passed to predict()
        
        Returns:
            Dict with stats about the video processing
        """
        import gc
        
        # Read video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Read all frames
        all_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()
        
        total_frames = len(all_frames)
        if total_frames == 0:
            raise ValueError("Failed to read any frames from the video.")
        
        # Calculate max frames based on max_seconds
        if max_seconds is not None and max_seconds > 0:
            max_frames_limit = int(max_seconds * fps)
            all_frames = all_frames[:max_frames_limit]
            total_frames = len(all_frames)
        
        # Sample frames based on target_fps
        if target_fps is not None and target_fps > 0 and fps > target_fps:
            # Calculate sampling interval
            sample_interval = int(fps / target_fps)
            sampled_indices = list(range(0, total_frames, sample_interval))
            frames_to_process = [all_frames[i] for i in sampled_indices]
        else:
            # Process all frames
            sampled_indices = list(range(total_frames))
            frames_to_process = all_frames
        
        category_str = "</c>".join(categories)
        
        # Run inference on sampled frames
        inference_results = []
        processed_count = 0
        
        for i, frame in enumerate(frames_to_process):
            # Convert BGR to RGB for PIL
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Run inference
            result = self.detect(pil_img, categories, generation_mode=generation_mode, verbose=False, **kwargs)
            output_text = result["answer"]
            
            inference_results.append(output_text)
            processed_count += 1
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Draw detections and create output video
        # Calculate output FPS based on sampling
        if target_fps is not None and target_fps > 0 and fps > target_fps:
            out_fps = target_fps
        else:
            out_fps = fps
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, out_fps, (vid_w, vid_h))
        
        detections_summary = []
        
        for i, (frame, output_text) in enumerate(zip(frames_to_process, inference_results)):
            # Parse detections
            detections = parse_mixed_results(output_text, category_str)
            valid_results = postprocess_detections(detections, vid_w, vid_h)
            
            # Draw on frame
            frame_to_draw = draw_on_frame(frame, valid_results, draw_label=True)
            out.write(frame_to_draw)
            
            # Collect summary
            for det in valid_results:
                detections_summary.append({
                    "frame": sampled_indices[i] + 1,
                    "label": det.get("label", "object"),
                    "type": det.get("type", "box"),
                    "coords": det.get("coords", [])
                })
        
        out.release()
        
        stats = {
            "total_frames": total_frames,
            "sampled_frames": len(sampled_indices),
            "processed_frames": processed_count,
            "original_fps": fps,
            "target_fps": target_fps if target_fps else fps,
            "max_seconds": max_seconds,
            "processed_seconds": total_frames / fps if fps > 0 else 0,
            "output_fps": out_fps,
            "detections_count": len(detections_summary)
        }
        
        return stats
