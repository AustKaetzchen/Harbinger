import gradio as gr
import torch
import tempfile
import os

from PIL import Image, ImageDraw
from inference import LocateAnythingWorker


# Initialize the worker (will load model on startup)
worker = LocateAnythingWorker("nvidia/LocateAnything-3B", device="cuda" if torch.cuda.is_available() else "cpu")


def draw_boxes_on_image(image: Image.Image, boxes: list[dict], color: str = "red", width: int = 3) -> Image.Image:
    """Draw bounding boxes on the image."""
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(
            [(box["x1"], box["y1"]), (box["x2"], box["y2"])],
            outline=color,
            width=width
        )
    return image


def draw_points_on_image(image: Image.Image, points: list[dict], color: str = "red", radius: int = 5) -> Image.Image:
    """Draw points on the image."""
    draw = ImageDraw.Draw(image)
    for point in points:
        draw.ellipse(
            [(point["x"] - radius, point["y"] - radius), (point["x"] + radius, point["y"] + radius)],
            fill=color,
            outline=color
        )
    return image


def draw_text_boxes_on_image(image: Image.Image, text_boxes: list[dict], color: str = "blue", width: int = 3) -> Image.Image:
    """Draw text boxes with text annotations on the image."""
    draw = ImageDraw.Draw(image)
    for item in text_boxes:
        # Draw bounding box
        draw.rectangle(
            [(item["x1"], item["y1"]), (item["x2"], item["y2"])],
            outline=color,
            width=width
        )
        # Draw text above the box
        text = item.get("text", "")
        if text:
            text_x = item["x1"]
            text_y = max(0, item["y1"] - 20)  # Position text above the box
            # Draw a semi-transparent background for the text for better readability
            bbox = draw.textbbox((text_x, text_y), text)
            draw.rectangle(bbox, fill=color, outline=color)
            draw.text((text_x, text_y), text, fill="white", font=None)
    return image


def process_detection(image, categories_text, generation_mode):
    """Process object detection."""
    if image is None:
        return None, "Please upload an image."
    
    categories = [cat.strip() for cat in categories_text.split(",") if cat.strip()]
    if not categories:
        return None, "Please enter at least one category."
    
    result = worker.detect(image, categories, generation_mode=generation_mode, verbose=False)
    answer = result["answer"]
    
    w, h = image.size
    text_boxes = LocateAnythingWorker.parse_text_boxes(answer, w, h)
    boxes = LocateAnythingWorker.parse_boxes(answer, w, h)
    points = LocateAnythingWorker.parse_points(answer, w, h)
    
    # Draw results on image
    result_image = image.copy()
    if text_boxes:
        # Use labeled boxes with annotations if available
        result_image = draw_text_boxes_on_image(result_image, text_boxes, color="red")
    elif boxes:
        # Fall back to plain boxes if labeled boxes parsing didn't work
        result_image = draw_boxes_on_image(result_image, boxes)
    if points:
        result_image = draw_points_on_image(result_image, points)
    
    return result_image, f"Raw Output:\n{answer}\n\nDetected {len(text_boxes)} boxes and {len(points)} points."


def process_grounding(image, phrase, grounding_type, generation_mode):
    """Process phrase grounding."""
    if image is None:
        return None, "Please upload an image."
    
    if not phrase.strip():
        return None, "Please enter a phrase to ground."
    
    if grounding_type == "single":
        result = worker.ground_single(image, phrase, generation_mode=generation_mode, verbose=False)
    else:  # multi
        result = worker.ground_multi(image, phrase, generation_mode=generation_mode, verbose=False)
    
    answer = result["answer"]
    
    w, h = image.size
    boxes = LocateAnythingWorker.parse_boxes(answer, w, h)
    points = LocateAnythingWorker.parse_points(answer, w, h)
    
    # Draw results on image
    result_image = image.copy()
    if boxes:
        result_image = draw_boxes_on_image(result_image, boxes)
    if points:
        result_image = draw_points_on_image(result_image, points)
    
    return result_image, f"Raw Output:\n{answer}\n\nDetected {len(boxes)} boxes and {len(points)} points."


def process_text_detection(image, generation_mode):
    """Process scene text detection."""
    if image is None:
        return None, "Please upload an image."
    
    result = worker.detect_text(image, generation_mode=generation_mode, verbose=False)
    answer = result["answer"]
    
    w, h = image.size
    text_boxes = LocateAnythingWorker.parse_text_boxes(answer, w, h)
    boxes = LocateAnythingWorker.parse_boxes(answer, w, h)
    points = LocateAnythingWorker.parse_points(answer, w, h)
    
    # Draw results on image
    result_image = image.copy()
    if text_boxes:
        # Use text boxes with annotations if available
        result_image = draw_text_boxes_on_image(result_image, text_boxes, color="blue")
    elif boxes:
        # Fall back to plain boxes if text boxes parsing didn't work
        result_image = draw_boxes_on_image(result_image, boxes, color="blue")
    if points:
        result_image = draw_points_on_image(result_image, points, color="blue")
    
    return result_image, f"Raw Output:\n{answer}\n\nDetected {len(text_boxes)} text regions and {len(points)} points."


def process_pointing(image, phrase, generation_mode):
    """Process pointing task."""
    if image is None:
        return None, "Please upload an image."
    
    if not phrase.strip():
        return None, "Please enter a phrase to point to."
    
    result = worker.point(image, phrase, generation_mode=generation_mode, verbose=False)
    answer = result["answer"]
    
    w, h = image.size
    boxes = LocateAnythingWorker.parse_boxes(answer, w, h)
    points = LocateAnythingWorker.parse_points(answer, w, h)
    
    # Draw results on image
    result_image = image.copy()
    if boxes:
        result_image = draw_boxes_on_image(result_image, boxes, color="green")
    if points:
        result_image = draw_points_on_image(result_image, points, color="green", radius=8)
    
    return result_image, f"Raw Output:\n{answer}\n\nDetected {len(boxes)} boxes and {len(points)} points."


def process_gui_grounding(image, phrase, output_type, generation_mode):
    """Process GUI grounding task."""
    if image is None:
        return None, "Please upload an image."
    
    if not phrase.strip():
        return None, "Please enter a GUI element description."
    
    result = worker.ground_gui(image, phrase, output_type=output_type, generation_mode=generation_mode, verbose=False)
    answer = result["answer"]
    
    w, h = image.size
    boxes = LocateAnythingWorker.parse_boxes(answer, w, h)
    points = LocateAnythingWorker.parse_points(answer, w, h)
    
    # Draw results on image
    result_image = image.copy()
    if boxes:
        result_image = draw_boxes_on_image(result_image, boxes, color="purple")
    if points:
        result_image = draw_points_on_image(result_image, points, color="purple", radius=8)
    
    return result_image, f"Raw Output:\n{answer}\n\nDetected {len(boxes)} boxes and {len(points)} points."


def process_video_detection(video, categories_text, generation_mode, target_fps, max_seconds):
    """Process video detection."""
    if video is None:
        return None, "Please upload a video."
    
    categories = [cat.strip() for cat in categories_text.split(",") if cat.strip()]
    if not categories:
        return None, "Please enter at least one category."
    
    # Create temp file for output video
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "output.mp4")
    
    try:
        # Run video detection
        stats = worker.detect_video(
            video, 
            categories, 
            output_path, 
            generation_mode=generation_mode,
            target_fps=target_fps if target_fps > 0 else None,
            max_seconds=max_seconds if max_seconds > 0 else None
        )
        
        summary = (
            f"Video Processing Complete!\n\n"
            f"Total frames: {stats['total_frames']}\n"
            f"Sampled frames: {stats['sampled_frames']}\n"
            f"Processed frames: {stats['processed_frames']}\n"
            f"Original FPS: {stats['original_fps']:.2f}\n"
            f"Target FPS: {stats['target_fps']:.2f}\n"
            f"Max seconds: {stats['max_seconds'] if stats['max_seconds'] else 'No limit'}\n"
            f"Processed seconds: {stats['processed_seconds']:.2f}\n"
            f"Output FPS: {stats['output_fps']:.2f}\n"
            f"Total detections: {stats['detections_count']}"
        )
        
        return output_path, summary
    except Exception as e:
        return None, f"Error processing video: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="LocateAnything - Vision-Language Grounding") as demo:
    gr.Markdown("# LocateAnything - Vision-Language Grounding")
    gr.Markdown("Upload an image or video and use one of the tasks below to locate objects, text, or GUI elements.")
    
    with gr.Tabs():
        # Object Detection Tab
        with gr.TabItem("Object Detection"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="pil", label="Upload Image")
                    categories_input = gr.Textbox(
                        label="Categories (comma-separated)",
                        placeholder="e.g., person, car, dog, cat"
                    )
                    generation_mode = gr.Radio(
                        choices=["fast", "slow", "hybrid"],
                        value="hybrid",
                        label="Generation Mode"
                    )
                    detect_btn = gr.Button("Detect Objects", variant="primary")
                with gr.Column():
                    output_image = gr.Image(label="Result with Bounding Boxes")
                    output_text = gr.Textbox(label="Output", lines=5)
            
            detect_btn.click(
                fn=process_detection,
                inputs=[input_image, categories_input, generation_mode],
                outputs=[output_image, output_text]
            )
        
        # Phrase Grounding Tab
        with gr.TabItem("Phrase Grounding"):
            with gr.Row():
                with gr.Column():
                    grounding_image = gr.Image(type="pil", label="Upload Image")
                    phrase_input = gr.Textbox(
                        label="Phrase to locate",
                        placeholder="e.g., people wearing red shirts"
                    )
                    grounding_type = gr.Radio(
                        choices=["single", "multi"],
                        value="multi",
                        label="Grounding Type"
                    )
                    grounding_mode = gr.Radio(
                        choices=["fast", "slow", "hybrid"],
                        value="hybrid",
                        label="Generation Mode"
                    )
                    ground_btn = gr.Button("Ground Phrase", variant="primary")
                with gr.Column():
                    grounding_output_image = gr.Image(label="Result with Bounding Boxes")
                    grounding_output_text = gr.Textbox(label="Output", lines=5)
            
            ground_btn.click(
                fn=process_grounding,
                inputs=[grounding_image, phrase_input, grounding_type, grounding_mode],
                outputs=[grounding_output_image, grounding_output_text]
            )
        
        # Text Detection Tab
        with gr.TabItem("OCR / Text Detection"):
            with gr.Row():
                with gr.Column():
                    text_image = gr.Image(type="pil", label="Upload Image")
                    text_mode = gr.Radio(
                        choices=["fast", "slow", "hybrid"],
                        value="hybrid",
                        label="Generation Mode"
                    )
                    text_btn = gr.Button("Detect Text", variant="primary")
                with gr.Column():
                    text_output_image = gr.Image(label="Result with Text Boxes")
                    text_output_text = gr.Textbox(label="Output", lines=5)
            
            text_btn.click(
                fn=process_text_detection,
                inputs=[text_image, text_mode],
                outputs=[text_output_image, text_output_text]
            )
        
        # Pointing Tab
        with gr.TabItem("Pointing"):
            with gr.Row():
                with gr.Column():
                    point_image = gr.Image(type="pil", label="Upload Image")
                    point_phrase = gr.Textbox(
                        label="Phrase to point to",
                        placeholder="e.g., the traffic light"
                    )
                    point_mode = gr.Radio(
                        choices=["fast", "slow", "hybrid"],
                        value="hybrid",
                        label="Generation Mode"
                    )
                    point_btn = gr.Button("Point to Object", variant="primary")
                with gr.Column():
                    point_output_image = gr.Image(label="Result with Point")
                    point_output_text = gr.Textbox(label="Output", lines=5)
            
            point_btn.click(
                fn=process_pointing,
                inputs=[point_image, point_phrase, point_mode],
                outputs=[point_output_image, point_output_text]
            )
        
        # GUI Grounding Tab
        with gr.TabItem("GUI Grounding"):
            with gr.Row():
                with gr.Column():
                    gui_image = gr.Image(type="pil", label="Upload Image")
                    gui_phrase = gr.Textbox(
                        label="GUI element description",
                        placeholder="e.g., the search button"
                    )
                    gui_output_type = gr.Radio(
                        choices=["box", "point"],
                        value="box",
                        label="Output Type"
                    )
                    gui_mode = gr.Radio(
                        choices=["fast", "slow", "hybrid"],
                        value="hybrid",
                        label="Generation Mode"
                    )
                    gui_btn = gr.Button("Locate GUI Element", variant="primary")
                with gr.Column():
                    gui_output_image = gr.Image(label="Result")
                    gui_output_text = gr.Textbox(label="Output", lines=5)
            
            gui_btn.click(
                fn=process_gui_grounding,
                inputs=[gui_image, gui_phrase, gui_output_type, gui_mode],
                outputs=[gui_output_image, gui_output_text]
            )
        
        # Video Detection Tab
        with gr.TabItem("Video Detection"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Upload Video")
                    video_categories = gr.Textbox(
                        label="Categories (comma-separated)",
                        placeholder="e.g., person, car, dog, cat"
                    )
                    video_mode = gr.Radio(
                        choices=["fast", "slow", "hybrid"],
                        value="hybrid",
                        label="Generation Mode"
                    )
                    video_target_fps = gr.Slider(
                        minimum=0,
                        maximum=30,
                        value=5,
                        step=1,
                        label="Target FPS (0 = all frames)"
                    )
                    video_max_seconds = gr.Slider(
                        minimum=0,
                        maximum=60,
                        value=10,
                        step=1,
                        label="Max Seconds (0 = no limit)"
                    )
                    video_btn = gr.Button("Process Video", variant="primary")
                with gr.Column():
                    video_output = gr.Video(label="Result Video")
                    video_output_text = gr.Textbox(label="Output", lines=12)
            
            video_btn.click(
                fn=process_video_detection,
                inputs=[video_input, video_categories, video_mode, video_target_fps, video_max_seconds],
                outputs=[video_output, video_output_text]
            )
    
    gr.Markdown("## Model: nvidia/LocateAnything-3B")
    gr.Markdown("For more information, visit [Hugging Face](https://huggingface.co/nvidia/LocateAnything-3B)")


if __name__ == "__main__":
    demo.launch()
