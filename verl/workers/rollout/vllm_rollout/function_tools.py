from PIL import Image
from PIL import ImageDraw
import re
import json
from qwen_vl_utils import smart_resize
import os


def resize_image(image: Image.Image, save_path=None):
    original_width, original_height = image.size
    # Calculate the new dimensions (double the size)
    new_width = original_width * 2
    new_height = original_height * 2
    print(f"[TOOL CALL RESIZE IMAGE]: NEW_IMAGE_SIZE: {(new_width, new_height)}.")
    # Resize the image
    resized_image = image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
    if save_path:
        # Save the enlarged image
        resized_image.save(save_path)
    return resized_image

def check_bbox_format(bbox, w, h, MAX_RATIO=200):
    """
    Check whether a bounding box is in valid absolute format within image dimensions.

    Parameters:
        bbox (list): Bounding box in [x0, y0, x1, y1] format.
        w (int or float): Width of the image.
        h (int or float): Height of the image.

    Returns:
        (bool, str): (True, message) if valid; otherwise (False, error message).
    """
    if not isinstance(bbox, list):
        return False, "[WARNING] Bounding box must be a list."

    if len(bbox) != 4:
        return False, f"[WARNING] Bounding box must contain 4 elements: {bbox}"

    if not all(isinstance(coord, (int, float)) for coord in bbox):
        return False, f"[WARNING] All bounding box elements must be numeric: {bbox}"

    x0, y0, x1, y1 = bbox

    if not (0 <= x0 < w and 0 <= y0 < h and 0 < x1 <= w and 0 < y1 <= h):
        return False, f"[WARNING] Bounding box values must be within image bounds [0, 0, {w}, {h}]: {bbox}"

    if x1 <= x0 or y1 <= y0:
        return False, f"[WARNING] Invalid bounding box coordinates: x1 must be > x0 and y1 must be > y0: {bbox}"
    
    height, width = y1 - y0, x1 - x0
    if max(height, width) / min(height, width) > MAX_RATIO:
        return False, f"[WARNING] Bounding box ratio is too large: {max(height, width) / min(height, width)}"

    print(f"[INFO] Valid absolute bounding box: {bbox}")

    return True, "Bounding box format is valid."


def get_bbox_from_response(response: str, image):
    W, H = image.size

    tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
    tool_call_match = re.search(tool_call_pattern, response, re.DOTALL)
    
    if not tool_call_match:
        print("[WARNING] No tool_call found in outputs_string.")
        return None
    
    tool_call_content = tool_call_match.group(1).strip()
        
    try:
        tool_call_data = json.loads(tool_call_content)
        print(f"Raw tool_call_data: {tool_call_data}")
        if 'name' in tool_call_data and 'arguments' in tool_call_data and 'bbox_2d' in tool_call_data['arguments'] and len(tool_call_data['arguments']['bbox_2d']) == 4:
            bbox_content = tool_call_data["arguments"]["bbox_2d"]
            x0, y0, x1, y1 = bbox_content
        else:
            print(f"[WARNING] Invalid bbox: {tool_call_data}")
            return None
            
    except Exception as e:
        print(f"[WARNING] Failed to parse tool_call JSON: {e}")
        return None
    
    # is_valid, error_msg = check_bbox_format([x0_int, y0_int, x1_int, y1_int], W2, H2)
    
    scale_factor = 2
    x0_int = int(x0 * scale_factor)
    y0_int = int(y0 * scale_factor)
    x1_int = int(x1 * scale_factor)
    y1_int = int(y1 * scale_factor)
    W2, H2 = int(W * scale_factor), int(H * scale_factor)
    
    x0_int = max(0, min(x0_int, W2))
    x1_int = max(0, min(x1_int, W2))
    y0_int = max(0, min(y0_int, H2))
    y1_int = max(0, min(y1_int, H2))
    
    is_valid, error_msg = check_bbox_format([x0_int, y0_int, x1_int, y1_int], W2, H2)

    if not is_valid:
        print(f"[WARNING] Invalid bbox after clamping to current image size: {[x0_int, y0_int, x1_int, y1_int]} - {error_msg}")
        return None
    
    return [x0_int, y0_int, x1_int, y1_int]

def crop_image_by_bbox(image: Image.Image, bbox: list):
    """
    Crop an image using a bounding box.
    """
    cropped_image = image.crop(bbox)
    original_w, original_h = cropped_image.size
    new_h, new_w = smart_resize(height=original_h, width=original_w, factor=28, min_pixels=4*28*28, max_pixels=3840*3840)
    return_image = cropped_image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    print(f"[INFO] Cropped image size: {return_image.size}")
    return return_image

def prepare_tool_call_inputs(json_objects: list):
    if not json_objects:
        raise ValueError("Empty tool call: no JSON object found inside <tool_call>.")
    if len(json_objects) != 1:
        raise AssertionError("You should only call function `resize` once per function call.")
    obj = json_objects[0]
    action_type = obj.get('arguments', {}).get('action')
    assert action_type in ["resize"], f"Unknown Tool Type: {action_type}. Available function tools are: `resize`"
    return action_type

def prepare_multi_tool_call_inputs(json_objects: list):
    if not json_objects:
        raise ValueError("Empty tool call: no JSON object found inside <tool_call>.")
    if len(json_objects) != 1:
        raise AssertionError("You should only call function `request_high_res_image` or `request_local_region` once per function call.")
    obj = json_objects[0]
    tool_type = obj.get('name')
    assert tool_type in ["request_high_res_image", "request_local_region"], (
        f"Unknown Tool Type: {tool_type}. Available function tools are: `request_high_res_image`, `request_local_region`"
    )
    return tool_type


def prepare_bbox_tool_call_inputs(json_objects: list):
    if not json_objects:
        raise ValueError("Empty tool call: no JSON object found inside <tool_call>.")
    if len(json_objects) != 1:
        raise AssertionError("You should only call function `request_local_region` once per function call.")
    obj = json_objects[0]
    action_type = obj.get('name')
    assert action_type in ["request_local_region"], (
        f"Unknown Tool Type: {action_type}. Available function tools are: `request_local_region`"
    )
    bbox_2d = obj.get('arguments', {}).get('bbox_2d')
    assert bbox_2d is not None, "Missing `bbox_2d` in tool call arguments."
    assert len(bbox_2d) == 4, f"Bounding box must contain 4 elements: {bbox_2d}"
    assert all(isinstance(coord, (int, float)) for coord in bbox_2d), f"All bounding box elements must be numeric: {bbox_2d}"
    assert bbox_2d[2] > bbox_2d[0] and bbox_2d[3] > bbox_2d[1], (
        f"Invalid bounding box coordinates: x1 must be > x0 and y1 must be > y0: {bbox_2d}"
    )
    return action_type, bbox_2d

def save_bbox_image(image: Image.Image, bbox: list, save_dir: str, file_stem: str = None):
    if bbox is None or not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox for save_bbox_image: {bbox}")

    os.makedirs(save_dir, exist_ok=True)

    try:
        import time
        unique_suffix = str(int(time.time() * 1000))
    except Exception:
        unique_suffix = "000000"

    file_stem = file_stem or f"bbox_debug_{unique_suffix}"

    # Ensure RGB for drawing
    if image.mode != "RGB":
        base_img = image.convert("RGB")
    else:
        base_img = image

    raw_path = os.path.join(save_dir, f"{file_stem}_input.png")
    bbox_path = os.path.join(save_dir, f"{file_stem}_bbox.png")

    # Save original image
    try:
        base_img.save(raw_path)
    except Exception as e:
        print(f"[WARNING] Failed to save raw image to {raw_path}: {e}")

    # Draw bbox
    x0, y0, x1, y1 = [int(v) for v in bbox]
    drawn = base_img.copy()
    draw = ImageDraw.Draw(drawn)

    # Dynamic line width based on image size
    line_width = max(2, min(base_img.size) // 200)
    try:
        draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 0, 0), width=line_width)
    except TypeError:
        # Pillow < 8.2.0 may not support width parameter; fall back
        draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 0, 0))

    try:
        drawn.save(bbox_path)
    except Exception as e:
        print(f"[WARNING] Failed to save bbox image to {bbox_path}: {e}")

    # --- Save downscaled versions: 1/4-pixel and 1/16-pixel (i.e., 1/2x and 1/4x per side) ---
    try:
        w, h = base_img.size
        # 1/4 pixel count -> 1/2 on each side
        w_q4 = max(1, int(round(w * 0.5)))
        h_q4 = max(1, int(round(h * 0.5)))
        # 1/16 pixel count -> 1/4 on each side
        w_q16 = max(1, int(round(w * 0.25)))
        h_q16 = max(1, int(round(h * 0.25)))

        # Input downscaled
        input_q4_path = os.path.join(save_dir, f"{file_stem}_input_1_4px.png")
        input_q16_path = os.path.join(save_dir, f"{file_stem}_input_1_16px.png")
        base_img.resize((w_q4, h_q4), resample=Image.Resampling.LANCZOS).save(input_q4_path)
        base_img.resize((w_q16, h_q16), resample=Image.Resampling.LANCZOS).save(input_q16_path)

    except Exception as e:
        print(f"[WARNING] Failed to save downscaled images for {file_stem}: {e}")

    print(f"[DEBUG] Saved input image: {raw_path}; bbox image: {bbox_path}; bbox: {[x0, y0, x1, y1]}")
    return {"raw_path": raw_path, "bbox_path": bbox_path}