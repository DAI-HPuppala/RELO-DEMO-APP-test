"""
Utilities for drawing bounding boxes with colors and labels.
"""
import hashlib
from PIL import ImageDraw, ImageFont
from typing import Tuple, Optional

def get_color_for_target(target_type: str) -> Tuple[int, int, int]:
    """
    Map defect types to fixed colors, otherwise hash the string
    to get a consistent RGB tuple.
    """
    color_map = {
        'paint scratches': (255, 0, 0),    # red
        'dents': (0, 255, 0),              # green
        'cracks': (0, 0, 255),             # blue
        'chips': (255, 255, 0),            # yellow
        'rust': (255, 165, 0),             # orange
        'corrosion': (128, 0, 128),        # purple
    }
    
    target = target_type.lower()
    if target in color_map:
        return color_map[target]
    
    # Hash to RGB for unknown targets
    h = hashlib.md5(target_type.encode()).hexdigest()
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

def draw_box_with_label(
    draw: ImageDraw.ImageDraw,
    box: list,
    target: str,
    score: Optional[float] = None,
    width: int = 6,
    img_size: Tuple[int, int] = None
) -> None:
    """
    Draw a single bounding box with label and confidence score.
    """
    x0, y0, x1, y1 = box
    color = get_color_for_target(target)
    color_str = f"rgb{color}"
    
    # Draw the box
    draw.rectangle([(x0, y0), (x1, y1)], outline=color_str, width=width)
    
    # Prepare the label text
    text = target
    if score is not None:
        text = f"{target}: {score:.2f}"
    
    # Try to load Arial font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except:
        font = ImageFont.load_default().font_variant(size=48)
    
    # Get text size and padding
    padding = 8  # Increased padding for better appearance
    try:
        # Get exact text bounds
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except:
        # Fallback for older Pillow versions
        tw, th = draw.textsize(text, font=font)
    
    # Compute label box size
    label_w = tw + padding * 2
    label_h = th + padding * 2

    # Determine background rectangle placement
    if img_size:
        img_w, img_h = img_size
        positions = []
        # above
        if y0 - label_h >= 0 and x0 + label_w <= img_w:
            positions.append(('above', x0, y0 - label_h, x0 + label_w, y0))
        # below
        if y1 + label_h <= img_h and x0 + label_w <= img_w:
            positions.append(('below', x0, y1, x0 + label_w, y1 + label_h))
        # left
        if x0 - label_w >= 0 and y0 + label_h <= img_h:
            positions.append(('left', x0 - label_w, y0, x0, y0 + label_h))
        # right
        if x1 + label_w <= img_w and y0 + label_h <= img_h:
            positions.append(('right', x1, y0, x1 + label_w, y0 + label_h))
        # choose preferred order
        for pos in ['above','below','left','right']:
            match = next((p for p in positions if p[0]==pos), None)
            if match:
                _, bg_x0, bg_y0, bg_x1, bg_y1 = match
                break
        else:
            # fallback: clamp within image
            bg_x0 = max(0, min(x0, img_w - label_w))
            bg_y0 = max(0, min(y0 - label_h, img_h - label_h))
            bg_x1 = bg_x0 + label_w
            bg_y1 = bg_y0 + label_h
    else:
        # default: above if fits else below
        preferred_bg_y0 = y0 - label_h
        if preferred_bg_y0 >= 0:
            bg_x0 = x0
            bg_y0 = preferred_bg_y0
        else:
            bg_x0 = x0
            bg_y0 = y1
        bg_x1 = bg_x0 + label_w
        bg_y1 = bg_y0 + label_h

    # Draw text background
    draw.rectangle([(bg_x0, bg_y0), (bg_x1, bg_y1)], fill=color_str)

    # Center text in background box
    text_x = bg_x0 + padding
    text_y = bg_y0 + padding

    # Draw text
    draw.text((text_x, text_y), text, fill="white", font=font, anchor="lt")
