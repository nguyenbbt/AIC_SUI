from typing import List, Dict, Any

def group_and_order_regions(regions: List[Dict[Any, Any]]) -> List[Dict[Any, Any]]:
    """
    Groups OCR regions into lines and orders them from top-to-bottom, left-to-right.
    
    Args:
        regions (List[Dict]): List of region dictionaries. Each must have a 'bbox' key.
        
    Returns:
        List[Dict]: The ordered list of regions.
    """
    if not regions:
        return []
        
    # Calculate cy, cx, and height for each region
    augmented_regions = []
    for r in regions:
        bbox = r['bbox']
        # bbox is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        cy = sum(ys) / 4.0
        cx = sum(xs) / 4.0
        height = max(ys) - min(ys)
        augmented_regions.append({
            'original': r,
            'cy': cy,
            'cx': cx,
            'height': height
        })
        
    # Sort vertically by cy
    augmented_regions.sort(key=lambda x: x['cy'])
    
    lines = []
    current_line = []
    current_line_avg_cy = -1.0
    current_line_avg_height = -1.0
    
    for r in augmented_regions:
        if not current_line:
            current_line.append(r)
            current_line_avg_cy = r['cy']
            current_line_avg_height = r['height']
        else:
            # Check if region belongs to the current line
            threshold = 0.5 * current_line_avg_height
            if abs(r['cy'] - current_line_avg_cy) <= threshold:
                current_line.append(r)
                # Update averages
                current_line_avg_cy = sum(item['cy'] for item in current_line) / len(current_line)
                current_line_avg_height = sum(item['height'] for item in current_line) / len(current_line)
            else:
                # Start a new line
                lines.append(current_line)
                current_line = [r]
                current_line_avg_cy = r['cy']
                current_line_avg_height = r['height']
                
    if current_line:
        lines.append(current_line)
        
    ordered_regions = []
    # Sort horizontally within each line
    for line in lines:
        line.sort(key=lambda x: x['cx'])
        ordered_regions.extend([item['original'] for item in line])
        
    return ordered_regions

def concat_text(ordered_regions: List[Dict[Any, Any]]) -> str:
    """
    Concatenates text from ordered regions.
    
    Args:
        ordered_regions (List[Dict]): Regions ordered top-bottom, left-right.
        
    Returns:
        str: Concatenated text string.
    """
    texts = [r['text'].strip() for r in ordered_regions if r.get('text', '').strip()]
    return " ".join(texts)
