from ocr_module.text_ordering import group_and_order_regions, concat_text

def test_group_and_order_regions():
    # Mock regions:
    # Line 1: Box 1 (x=10), Box 2 (x=100) -> cy around 20
    # Line 2: Box 3 (x=50) -> cy around 60
    
    regions = [
        {"id": 3, "text": "World", "bbox": [[50, 50], [90, 50], [90, 70], [50, 70]]},  # Line 2, Middle
        {"id": 2, "text": "There", "bbox": [[100, 10], [150, 10], [150, 30], [100, 30]]}, # Line 1, Right
        {"id": 1, "text": "Hello", "bbox": [[10, 15], [40, 15], [40, 35], [10, 35]]}    # Line 1, Left (slightly lower y)
    ]
    
    ordered = group_and_order_regions(regions)
    
    assert len(ordered) == 3
    assert ordered[0]["id"] == 1
    assert ordered[1]["id"] == 2
    assert ordered[2]["id"] == 3
    
    text = concat_text(ordered)
    assert text == "Hello There World"

def test_empty_regions():
    assert group_and_order_regions([]) == []
    assert concat_text([]) == ""
