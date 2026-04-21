
import os
import json
import numpy as np
from PIL import Image, ImageDraw

# Configuration
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MASKS_DIR = os.path.join(DATA_DIR, "masks")

# Mapping from Cityscapes label names to 8 classes
# classes: 0:flat, 1:human, 2:vehicle, 3:construction, 4:object, 5:nature, 6:sky, 7:void
LABEL_TO_CLASS = {
    'road': 0, 'sidewalk': 0, 'parking': 0, 'rail track': 0,
    'person': 1, 'rider': 1,
    'car': 2, 'truck': 2, 'bus': 2, 'on rails': 2, 'motorcycle': 2, 'bicycle': 2, 'caravan': 2, 'trailer': 2, 'train': 2,
    'building': 3, 'wall': 3, 'fence': 3, 'guard rail': 3, 'bridge': 3, 'tunnel': 3,
    'pole': 4, 'polegroup': 4, 'traffic sign': 4, 'traffic light': 4,
    'vegetation': 5, 'terrain': 5,
    'sky': 6,
    'unlabeled': 7, 'ego vehicle': 7, 'rectification border': 7, 'out of roi': 7, 'static': 7, 'dynamic': 7, 'ground': 7, 'license plate': 7
}

# Palette de couleurs Cityscapes (8 classes)
# Correspond à app/ui/app.py
PALETTE = [
    (128, 64, 128),  # 0: flat - Violet
    (220, 20, 60),   # 1: human - Rouge
    (0, 0, 142),     # 2: vehicle - Bleu
    (70, 70, 70),    # 3: construction - Gris
    (220, 220, 0),   # 4: object - Jaune
    (107, 142, 35),  # 5: nature - Vert
    (70, 130, 180),  # 6: sky - Ciel
    (0, 0, 0)        # 7: void - Noir
]

def generate_mask_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    height = data['imgHeight']
    width = data['imgWidth']
    
    # Create an empty RGB mask initialized to Void (Black)
    mask_img = Image.new('RGB', (width, height), PALETTE[7])
    draw = ImageDraw.Draw(mask_img)
    
    for obj in data['objects']:
        label = obj['label']
        class_id = LABEL_TO_CLASS.get(label, 7) # Default to Void if not found
        color = PALETTE[class_id]
        
        polygon = [tuple(p) for p in obj['polygon']]
        if len(polygon) < 2:
            continue
        draw.polygon(polygon, fill=color)
        
    return mask_img

def main():
    json_files = [f for f in os.listdir(MASKS_DIR) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files to process.")
    
    for json_file in json_files:
        json_path = os.path.join(MASKS_DIR, json_file)
        print(f"Processing {json_file}...")
        
        try:
            colored_mask = generate_mask_from_json(json_path)
            
            # Save the colored mask as PNG, replacing the previous black one
            # The UI expects _gtFine_labelIds.png suffix
            output_name = json_file.replace('_polygons.json', '_labelIds.png')
            output_path = os.path.join(MASKS_DIR, output_name)
            
            colored_mask.save(output_path)
            print(f"  Saved to {output_name}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  Error processing {json_file}: {e}")

if __name__ == "__main__":
    main()
