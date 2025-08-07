import cv2
import json
import os
import pandas as pd
import numpy as np



img_paths = [
    'data/processed/images/Ta00070_20170703.jpg',
    'data/processed/images/Ta00079_20170703.jpg',
    'data/processed/images/Ta00058_20170703.jpg',
    'data/processed/images/Ta00696_20170703.jpg'

]
mask_paths = [
    'data/processed/targets/Ta00070_20170703.png',
    'data/processed/targets/Ta00079_20170703.png',
    'data/processed/targets/Ta00058_20170703.png',
    'data/processed/targets/Ta00696_20170703.png'
]

bbox_path = 'data/bboxes.csv'

SCALE_FACTOR = 0.005
PADDING = 0.005
FONT_THICKNESS = 0.001

color_map = np.array(
    [
        [20, 66, 112],     # Class 0
        [255, 255, 255],    # Class 1
        [100, 200, 0]       # Class 2
    ],
    dtype=np.uint8
)

bbox_color_map = {
    'clover': (0, 239, 255),
    'quadrat': (67, 179, 255),
    'quadrat_corner': (255, 100, 0)
}


df = pd.read_csv(bbox_path)

for img_path, mask_path in zip(img_paths, mask_paths):
    img = cv2.imread(img_path)
    img_name = os.path.basename(img_path)

    # Read in grayscale mask and convert to binary mask with numpy broadcasting
    bin_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    color_mask = color_map[bin_mask]

    overlay = cv2.addWeighted(src1=img.copy(), alpha=.5, src2=color_mask.copy(), beta=0.5, gamma=0.1)

    bboxes = df[df['img_name'] == img_name]
    
    for row in bboxes.iterrows():
        row_data = row[1]

        label, x1, y1, x2, y2 = row_data['class'], int(row_data.x1), int(row_data.y1), int(row_data.x2), int(row_data.y2)
        color = bbox_color_map.get(label)
        cv2.rectangle(overlay, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=int(img.shape[0]*SCALE_FACTOR))
        
        (text_width, text_height), baseline = cv2.getTextSize(label, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.3, thickness=int(img.shape[0]*FONT_THICKNESS))
        
        padding = int(PADDING*img.shape[0])

        rx_start = x1
        ry_start = y1
        rx_end = x1 + text_width + 2*padding
        ry_end = y1 + text_height + 2*padding

        cv2.rectangle(overlay, (rx_start, ry_start), (rx_end, ry_end), (0, 0, 0), cv2.FILLED)
        cv2.putText(overlay, label, (x1 + padding, y1 + 2*padding), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.3, color=(255, 255, 255), thickness=int(img.shape[0]*FONT_THICKNESS), lineType=cv2.LINE_AA)

        out_path = f'assets/{img_name}'
        cv2.imwrite(out_path, overlay)
