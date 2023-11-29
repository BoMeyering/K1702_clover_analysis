# Mask Generation with SAM

# Import statements
import torch
import wget
import sys
import cv2
import pandas as pd
import numpy as np
from glob import glob
import os
from tqdm.auto import tqdm
import sys
import scipy
from segment_anything import sam_model_registry, SamPredictor

# Configuration
sys.path.append("..")

IMG_DIR = './data/images/all_images'
OUT_DIR = './src/mask_generation/output'
MASK_DIR = './data/annotations/mat_files'
AN_PATH = './data/annotations/SAM_points.csv'

PATHS = glob(IMG_DIR+'/*.jpg')
PATHS = sorted(PATHS)
FILES = [path.split('/')[-1] for path in PATHS]

ALPHA = .5
class_dict: dict[str, tuple] = {
	'quadrat_point': [(255, 0, 0), 1],
	'soil': [(153, 51, 255), 2],
	'kura_clover_point': [(0, 200, 0), 3]
}

if not os.path.exists(OUT_DIR):
	os.mkdir(OUT_DIR)

points = pd.read_csv(AN_PATH)

print(f"This system has a CUDA capable device: {torch.cuda.is_available()}")

# SAM configuration and registration
SAM_path = './src/mask_generation/SAM_checkpoint/sam_vit_h_4b8939.pth'
if not os.path.exists(SAM_path):
	url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
	print(f"Retrieving SAM model checkpoint from {url}")
	wget.download(url, out=SAM_path)
else:
	print("Found SAM model checkpoint")

model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Registering model checkpoint with SAM registry and sending model to device: {device}")
sam = sam_model_registry[model_type](checkpoint=SAM_path)
sam.to(device=device)
predictor = SamPredictor(sam)

print("Generating masks from point prompts and saving to .mat format.")

# Main inference function
def main(points):
	for i, file in enumerate(tqdm(FILES)):
		print(i, file)
		if file not in points['img_id'].unique():
			print(f"{file} not in annotations. Skipping for now.")
			continue
		elif os.path.exists(os.path.join(MASK_DIR, f"{file[:-4]}_mask.mat")):
			print(f"{file} has already been processed. Passing")
			continue
		else:
			sub_points = points[points['img_id']==file]
			img = cv2.imread(PATHS[i], cv2.COLOR_BGR2RGB)
			splash_img = img.copy()
			assert file == PATHS[i].split('/')[-1]
			predictor.set_image(img) # Send img to model predictor
			mask = np.zeros(shape = img.shape[0:2], dtype=np.uint8)
			if len(sub_points) == 0:
				continue
			else:
				for i in class_dict.keys():
					class_rgb = class_dict[i][0]
					class_code = class_dict[i][1]
					class_points = sub_points[sub_points['class']==i].loc[:,['x', 'y']].to_numpy()
					labels = np.array([1]*len(class_points))

					# Generate mask predictions from point prompts
					masks, scores, logits = predictor.predict(
						point_coords=class_points,
						point_labels=labels, 
						multimask_output=True
					)
					max_idx = np.argmax(scores)
					class_mask = masks[max_idx]
					
					# Construct a filtered mask where no pixels have been assigned yet
					class_mask = np.bitwise_and(np.where(mask == 0, 255, 0), np.where(class_mask > 0, 255, 0)).astype(np.int64)
					mask[np.where(class_mask > 0)] = class_code

					# Create a filled mask and overlay on image
					colored_mask = np.expand_dims(class_mask, 0).repeat(3, axis=0)
					colored_mask = np.moveaxis(colored_mask, 0, -1)
					masked = np.ma.MaskedArray(splash_img, mask=colored_mask, fill_value=class_rgb)
					image_overlay = masked.filled()
					splash_img = cv2.addWeighted(splash_img, 1 - ALPHA, image_overlay, ALPHA, 0)
			
				cv2.imwrite(os.path.join(OUT_DIR, f"{file[:-4]}_masked.jpg"), splash_img)
				scipy.io.savemat(os.path.join(MASK_DIR, file[:-4] + "_mask.mat"), {'data': mask})

if __name__ == '__main__':
	main(points=points)
	print("Mask inference complete")
