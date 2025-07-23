from scipy import io
from glob import glob
from pathlib import Path
import numpy as np
import cv2

ROOT_DIR = Path('data/annotations/all_mat_files')
IMG_DIR = Path('data/images/all_images')
filenames = glob("*.mat", root_dir=ROOT_DIR)

print(filenames)


for filename in filenames:
    mask_array = io.loadmat(ROOT_DIR / filename)
    basename = filename.split('.')[0]
    name_list = basename.split('_')

    accession = name_list[0]

    if len(name_list) > 2:
        name_list = [accession, '20170608']
        img_array = cv2.imread(IMG_DIR / ("_".join([accession, "6_8_17"]) + ".jpg"))
    else:
        name_list = [accession, '20170703']
        img_array = cv2.imread(IMG_DIR / (accession + ".jpg"))
    basename = '_'.join(name_list)
    out_path = Path('masks_png') / (basename + '.png')
    cv2.imwrite(out_path, mask_array['data'])

    out_path = Path('images_jpg') / (basename + '.jpg')
    cv2.imwrite(out_path, img_array)
