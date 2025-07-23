# Move all mat files down one index
# Export to csv

import numpy as np
import os
from scipy.io import loadmat, savemat
from glob import glob
import cv2

ALL_ANN_PATH = 'all_mat_files'
TEST_PATH = 'test'
TRAIN_PATH = 'train'
VAL_PATH = 'val'

DIR_PATHS = [ALL_ANN_PATH, TEST_PATH, TRAIN_PATH, VAL_PATH]
OUTROOT = 'output'

def reduce_indices(img):

	unique_classes = np.unique(img)

	zero_mask = np.zeros(img.shape).astype(np.uint8)

	for idx in unique_classes:
		if idx == 0 or idx == 2:
			continue
		elif idx == 1:
			class_indices = np.where(img == idx)
			zero_mask[class_indices] = 1
		elif idx == 3:
			class_indices = np.where(img == idx)
			zero_mask[class_indices] = 2

	return zero_mask

def show_img(img):
	cv2.namedWindow('test', cv2.WINDOW_NORMAL)
	cv2.imshow('test', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	for root_dir in DIR_PATHS:
		print(root_dir)
		filenames = glob('*', root_dir=root_dir)
		for filename in filenames:
			mat = loadmat(os.path.join(ALL_ANN_PATH, filename))
			array = mat['data']

			new_mask = reduce_indices(array)

			out_path = os.path.join(OUTROOT, root_dir, filename)

			mat_dict = {
				'data': new_mask,
				'project': 'K1702'
			}

			savemat(out_path, mat_dict)

