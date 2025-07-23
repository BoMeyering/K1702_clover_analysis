from shutil import copyfile
from glob import glob
import os

mat_dir = './data/annotations/all_mat_files'
mat_ids = [name[:-9] for name in sorted(glob("*.mat", root_dir=mat_dir))]

train_ids = [name[:-4] for name in sorted(glob("*.jpg", root_dir='./data/images/train'))]
val_ids = [name[:-4] for name in sorted(glob("*.jpg", root_dir='./data/images/val'))]
test_ids = [name[:-4] for name in sorted(glob("*.jpg", root_dir='./data/images/test'))]

for ID in mat_ids:
    if ID in train_ids:
        copyfile(
            src=os.path.join(mat_dir, ID + "_mask.mat"),
            dst=os.path.join("./data/annotations/train/", ID + "_mask.mat")
        )
    elif ID in val_ids:
        copyfile(
            src=os.path.join(mat_dir, ID + "_mask.mat"),
            dst=os.path.join("./data/annotations/val/", ID + "_mask.mat")
        )
    elif ID in test_ids:
        copyfile(
            src=os.path.join(mat_dir, ID + "_mask.mat"),
            dst=os.path.join("./data/annotations/test/", ID + "_mask.mat")
        )
