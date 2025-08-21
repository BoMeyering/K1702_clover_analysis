"""
barebones_testing script
Main Training Script for the Object Detection Model
BoMeyering 2025
"""

import torch
import cv2
import json
from tqdm import tqdm
import polars as pl
import albumentations as A
from albumentations import ToTensorV2
from typing import Tuple
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import SGD
from pathlib import Path
from effdet import DetBenchTrain, DetBenchPredict, EfficientDet, get_efficientdet_config
from effdet.efficientdet import HeadNet

DATA_SPLIT = 'train'

model_config = {
    "image_size": (512, 512),
    "architecture": "tf_efficientdet_d0",
    "pretrained": True,
    "num_classes": 1,   # excluding background class
    "max_det_per_image": 200
}

def show_image(img):
    cv2.namedWindow('test', cv2.WINDOW_GUI_NORMAL)
    cv2.imshow('test', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def create_effdet_model(num_classes: int = 1,
                        image_size: tuple = (512, 512),
                        architecture: str = 'efficientdet_d0'
                        ):

    config = get_efficientdet_config(architecture)
    config.update({
        'num_classes': num_classes,
        'image_size': image_size,
    })

    print(config)


    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(
        config=config, 
        num_outputs=config.num_classes
    )

    return DetBenchTrain(net) # don't think we need to pass the config as an argument here


def get_train_transforms(resize: tuple[int, int]=(512, 512)):
    transforms = A.Compose(
        [
            A.Resize(resize[0], resize[1], p=1.0),
            A.Normalize(),
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["labels"]
        )
    )

    return transforms

def read_annotations(filepath='data/bboxes.csv', split=DATA_SPLIT, filter=None):
    
    df = pl.read_csv(filepath)
    split_df = pl.read_csv('data/data_split.csv')\
        .filter(pl.col('split')==split)

    df = df.join(split_df.select('img_id'), on='img_id', how='inner')

    if filter:
        df = df.filter(pl.col('class').is_in(filter))
    
    df, map_dict = _map_labels(df)

    # output map_dict for testing
    with open('test_mapping.json', 'w') as f:
        json.dump(map_dict, f)

    return df

def _map_labels(df):
    unique_labels = df['class'].unique().to_list()
    map_dict = {unique_labels[i]: i+1 for i in range(len(unique_labels))}

    df = df.with_columns(
        pl.col('class').replace(map_dict).alias('label').cast(pl.Int64)
    )

    print(f"Unique labels are {unique_labels}")

    return df, map_dict


def collate_fn(batch):
    imgs, targets, img_ids = tuple(zip(*batch))

    imgs = torch.stack(imgs).float()

    bboxes = [target['bboxes'].float() for target in targets]
    labels = [target['labels'].float() for target in targets]
    img_size = torch.tensor([target['img_size'] for target in targets]).float()
    img_scale = torch.tensor([target['img_scale'] for target in targets]).float()

    annotations = {
        'bbox': bboxes,
        'cls': labels,
        'img_size': img_size,
        'img_scale': img_scale
    }

    return imgs, annotations, targets, img_ids


def move_to_device(obj, device):
    """ Recursive function to move targets to device """
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to_device(o, device) for o in obj]
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    else:
        return obj


class KuraDataset(Dataset):
    def __init__(self, annotations, transforms, split: str=DATA_SPLIT):
        self.annotations = annotations
        self.img_ids = self.annotations['img_id'].unique().to_list()
        self.transforms = transforms

    def __getitem__(self, idx):
        
        # Grab img_id and annotations
        img_id = self.img_ids[idx]
        id_anns = self.annotations.filter(pl.col('img_id')==img_id)
        img_name = id_anns['img_name'].unique().to_list()[0]
        bboxes = id_anns[:, [5, 6, 7, 8]].to_numpy()
        labels = id_anns[:, 9].to_numpy()

        # Read in image from path
        img_path = Path('data/processed/images') / img_name
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR_RGB)

        sample = self.transforms(image=img, bboxes=bboxes, labels=labels)
        
        img = sample['image']
        bboxes = sample['bboxes']
        bboxes = bboxes[:, [1, 0, 3, 2]] # Reorder to yxyx format
        labels = sample['labels']
        _, new_h, new_w = img.shape

        target = {
            'bboxes': torch.as_tensor(bboxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels),
            'img_size': (new_h, new_w),
            'img_scale': torch.tensor([1.0])
        }

        return img, target, img_id

    def __len__(self):
        return len(self.img_ids)




def main():
    # Get device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Create Effdet Train model
    model = create_effdet_model().to(device)
    model.train()

    optimizer = SGD(model.parameters())

    # Read in all annotations and filter
    annotations = read_annotations(split='train', filter=['quadrat_corner'])


    # Make transforms function
    transforms = get_train_transforms()
    print(transforms)

    # Create training dataset
    train_ds = KuraDataset(annotations=annotations, transforms=transforms)
    print(train_ds)

    # Create a dataloader
    train_dl = DataLoader(
        train_ds, 
        batch_size=3,
        shuffle=True,
        num_workers=0, 
        collate_fn=collate_fn
    )

    train_loader = iter(train_dl)

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), dynamic_ncols=True)
    for batch_idx, batch in pbar:
        imgs, annotations, _, img_ids = batch

        imgs = imgs.to(device)
        annotations = move_to_device(annotations, device)

        optimizer.zero_grad()
        out = model(imgs, annotations)
        loss = out['loss']
        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]['lr']
        pbar.set_description(f"Epoch 1 Loss: {loss.item():.4f}")
        pbar.set_postfix(loss=f"lr={lr:.2e}")


if __name__ == '__main__':
    main()




