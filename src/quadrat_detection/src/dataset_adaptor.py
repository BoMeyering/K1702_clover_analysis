from pathlib import Path

from PIL.Image import open

import numpy as np

class QuadratDatasetAdaptor:
    def __init__(self, images_dir_path, annotations_dataframe):
        self.images_dir_path = Path(images_dir_path)
        self.annotations_df = annotations_dataframe
        self.images = self.annotations_df.img_id.unique().tolist()

    def __len__(self) -> int:
        return len(self.images)

    def get_image_and_labels_by_idx(self, index):
        image_name = self.images[index]
        image = open(self.images_dir_path / image_name)
        pascal_bboxes = self.annotations_df[self.annotations_df.img_id == image_name][
            ["xmin", "ymin", "xmax", "ymax"]
        ].values
        class_labels = np.ones(len(pascal_bboxes))

        return image, pascal_bboxes, class_labels, index
    
    # def show_image(self, index):
    #     image, bboxes, class_labels, image_id = self.get_image_and_labels_by_idx(index)
    #     print(f"image_id: {image_id}")
    #     show_image(image, bboxes.tolist())
    #     print(class_labels)