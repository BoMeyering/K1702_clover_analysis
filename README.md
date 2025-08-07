# K1702_clover_analysis
Semantic segmentation code for analyzing images of kura clover plants in K1702.

## Dataset
### Description
The goal of this project is to detect and segment field-grown Kura clover (Trifolium ambiguum L.) plants, and calculate some standardized metrics and shape descriptions and train an object detection and semantic segmentation models for images of individual kura clover plants taken within a pvc sampling quadrat.

***

### Background and Workflow
Our lab grew out 1135 unique accesssions of kura clover for our clover breeding program. We want to train a model to effectively mask the kura clover against the soil and evaluated shape and density parameters of the plant against the defined ROI (PVC quadrat). The general workflow is as follows:
* Annotate all of the images in Labelbox. Classes include `quadrat_corner`, `kura_point`, `quadrat_point`, and `soil_point`.
* Using the point prompts, generate masks for the entire image using the Segment-Anything ([SAM](https://segment-anything.com/)) from Meta AI as ground truth segmentation masks.
* Develop two different models: EfficientDet object detector to locate the corners of the PVC quadrat, and a DeepLabV3 semantic segmentation model to mask the clover plants, soil and quadrat. 
* Using the 4 detected corner points, we will transform the masked image to the correct relative dimensions of the PVC quadrat to remove skew distortions introduced by semi-oblique imaging.
* Measure standard shape descriptors of the kura mask, as well as perform connected components analysis
* Report family-wise density estimates and compute breeding values for each of the accessions, producing a ranking for each accession based on how compact or sparse it is.

***

### Image Examples

We acquired images of individual kura clover accessions grown in the field. Each plant was demarcated using a standard sampling quadrat constructed of 3/4" Schedule 40 PVC pipe. Quadrat dimensions were as follows:
* OD (HxW) = 18"x18"
* ID (HxW) = 16.25"

Here is an example of one of the plants with a small but very dense canopy
![Accession Ta00070: A small, but dense plant](assets/Ta00070.jpg)

In contrast, here is an example of a plant with several small, dense clustered canopies results from rhizomatous growth below the soil.
![Accession Ta00079: A plant exhibiting rhizomatous growth](assets/Ta00079.jpg)

Finally, this accession exhibits uniformly sparse canopy, with most of the leaves growing at the margins of the plant.
![Accession Ta00696: A plant exhibiting sparse growth](assets/Ta00696.jpg)

### Image Mask Generation

### Quadrat Corner Detection

### Semantic Segmentation Model





















# K1702 Clover Analysis :four_leaf_clover:
Semantic segmentation and object detection pipeline for analyzing images of field-grown Kura clover (Trifolium ambiguum L.) plants.

## Dataset Overview :open_file_folder:
The Perennial Legumes Program at The Land Institute cultivated XXXX unique accessions from the USDA National Plant Germplasm System (NPGS) in the summer of 2017. Each plant was cultivated as a single plant plot. Plots were imaged with a Canon DLSR camera at least one time during the season. Clover plant were framed by a standard sampling quadrat constructed of 3/4" Schedule 40 PVC pipe with the following dimensions:
* OD (HxW) = 18"x18"
* ID (HxW) = 16.25"x16.25"

The annotated classes for object detection bounding boxes are:
```
{
  "clover": 1,
  "quadrat": 2,
  "quadrat_corner": 3
}
```
For the semantic segmentation masks we have the following classes:
```
{
  "soil": 0,
  "quadrat": 1,
  "clover": 3
}
```
The full dataset including the images, annotations, metadata, sampling strategy and training/validation splits are published in Zenodo and can be found at:

*Meyering et al* (2025). **K1702 - Kura Clover (Trifolium ambiguum) USDA Accession Image Dataset** [![DOI:10.5281/zenodo.14051741](https://zenodo.org/badge/DOI/10.5281/zenodo.14051741.svg)](https://doi.org/10.5281/zenodo.14051741)

## Project Workflow :repeat:
This project focuses on segmenting and analyzing Kura clover plants grown in the field. The goal is to develop computer vision models that can:
1. Detect and segment clover plants, and sampling quadrats from the soil background.
2. Detect and localize the corner PVC elbows on the sampling quadrat.
3. Compute the homology matrix between the image and the ROI bounded by the corners of the PVC quadrats.
4. Extract the kura clover contours from the standardized, prediction masks.
5. Calculate accession specific shape metrics such as solidity, canopy density, number of connected components, and fractal dimensions.
6. Rank accessions based on desired phenotypic traits and map to geographic origin.
7. Breeding values are computed for each accession based on shape/density metrics, enabling selection based on compactness, vigor, or spread.

## Model Development :computer:
We will train two models for this pipeline:
* Object Detection (Quadrat Corner Detection): We use an EfficientDet-based object detector to locate the four PVC quadrat corners using the Ross Wightman ```effdet``` library. [https://github.com/rwightman/efficientdet-pytorch](https://github.com/rwightman/efficientdet-pytorch)
* Semantic Segmentation (Plant/Soil/Quadrat): We will experiment with several different model architectures to segment kura clover from soil and quadrat background using the ```segmentation-models-pytorch``` [https://github.com/qubvel-org/segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorchlibrary).

Geometric Correction
Detected quadrat corners are used to perform a perspective transform, warping the image to a standardized top-down view to correct for skew and perspective distortion.

Feature Extraction
From the warped segmentation masks, we extract standardized shape and density features using:

Shape descriptors (e.g., area, solidity, convexity),

Connected components analysis,

Family-wise density estimates.

Breeding Value Computation
Breeding values are computed for each accession based on shape/density metrics, enabling selection based on compactness, vigor, or spread.

# Example Images 🖼️ 
Dense, Small Canopy
Accession Ta00070


Rhizomatous Growth
Accession Ta00079


Sparse, Margin Growth
Accession Ta00696




# Feature Outputs 📊 
From the warped masks, we compute:

Total clover area

Convex hull area

Solidity (area / convex hull area)

Number of clumps / components

Average component size

Edge distance metrics

These metrics are used to evaluate morphological traits relevant to clover breeding (e.g., canopy density, spread, compactness).

# Downstream Applications 📈
Ranking accessions by morphological traits

Identifying breeding targets for canopy architecture

Estimating family-wise genetic effects (e.g., BLUPs)

Automating phenotyping pipelines for breeding trials

# Directory Structure 📁
```
data/
├── processed/
    ├── images/
    └── targets/
├── raw/
    ├── images/
    └── mat_files/
├── bboxes.csv
├── data_split.csv
├── plant_status.csv
└── SAM_points.csv
logs/
metadata/
├── obj_det_class_map.json
└── segmentation_class_map.json
outputs/
├── object_detection/
└──segmentation /
scripts/
src/
├── utils/
    ├── collate_functions.py
    └── loggers.py
├── callbacks.py
├── datasets.py
├── eval.py
├── metrics.py
├── models.py
├── trainer.py
└── transforms.py
.gitignore
LICENSE
README.md
requirements.txt
train_object_detection_model.py
train_segmentation_model.py
```

# Tools & Technologies 🛠️ 
* [Labelbox](https://labelbox.com/): Image annotation
* [PyTorch](https://pytorch.org/): Model training
* [Albumentations](https://albumentations.ai/): Image augmentation
* [OpenCV](https://opencv.org/): Image warping and homography
* [NumPy](https://numpy.org/)
* [SciPy](https://scipy.org/): Connected components and shape analysis
* [Segment Anything (SAM)](https://segment-anything.com/): Prompt-based segmentation

# References 🔗 
* *Kirillov et al.* (2023). **Segment Anything** [DOI:10.48550/arXiv.2304.02643](https://doi.org/10.48550/arXiv.2304.02643)
* *Tan et al.* (2019). **EfficientDet: Scalable and Efficient Object Detection** [DOI:10.48550/arXiv.1911.09070](https://doi.org/10.48550/arXiv.1911.09070)
* *Chen et al.* (2017). **Rethinking Atrous Convolution for Semantic Image Segmentation** [DOI:10.48550/arXiv.1706.05587](https://doi.org/10.48550/arXiv.1706.05587)
* *Chen et al.* (2018). **Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation** [DOI:10.48550/arXiv.1802.02611](https://doi.org/10.48550/arXiv.1802.02611)
* *Xie et al.* (2021). **SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers** [DOI:10.48550/arXiv.2105.15203](https://doi.org/10.48550/arXiv.2105.15203)

# Acknowledgments 🤝
This work is part of our Kura clover breeding efforts at [The Land Institute](landinstitute.org). Many thanks to the field and lab teams who assisted in image collection and annotation.



