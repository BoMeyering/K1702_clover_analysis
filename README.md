# K1702_clover_analysis
Semantic segmentation code for analyzing images of kura clover plants in K1702.

### Description
Code developed to train an object detection and semantic segmentation models for images of individual kura clover plants taken within a pvc sampling quadrat.

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

Here is an example of one of the plants with a small but very dense canopy
![Accession Ta00070: A small, but dense plant](assets/Ta00070.jpg)

In contrast, here is an example of a plant with several small, dense clustered canopies results from rhizomatous growth below the soil.
![Accession Ta00079: A plant exhibiting rhizomatous growth](assets/Ta00079.jpg)

Finally, this accession exhibits uniformly sparse canopy, with most of the leaves growing at the margins of the plant.
![Accession Ta00696: A plant exhibiting sparse growth](assets/Ta00696.jpg)

### Image Mask Generation

### Quadrat Corner Detection

### Semantic Segmentation Model

