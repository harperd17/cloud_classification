# Cloud Classification
## Classifying cloud type using satellite imagery

## Overview
With increasing concerns about climate change, researchers are trying to build more robust climate models. Clouds play a large role in Earth's climate, making them a crucial part of climate models. However, cloud classification is a tedious task, so automating this process would lead to great strides in existing climate models and more accurate weather prediction. In particular, the following cloud structures are of interest to scientists; Sugar, Gravel, Fish, and Flower clouds.

## Goals
This project aims to use satellite images with masks for different cloud type classifications in order to build an image segmentation model for classifying clouds. This project came from [Kaggle](https://www.kaggle.com/c/understanding_cloud_organization/overview) so please visit the link for more information if you are interested.

## Data
The data used in this project is satellite imagery from NASA Worldview. The images have three bands (R, G, B) and are each accompanied with pixel encodings for four classifications (fish, flower, gravel, and sugar clouds). The pixel encodings were determined as the union of the labels produced by three different scientists per image. In total, a team of 68 scientists at the Max-Planck-Institute for Meterology in Hamburg, Germany created the labels for this data. More information can be found [here](https://www.kaggle.com/c/understanding_cloud_organization/data). In total, there are 5,546 images with pixel encodings for the cloud classification masks. There are an additional 3,703 test images that have to pixel encoding information. Each image has shape 1,400 X 2,100.

The data is stored in a shared drive. If you have permissions for this drive, you can access it through this [link](https://drive.google.com/drive/folders/1L11seELddhbPjdnh_7NUZ3rt1OsJAfhs?usp=sharing). With access to this drive, you can run the notebooks in this repo through google colab. If you don't have access, the data can be downloaded from [here](https://www.kaggle.com/c/understanding_cloud_organization/data).

## Repository Contents
---
<pre>
EDA             : <a href=https://github.com/harperd17/cloud_classification/tree/main/EDA>EDA</a>

Final Modeling  : <a href=https://github.com/harperd17/cloud_classification/tree/main/modeling/final_model.ipynb>Final Model</a>

Intermediate    : <a href=https://github.com/harperd17/cloud_classification/tree/main/modeling/CNN_segmentation_model.ipynb>Basic CNN Modeling</a>
Modeling        : <a href=https://github.com/harperd17/cloud_classification/tree/main/modeling/transposed_CNN_segmentation_model.ipynb>CNN Modeling With Transpose Layers</a>
                : <a href=https://github.com/harperd17/cloud_classification/tree/main/modeling/UNet_segmentation_model.ipynb>UNet Modeling</a>
                : <a href=https://github.com/harperd17/cloud_classification/tree/main/modeling/pretrained_models.ipynb>Pretrained Modeling</a>
                
Report          : <a href=https://github.com/harperd17/cloud_classification/blob/main/report/Report.md>Report Notebook</a>
</pre>

## Project Information
---
<b>Author: </b>David Harper <br>
<b>Language: </b>Python <br>
