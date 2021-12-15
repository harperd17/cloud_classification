# Cloud Classification
## Classifying cloud type using satellite imagery

## Overview
With increasing concerns about climate change, researchers are trying to build more robust climate models. Clouds play a large role in Earth's climate, making them a 
crucial part of climate models. However, cloud classification is a tedious task. ****

## Goals
This project aims to use satellite images with masks for different cloud type classifications in order to build an image segmentation model for classifying clouds. This project 
came from [Kaggle](https://www.kaggle.com/c/understanding_cloud_organization/overview) so please visit the link for more information if you are interested.

## Data
The data used in this project is satellite imagery from NASA Worldview. The images have three bands (R, G, B) and are each accompanied with pixel encodings for four classifications (fish, flower, gravel, and sugar clouds). The pixel encodings were determined as the union of the labels produced by three different scientists per image. In total, a team of 68 scientists at the Max-Planck-Institute for Meterology in Hamburg, Germany created the labels for this data. More information can be found [here](https://www.kaggle.com/c/understanding_cloud_organization/data). 
In total, there are 5,546 images with pixel encodings for the cloud classification masks. There are an additional 3,703 test images that have to pixel encoding information.

## Repository Contents
---
<pre>
EDA             : <a href=https://github.com/harperd17/cloud_classification/tree/main/EDA/EDA.ipynb>EDA</a>

Modeling        : <a href=https://github.com/harperd17/cloud_classification/tree/main/modeling/CNN_segmentation_model.ipynb>Basic CNN Modeling</a>
                : <a href=https://github.com/harperd17/cloud_classification/tree/main/modeling/transposed_CNN_segmentation_model.ipynb>CNN Modeling With Transpose Layers</a>
                : <a href=https://github.com/harperd17/cloud_classification/tree/main/modeling/UNet_segmentation_model.ipynb>UNet Modeling</a>
                : <a href=https://github.com/harperd17/cloud_classification/tree/main/modeling/single_class_segmentation_model.ipynb>Modeling One Class</a>
                : <a href=https://github.com/harperd17/cloud_classification/tree/main/modeling/UNet_Resnet_segmentation_model.ipynb>UNet ResNet Modeling</a>
                
Report          : <a href=https://github.com/harperd17/cloud_classification/blob/main/report/Report.md>Report Notebook</a>
</pre>

## Project Information
---
<b>Author: </b>David Harper <br>
<b>Language: </b>Python <br>
<b>Tools/IDE: </b>********Jupyter Notebook and Spyder via Anaconda <br>
<b>Libraries: </b>*********numpy, pandas, plotly, matplotlib, math, sklearn, seaborn, mpu, geopy, sentinelhub, zipfile, os, datetime, requests, urllib, urllib3, scipy
