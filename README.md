# Melonoma Skin Cancer Detection
 
<p align="center">

  [![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

  [![GitHub contributors](https://img.shields.io/github/contributors/rstak/Melonoma-Skin-Cancer-Detection)](https://github.com/RsTaK/Melonoma-Skin-Cancer-Detection/graphs/contributors/)
  [![GitHub license](https://img.shields.io/github/license/rstak/Melonoma-Skin-Cancer-Detection)](https://github.com/RsTaK/Melonoma-Skin-Cancer-Detection/blob/master/LICENSE)
</p>  

<img src="assets\header.png"/>

# About
Skin cancer is the most prevalent type of cancer. Melanoma, specifically, is responsible for 75% of skin cancer deaths, despite being the least common skin cancer. The American Cancer Society estimates over 100,000 new melanoma cases will be diagnosed in 2020. It's also expected that almost 7,000 people will die from the disease. As with other cancers, early and accurate detection potentially aided by data science can make treatment more effective.

Leveraging the power of Deeplearning and Datascience, a solution is given to identify melanoma in images of skin lesions.

Dataset provided by [Society for Imaging Informatics in Medicine (SIIM)](https://siim.org). 

# Summary of Approach

After experimenting with different approaches, We used a Three Layered Approach having diverse models with different architectures and augmentations. All models are initially pre-trained on imagenet and fine-tuned on the dataset. 

Achived Area under the ROC Curve (AUC) Score of 0.93

<img src="assets\architecture.png"/>

# Data Augmentations

Used a powerful augmentation library [Albumentations](https://albumentations.readthedocs.io/en/latest/api/augmentations.html) to perform augmentations are included in the final approach :
* Resize
* RandomSizedCrop
* RandomRotate90
* HorizontalFlip
* VerticalFlip
* CoarseDroupout
* ToTensorV2

# Other Configuration
* Loss : Criterion Margin Focal Binary Cross Entropy
* Metrics : Area under the ROC Curve (AUC)
* Optimizer : AdamW 
* Test Train Augmentation (TTA)

# Biggest Challenge

* We had a problem of highly imbalance dataset i.e. 2% - 98% class distribution.
* Single Patient had multiple Images associated to its ID

# How We Approcahed

### 1. Imbalanced Dataset

* To overcome the the problem of uneven class distribution, we used external data. Dataset provdided by [Society for Imaging Informatics in Medicine (SIIM)](https://siim.org) in 2019, 2018 and 2017 were also included which resulted in 50K + image count with class distribution around 9% - 91% which is a good ratio in Medical problem.

* Trained on external data but only 2020 dataset was used for validation

* Along with including more data, We used downsampling to inorder to get even class distribution for training. 

* Used a varient of Focal Loss since focal loss is good with imbalanced dataset

### 2. Single Id with Multiple Images
* Used Triple Stratified 5 Fold
* * Stratify 1 - Isolate Patients
* * * A single patient can have multiple images. Now all images from one patient are fully contained inside a single TFRecord. This prevents leakage during cross validation
* * Stratify 2 - Balance Malignant Images
* * * The entire dataset has 9% malignant images. Each TFRecord contains 9% malignant images. This makes validation score more reliable.
* * Stratify 3 - Balance Patient Count Distribution
* * * Some patients have as many as 115 images and some patients have as few as 2 images. When isolating patients into TFRecords, each record has an equal number of patients with 115 images, with 100, with 70, with 50, with 20, with 10, with 5, with 2, etc. This makes validation more reliable.

All thanks goes to [Chris Deotte](https://www.kaggle.com/cdeotte) for this contribuiton.

# File Description
* src/augmentation.py : Contains the augmentation used
* src/config.py : Initialied basic configuration for training purpose
* src/dataset.py : CustomDataLoader built over nn.DataSet
* src/efficientnet.py : Function to load the required EfficientNet model
* src/loss.py : Defined Custom Loss (Criterion Margin Focal Binary Cross Entropy)
* src/meta_classifier.py : Used meta features of patient for classification
* src/prediction.py : Inference file used for Predicition and OOF calculation
* src/stacking.py : Script used for stacking
* src/train.py : Script that connects all the componenet to be used for training

# License 
This project is licensed under the MIT License - see the [LICENSE](https://github.com/RsTaK/Melonoma-Skin-Cancer-Detection/blob/master/LICENSE) file for details.