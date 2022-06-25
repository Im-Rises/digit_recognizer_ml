# digit_recognizer

<p align="center">
    <img src="https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter" alt="jupyterLogo">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="pythonLogo">
</p>

## Description

AI programmed in python to recognize digits.  
The goal is to reach the max score using only Machine Learning.

I try to use a wide variety of models. I get the best score using SVC model with `98.985%` test accuracy which is a
pretty good score for a Machine Learning model.
I also tried KNN, RandomForest, but I didn't reach a better score than SVC :

- KNN's Score = 97.882%
- RandomForest's Score = 98.064%

For each model training I use scaling and data augmentation. For data augmentation I created functions ti shift, rotate
and zoom the images.
I ended up not using the zoom because it wasn't increasing the performance of the model.

The MNIST dataset I used is a sliced one for the Kaggle competition, you can find the information in the section
below `Kaggle competition`.

**Note**
> A Deep Learning CNN model could have reach that max score easily, but I wanted to test what might reach the common
> Machine Learning classifier.

## Kaggle competition

The app was made for the Kaggle Competition, you can find the link of my Notebook below:  
<https://www.kaggle.com/imrises/mnist-ml-svc-99>

I got a `98.975%` which is superb score for a Machine Learning model on MNIST dataset. I only used the provided
part of the MNIST dataset, this MNIST dataset is composed of 42 000 images (the real MNIST dataset as around 60 000
images for training).

## Quick start

The project is composed of three main files I found the best for MNIST classification. All at the root of the project.

- main_svc.ipynb
- main_knn.ipynb
- main_random_forest.ipynb

They are Jupyter Notebook files, the outputs are still available in the file, but you can start it to re-train the
models.

Before, you need to install the Python Packages, you can find them all in the `requirements.txt` file. To install them
all directly type the following command in your terminal:

```bash
pip install -r requirements.txt
```

You also need to have an IDE or use the Jupyter Notebook server directly.  
<https://jupyter.org>

## MNIST images

![mnist_images](https://user-images.githubusercontent.com/59691442/175500317-960a195c-6b82-4538-bb8a-ebad84504e76.png)

<!--
| MNIST | MNIST |
|------------------|-----------|
|![mnist_images](https://user-images.githubusercontent.com/59691442/175499175-62fb55f9-1fb6-4615-840f-3701c1aa2cdf.png)|![mnsit_images](https://user-images.githubusercontent.com/59691442/175499704-5920ab92-633a-41a6-9f34-8f67b9cbd57b.png)|
-->

## Study images

| Confusion Matrix | ROC curve|
|---|---|
| ![confusion_matrix](https://user-images.githubusercontent.com/59691442/175617912-72551a00-7f05-4967-adfc-a96d9924a40e.png) | ![roc_curve](https://user-images.githubusercontent.com/59691442/175617938-ff23dfb9-aa45-4de5-8d79-9c9b54d1cde2.png) |

## MNIST dataset

MNIST Kaggle dataset :  
<https://www.kaggle.com/c/digit-recognizer/>

## Documentations

Wikipedia MNIST:  
<https://en.wikipedia.org/wiki/MNIST_database>

Tutorial from Benoit Cayla:  
<https://www.datacorner.fr/mnsit-1/>

Models for MNIST best score by Chris Deotte:  
<https://www.kaggle.com/c/digit-recognizer/discussion/61480#latest-550096>

## Libraries and languages

Python:  
<https://www.python.org>

Jupyter Notebook:  
<https://jupyter.org>

Scikit-Learn:  
<https://scikit-learn.org/>

## Contributors

Quentin MOREL :

- @Im-Rises
- <https://github.com/Im-Rises>

[![GitHub contributors](https://contrib.rocks/image?repo=Im-Rises/page_rank)](https://github.com/Im-Rises/page_rank/graphs/contributors)

<!--
I try to use a wide variety of models. I get the best score using SVC model from sklearn. I also tried KNN,
RandomForest, SGD, DecisionTreeClassifier. I also tried ensemble learning with VotingClassifier and a Stacked Model which I get a score close to
the SVC model.
-->

<!--
Classifier:
- KNeighborsClassifier
- svm.svc
- RandomForestClassifier
- DecisionTreeClassifier
- SGDClassifier

Ensemble Learning:
- VotingClassifier
- StackingClassifier
-->
