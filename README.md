# digit_recognizer

<p align="center">
    <img src="https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter" alt="jupyterLogo">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="pythonLogo">
</p>

## Description

AI programmed in python to recognize digits.  
The goal is to reach the max score using only Machine Learning.

I try to use a wide variety of models. I get the best score using SVC model with `98.975%` test accuracy which is a
pretty good score for a Machine Learning model.
I also tried KNN, RandomForest, but I didn't reach a better score than SVC :

- KNN's Score = 0.971
- RandomForest's Score = 0.968

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

On the SVC model I tried to train it using data augmentation, but it didn't increase the results a lot, like 0.1% more
accuracy.

**Note**
> A Deep Learning CNN model could have reach that max score easily, but I wanted to test what might reach the common
> Machine Learning classifier.

<!--
I try to use a wide variety of models. I get the best score using SVC model from sklearn. I also tried KNN,
RandomForest, SGD, DecisionTreeClassifier. I also tried ensemble learning with VotingClassifier and a Stacked Model which I get a score close to
the SVC model.
-->

## Kaggle competition

The app was made for the Kaggle Competition, you can find the link of my Notebook below:
<PLACEHOLDER>

I got a `98.17%` with only Machine Learning models. It is the limit we can have in Machine Learning models with the
given MNIST dataset of 42000 images (the real MNIST dataset as around 60,000 images for training).

## Quick start

The project is composed of one file named `main_kaggle.ipynb` at the root of the project.

It is a Jupyter Notebook file, the outputs are still available in the file, but you can start it to re-train the model.

Before, you need to install the Python Packages, you can find them all in the `requirements.txt` file. To install them
all directly type the following command in your terminal:

```bash
pip install -r requirements.txt
```

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

Sklearn documentation:  
<https://scikit-learn.org/>

Models for MNIST best score by Chris Deotte:  
<https://www.kaggle.com/c/digit-recognizer/discussion/61480#latest-550096>

## Contributors

Quentin MOREL :

- @Im-Rises
- <https://github.com/Im-Rises>

[![GitHub contributors](https://contrib.rocks/image?repo=Im-Rises/page_rank)](https://github.com/Im-Rises/page_rank/graphs/contributors)
