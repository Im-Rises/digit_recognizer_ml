# digit_recognizer

<p align="center">
        <img src="https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter" alt="jupyterLogo">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="pythonLogo">
</p>

## Description

AI programmed in python to recognize digits.  
The goal is to reach the max score using only Machine Learning.

**Note**
> A Deep Learning CNN model could have reach that max score easily, but I wanted to test what might reach the common
> Machine Learning classifier.

I try to use a wide variety of models. I got the best scores with Stacked model and Voting Classifier composed of SVM,
RandomForestClassifier and KnnClassifier.

## Kaggle competition

The app was made for the Kaggle Competition, you can find the link of my Notebook below:
<PLACEHOLDER>

I got a `%` with only Machine Learning models. It is the limit we can obtain in Machine Learning models with the given
MNIST dataset of 42000 images (the real MNIST dataset as around 60,000 images for training).

## Quick start

The project is composed of one file named `main_kaggle.ipynb` at the root of the project.

It is a Jupyter Notebook file, the outputs are still available in the file, but you can start it to re-train the model.

Before, you need to install the Python Packages, you can find them all in the `requirements.txt` file. To install them
all directly type the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Images



| Confusion Matrix | ROC curve |
|------------------|-----------|
| <>               | <>        |

## MNIST dataset

MNSIT Kaggle dataset :
<https://www.kaggle.com/c/digit-recognizer/>

## Documentations

Wikipedia MNSIT:  
<https://en.wikipedia.org/wiki/MNIST_database>

Tutorial from Benoît Cayla:  
<https://www.datacorner.fr/mnsit-1/>

Sklearn documentation:  
<https://scikit-learn.org/>

## Contributors

Quentin MOREL :

- @Im-Rises
- <https://github.com/Im-Rises>

[![GitHub contributors](https://contrib.rocks/image?repo=Im-Rises/page_rank)](https://github.com/Im-Rises/page_rank/graphs/contributors)
