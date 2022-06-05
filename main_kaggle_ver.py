import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

import common_functions as cf

if __name__ == "__main__":
    # Images are 28x28x1

    df_train = pd.read_csv("Datasets/train.csv")
    df_test = pd.read_csv("Datasets/test.csv")

    y_train = df_train["label"]
    X_train = df_train.drop("label", axis=1)

    y_test = []  # Empty no labels for test data
    X_test = df_test

    cf.plot_images(X_train)

    # # Display digit at index 0
    # plt.imshow(X_train.iloc[3, 0:].values.reshape(28, 28), cmap=plt.cm.gray_r)
    # plt.axis("off")
    # plt.show()


