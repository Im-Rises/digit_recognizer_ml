import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

import common_functions as cf

if __name__ == "__main__":
    # Images are 28x28x1

    # Load data
    df_train = pd.read_csv("Datasets/train.csv")
    df_test = pd.read_csv("Datasets/test.csv")

    y_train = df_train["label"]
    X_train = df_train.drop("label", axis=1)

    y_test = []  # Empty no labels for test data
    X_test = df_test

    # Plot images
    cf.plot_images(X_train)

    # Display digit at index 3
    plt.imshow(X_train.iloc[3, 0:].values.reshape(28, 28), cmap=plt.cm.gray_r)
    plt.axis("off")
    plt.show()
    # print(X_train.values[0])
    # print(y_train.values[0])

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Setup arrays to store train and test accuracies
    neighbors = np.arange(1, 10)
    train_accuracy = np.empty(len(neighbors))
    # test_accuracy = np.empty(len(neighbors))

    # Loop over different values of k
    for i, k in enumerate(neighbors):
        # Setup a k-NN Classifier with k neighbors: knn
        knn = KNeighborsClassifier(n_neighbors=k)

        # Fit the classifier to the training data
        knn.fit(X_train, y_train)

        # Compute accuracy on the training set
        train_accuracy[i] = knn.score(X_train, y_train)

        # # Compute accuracy on the testing set
        # test_accuracy[i] = knn.score(X_test, y_test)

    # Generate plot
    plt.title('k-NN: Varying Number of Neighbors')
    # plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
    plt.plot(neighbors, train_accuracy, label='Training Accuracy')
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.show()
