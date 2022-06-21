import pandas as pd
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import common_functions as cf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if __name__ == "__main__":
    # Images are 8x8x1

    # Load the digits dataset: digits
    digits = datasets.load_digits()

    # Print the keys and DESCR of the dataset
    print(digits.keys())
    print(digits.DESCR)

    # Print the shape of the images and data keys
    print(digits.images.shape)
    print(digits.data.shape)

    # Display digit at index 0
    plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()

    # Create feature and target arrays
    X = digits.data
    y = digits.target

    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    # knn = KNeighborsClassifier(7)
    #
    # # Fit the classifier to the training data
    # knn.fit(X_train, y_train)
    #
    # # Print the score
    # print(knn.score(X_test, y_test))

    # Setup arrays to store train and test accuracies
    neighbors = np.arange(1, 10)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    # Loop over different values of k
    for i, k in enumerate(neighbors):
        # Setup a k-NN Classifier with k neighbors: knn
        knn = KNeighborsClassifier(n_neighbors=k)

        # Fit the classifier to the training data
        knn.fit(X_train, y_train)

        # Compute accuracy on the training set
        train_accuracy[i] = knn.score(X_train, y_train)

        # Compute accuracy on the testing set
        test_accuracy[i] = knn.score(X_test, y_test)

    # Generate plot
    plt.title('k-NN: Varying Number of Neighbors')
    plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
    plt.plot(neighbors, train_accuracy, label='Training Accuracy')
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.show()

    # It seems that the best accuracy would be with one neighbor as the Testing and training accuracy are the higher

    # Predict with k=1
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    print(knn.score(X_test, y_test))
    y_pred = knn.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True)
    plt.show()

    # Now that we got a good score on this set of data, we can try to switch to the real MNIST
    # which is composed of 28x28 images.
