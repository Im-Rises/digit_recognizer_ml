from matplotlib import pyplot as plt


def plot_images(dataset):
    plt.figure(figsize=(9, 9))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(dataset.iloc[i, 0:].values.reshape(28, 28))
        plt.axis("off")
    plt.show()
