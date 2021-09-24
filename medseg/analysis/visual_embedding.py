# Created by cc215 at 09/12/19
# Enter feature description here
# Enter scenario name here
# Enter steps here


import umap
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(context="paper", style="white")


def plot_umap(x, y, num_classes=10, display_labels=None, title="data embedded into two dimensions by UMAP", save_path=None):
    '''
    visualize the data embedding from a single dataset with UMAP
    :param x: input features or images, 2D array [N by D]
    :param y: target labels for display: 1D array [N]
    :param num_classes: how many classes in total, e.g. MNIST: has 10 classes
    :param display_labels: : the labels shown in its colorbar for reference
    :return:
    '''

    reducer = umap.UMAP(random_state=42)
    print('load data:', x.shape)
    embedding = reducer.fit_transform(x)

    fig, ax = plt.subplots(figsize=(12, 10))
    color = y.astype(int)
    plt.scatter(
        embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral"
    )
    plt.setp(ax, xticks=[], yticks=[])
    cbar = plt.colorbar(boundaries=np.arange(num_classes + 1) - 0.5)
    cbar.set_ticks(np.arange(num_classes))
    classes = [str(i) for i in range(num_classes)]
    # set colorbar font size
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=14)
    if display_labels is None:
        cbar.set_ticklabels(classes)
    else:
        cbar.set_ticklabels(display_labels)
    plt.title(title, fontsize=18)
    if not save_path is None:
        plt.savefig(save_path)
    plt.show()


def plot_multi_dataset_umap(dataArraysList, dataset_labels=None, display_labels=None):
    '''
    plot umap for multiple datasets
    :param dataArraysList: a list of 2D arrays, each element contains all images from one dataset
    :param dataset_labels: the labels for identifying different dataset
    :param display_labels: the labels for displaying on the plots.
    :return: concatenated image features 2D array dim=[all_samples in the datasets]*features, labels: 1D array: dim=[all_samples in the datasets],
    '''
    if dataset_labels == None:
        labels = list(np.arange(len(dataArraysList)))
    else:
        labels = dataset_labels
    total_dataset_features = []
    total_dataset_labels = []
    for dataset_id, dataset in enumerate(dataArraysList):
        print('load {} [dataset_size*num_features] from dataset {}'.format(dataset.shape, dataset_id))
        dataset_marker = labels[dataset_id]
        dataset_label_array = np.ones((len(dataset))) * dataset_marker
        total_dataset_features.append(dataset)
        total_dataset_labels.append(dataset_label_array)

    if len(dataArraysList) >= 2:
        input = np.vstack(total_dataset_features)
        label = np.concatenate(total_dataset_labels, axis=0)
    else:
        input = total_dataset_features[0]
        label = total_dataset_labels[0]

    assert len(input.shape) == 2, 'input features must be 2D array,  but got {}'.format(input.shape)
    assert len(label.shape) == 1, 'labels must be 1D array, but got {}'.format(label.shape)
    print('X:', input.shape)
    print('Y:', label.shape)

    plot_umap(x=input, y=label, num_classes=len(dataArraysList), display_labels=display_labels)
    return total_dataset_features, total_dataset_labels


if __name__ == '__main__':
    import numpy as np
    y = np.arange(0, 1000)
    y[:50] = 0
    y[50:] = 1
    X = np.random.random((1000, 100))
    plot_umap(X, y, num_classes=2, save_path='./output/umap.png')
