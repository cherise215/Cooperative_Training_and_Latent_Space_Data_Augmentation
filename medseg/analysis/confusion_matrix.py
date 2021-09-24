# Created by cc215 at 25/11/19
# draw confusion matrix given predictions and targets (support multi-class)
# Enter scenario name here
# Enter steps here

import numpy as np
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

def plot_confusion_matrix( y_gt, y_pred, labels=None, normalize=None,
                           display_labels=None,  include_values=True,cmap='viridis',
                           xticks_rotation='horizontal', values_format=None, ax=None):
    '''
    adapted from scikit-learn/metrics/_classification.py
    :param y_gt:   array-like of shape (n_samples,)   Ground truth (correct) target values.
    :param y_pred: array-like of shape (n_samples,)   Estimated targets as returned by a classifier.
    :param labels : array-like of shape (n_classes), default=None
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If ``None`` is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.
    :param normalize: {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.
    :param display_labels : array-like of shape (n_classes,), default=None
        Target names used for plotting. By default, `labels` will be used if
        it is defined, otherwise the unique labels of `y_true` and `y_pred`
        will be used.
    :param  cmap : str or matplotlib Colormap, default='viridis'
            Colormap recognized by matplotlib.
    :param include_values: Bool, default= True, Includes values in confusion matrix.
    :param xticks_rotation : {'vertical', 'horizontal'} or float, \
                         default='vertical'
            Rotation of xtick labels.
     :param values_format : specify the format of values displayed in the matrix
        Rotation of xtick labels.
    :param ax_ : matplotlib Axes
        Axes with confusion matrix.

    :return:
     C : ndarray of shape (n_classes, n_classes)
        Confusion matrix.

    '''
    cm =  confusion_matrix(y_true=y_gt,y_pred=y_pred,sample_weight=None,labels=labels,normalize=normalize)
    if labels is None and display_labels is None:
        n_classes = cm.shape[0]
        display_labels= np.arange(0,n_classes) ## use digits for displaying
    else:
        if display_labels is None:
           display_labels=labels

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=display_labels)


    return disp.plot(include_values=include_values,cmap=cmap, ax=ax, xticks_rotation=xticks_rotation,values_format=values_format)
