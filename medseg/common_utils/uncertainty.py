# Created by cc215 at 22/05/19
# this file contains common method to estimate uncertainty
# using MC dropout
import numpy as np


def cal_entropy_maps(pred_probs, eps=1e-7, threhold=0., use_max=False):
    '''

    calculate entropy maps from probs which is the output from neural networks after the softmax layer.
     eps is used to prevent np.log2(zero). Note that this function is working on one image only.
    :param 2D soft_max_prob_maps: C_classes*H*W
    :param threshold: float uncertainty below this value will be filtered out.
    :param use_max: if use max, then find the maximum prob over classes and use the value to cal entropy,
    other wise calculate entropy across each channel and then averaged them.
    :return: A 2D map  H*W with values >0
    '''

    assert len(pred_probs.shape) == 3, 'only support input of three dimension [Channel, H, W]'
    if use_max:
        ax_probs = np.amax(pred_probs, axis=0)  # using maxium prob to cal entropy
        entropy = (-ax_probs * np.log2(ax_probs + eps))
    else:
        entropy = (-pred_probs * np.log2(pred_probs + eps)).sum(axis=0)
    entropy = np.nan_to_num(entropy)
    entropy[entropy < threhold] = 0.
    return entropy


def cal_batch_entropy_maps(pred_probs, eps=1e-7, threhold=0., use_max=False):
    '''

    calculate entropy maps from probs which is the output from neural networks after the softmax layer.
     eps is used to prevent np.log2(zero). Note that this function is working on batches of image.
    :param 3D soft_max_prob_maps: N*C_classes*H*W
    :param threshold: float uncertainty below this value will be filtered out.
    :param use_max: if use max, then find the maximum prob over classes and use the value to cal entropy,
    other wise calculate entropy across each channel and then averaged them.
    :return: A 3D map [ N*H*W] with values >0
    '''

    assert len(pred_probs.shape) == 4, 'only support input of four dimension [N, Channel, H, W]'
    if use_max:
        ax_probs = np.amax(pred_probs, axis=1)  # using maxium prob to cal entropy
        entropy = (-ax_probs * np.log2(ax_probs + eps))
    else:
        entropy = (-pred_probs * np.log2(pred_probs + eps)).sum(axis=1)
    entropy = np.nan_to_num(entropy)
    entropy[entropy < threhold] = 0.

    if len(entropy.shape) < 3:
        print('check dimensionality of the output')
        raise ValueError
    return entropy
