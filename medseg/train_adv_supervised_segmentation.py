# Created by cc215 at 02/05/19
# This code is for training basic segmentation networks
# Scenario: learn to segment cardiac short-axis images
# Steps:
#  1. define the segmentation network and optimiser
#  2. fetch images tuples from the disk to train the segmentation
#       ## input randomly selected batch of images slices
#  3. calculate loss
#  4. optimize the network, back to step 2.
import argparse
from os.path import join, exists
import gc
import os
from xml.sax.handler import property_declaration_handler
import numpy as np
import random
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch
import torch.distributions.bernoulli as BNP
import torch.nn.functional as F
import torch.optim as optim
import socket

import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker, cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt

from common_utils.load_args import Params
from common_utils.save import save_imgs
from common_utils.basic_operations import rescale_intensity, check_dir
from models.custom_loss import EntropyLoss
from models.base_segmentation_model import SegmentationModel
from models.losspicker import SegmentationLossPicker
from models.EMA import EMA
from common_utils.metrics import print_metric
from models.model_util import makeVariable, _disable_tracking_bn_stats
from models.image_transformer.adv_affine import AdvAffine
from models.image_transformer.adv_noise import AdvNoise
from models.image_transformer.adv_morph import AdvMorph
from models.image_transformer.adv_bias import AdvBias
from models.image_transformer.adv_compose_transform import ComposeAdversarialTransform
from common_utils.basic_operations import construct_input

from dataset_loader.cardiac_ACDC_dataset import CardiacACDCDataset
from dataset_loader.base_segmentation_dataset import CombinedDataSet
from dataset_loader.transform import Transformations  #
from models.custom_loss import EntropyLoss


# def get_batch(dataiter, dataloader):
#     try:
#         batch = next(dataiter)
#     except StopIteration:
#         dataiter = dataloader.__iter__()
#         batch = next(dataiter)
#     return batch


def get_default_augmentor(data_size, divergence_types, divergence_weights, random_select=True, unlabelled_data=False, dataset_name='cardiac', debug=False, use_gpu=True, transformation_type='composite'):
    '''
    return a data augmentor and a list of flags indicating the component of the data augmentation
    e.g [1,1,1,1]->[bias,noise,morph,affine]
    '''
    if dataset_name == 'cardiac':
        # 192X192
        vector_size = [data_size[2] // 8, data_size[3] // 8]
        downscale = 2
        control_point_spacing = [data_size[2] // (downscale * 2), data_size[3] // (downscale * 2)]
        deform_strenth = 1.5
        bias_mag = 0.3

    elif dataset_name == 'prostate':
        vector_size = [data_size[2] // 8, data_size[3] // 8]
        deform_strenth = 1.5
        downscale = 2
        bias_mag = 0.3
        control_point_spacing = [data_size[2] // (downscale * 2), data_size[3] // (downscale * 2)]
    else:
        raise NotImplementedError
    transformation_list = []
    if transformation_type == 'composite':
        opt_flags = [1, 1, 1, 1]
    else:
        if transformation_type == 'bias':
            opt_flags = [1, 0, 0, 0]
        elif transformation_type == 'noise':
            opt_flags = [0, 1, 0, 0]
        elif transformation_type == 'morph':
            opt_flags = [0, 0, 1, 0]
        elif transformation_type == 'affine':
            opt_flags = [0, 0, 0, 1]
        elif transformation_type == 'no bias':
            opt_flags = [0, 1, 1, 1]
        elif transformation_type == 'no noise':
            opt_flags = [1, 0, 1, 1]
        elif transformation_type == 'no morph':
            opt_flags = [1, 1, 0, 1]
        elif transformation_type == 'no affine':
            opt_flags = [1, 1, 1, 0]
        else:
            raise NotImplementedError

    if random_select and transformation_type == 'composite':
        select_flags = torch.randint(low=0, high=2, size=(4,))
        select_flags = np.array(select_flags) * np.array(opt_flags)
        if sum(select_flags) == 0:
            # select all
            select_flags = [1] * 4
        # rand_flags[3]=1
    else:
        select_flags = [1, 1, 1, 1]
    opt_flags = np.array(select_flags) * np.array(opt_flags)

    assert sum(opt_flags) > 0, 'must have at least one type of transformation'

    if opt_flags[3] == 1:
        augmentor_affine = AdvAffine(config_dict={
            'rot': 15.0 / 180,
            'scale_x': 0.2,
            'scale_y': 0.2,
            'shift_x': 0.1,
            'shift_y': 0.1,
            'xi': 1e-6,
            'data_size': data_size,
            'forward_interp': 'bilinear',
            'backward_interp': 'bilinear'},
            debug=debug, use_gpu=use_gpu)
        transformation_list.append(augmentor_affine)

    if opt_flags[2] == 1:
        augmentor_morph = AdvMorph(
            config_dict={'epsilon': deform_strenth,
                         'xi': 1e-6,
                         'data_size': data_size,
                         'vector_size': vector_size,
                         'interpolator_mode': 'bilinear'
                         },
            debug=debug, use_gpu=use_gpu)
        transformation_list.append(augmentor_morph)

    if opt_flags[1] == 1:
        augmentor_noise = AdvNoise(
            config_dict={'epsilon': 1.0,
                         'xi': 1e-6,
                         'data_size': data_size},
            debug=debug, use_gpu=use_gpu)
        transformation_list.append(augmentor_noise)
    if opt_flags[0] == 1:
        augmentor_bias = AdvBias(
            config_dict={'epsilon': bias_mag,
                         'xi': 1e-6,
                         'control_point_spacing': control_point_spacing,
                         'downscale': downscale,
                         'data_size': data_size,
                         'interpolation_order': 3,
                         'init_mode': 'random',
                         'space': 'log'},
            debug=debug, use_gpu=use_gpu)
        transformation_list.append(augmentor_bias)

    if random_select and transformation_type == 'composite':
        # shuffle the order of the transformations
        r = random.random()            # randomly generating a real in [0,1)
        random.shuffle(transformation_list, lambda: r)  # lambda : r is an unary function which returns r
        # using the same function as used in prev line so that shuffling order is same
        random.shuffle(opt_flags, lambda: r)

    composed_augmentor = ComposeAdversarialTransform(chain_of_transforms=transformation_list, divergence_types=divergence_types, divergence_weights=divergence_weights, use_gpu=use_gpu,
                                                     debug=False)

    return composed_augmentor, opt_flags


def train_network(training_opt,
                  experiment_name: str,
                  dataset: list,
                  segmentor_opt: dict,
                  segmentor_resume_path: str,
                  experiment_opt: dict,
                  save_dir: str,
                  log: bool, dataset_name: str = 'cardiac', debug=False):
    '''

    :param experiment_name:
    :param dataset:

    :param resume_path:
    :param save_dir:
    :param log:
    :return:
    '''
    # output setting
    if log:
        machine_name = socket.gethostname().split('.')[0]
        log_dir = './runs/'
        check_dir(log_dir, create=True)
        writer = SummaryWriter(log_dir=log_dir + experiment_name + '.' +
                               machine_name, comment=experiment_name, purge_step=0)

    # ========================Define models==================================================#
    num_classes = segmentor_opt["num_classes"]
    use_gpu = segmentor_opt["use_gpu"]
    use_ema = False
    SSL_flag = experiment_opt["learning"]["semi"]
    print('SEMI:', SSL_flag)
    if 'EMA'in experiment_opt['learning'].keys():
        use_ema = experiment_opt['learning']['EMA']
        print('EMA:', use_ema)

    adv_consistency_learning = True
    if 'adv_consistency_learning'in experiment_opt['learning'].keys():
        adv_consistency_learning = experiment_opt['learning']['adv_consistency_learning']
    print('adv_consistency_learning:', adv_consistency_learning)

    # baseline methods:
    baseline_method = ""
    if 'baseline_method'in experiment_opt['learning'].keys():
        baseline_method = experiment_opt['learning']['baseline_method']
    print('baseline_method:', baseline_method)
    if baseline_method != "":
        assert adv_consistency_learning is False
        if baseline_method == "TCSM":
            use_ema = True

    optimizer = 'Adam'
    if 'optimizer'in experiment_opt['learning'].keys():
        optimizer = experiment_opt['learning']['optimizer']
        print('optimizer:', optimizer)
    segmentation_model = SegmentationModel(network_type=segmentor_opt["network_type"], num_classes=num_classes,
                                           encoder_dropout=segmentor_opt["encoder_dropout"],
                                           decoder_dropout=segmentor_opt["decoder_dropout"],
                                           use_gpu=use_gpu, lr=segmentor_opt["lr"],
                                           resume_path=segmentor_resume_path,
                                           use_ema=use_ema,
                                           optimizer=optimizer
                                           )

    # =========================dataset config==================================================#
    train_set = dataset[0]
    unlabelled_set = dataset[1]
    validate_set = dataset[2]
    test_set = dataset[3]
    original_bs = segmentor_opt["batch_size"]
    if experiment_opt['data']['keep_orig_image_label_pair_for_training']:
        sup_batch_size = original_bs / 2.0

    labelled_data_loader = DataLoader(dataset=train_set, num_workers=0,
                                      batch_size=int(sup_batch_size), shuffle=True, drop_last=False)
    labelled_data_iter = labelled_data_loader.__iter__()

    if unlabelled_set is not None:
        unlabelled_loader = DataLoader(dataset=unlabelled_set, num_workers=0,
                                       batch_size=int(sup_batch_size), shuffle=True, drop_last=False)
        unlablled_data_iter = unlabelled_loader.__iter__()

    validate_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=int(original_bs), shuffle=False,
                                 drop_last=False)
    # test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=int(original_bs), shuffle=False,
    #                              drop_last=False)
    best_score = -10000

    # adversarial setting:
    adversarial_setting = experiment_opt["adversarial training"]
    optimization_mode = adversarial_setting["optimization mode"]  # 'chain'
    power_iteration = adversarial_setting["power iteration"]  # true or a list of true/false
    if "use mean teacher" in adversarial_setting.keys():
        # if true, we use the prediction from EMA model of the student model
        use_mean_teacher = adversarial_setting["use mean teacher"]
    else:
        use_mean_teacher = False

    if "if norm image" in adversarial_setting.keys():
        # if true, we use the prediction from EMA model of the student model
        if_norm_image = adversarial_setting["if norm image"]
    else:
        if_norm_image = False
    print("if norm image", if_norm_image)
    if "disable adv noise" in adversarial_setting.keys():
        # if true, then will only apply random noise if noise augmentation is turned on.
        disable_adv_noise = adversarial_setting["disable adv noise"]
        print('disable adv noise', disable_adv_noise)
    else:
        disable_adv_noise = False

    n_iter = int(adversarial_setting["iteration step"])  # 1
    use_gt_for_consistency = False

    if 'use gt' in adversarial_setting.keys():
        use_gt_for_consistency = adversarial_setting["use gt"]
    else:
        use_gt_for_consistency = False
    print('use gt for supervised CR', str(use_gt_for_consistency))
    transformation_type = adversarial_setting["transformation type"]
    if 'supervised divergence types' in adversarial_setting.keys():
        supervised_divergence_types = adversarial_setting["supervised divergence types"]
        print('supervised divergence types', str(supervised_divergence_types))
    else:
        supervised_divergence_types = ['kl', 'contour']
    if 'supervised divergence weights' in adversarial_setting.keys():
        supervised_divergence_weights = adversarial_setting["supervised divergence weights"]
        print('supervised divergence weights', str(supervised_divergence_weights))

    else:
        supervised_divergence_weights = [1.0, 0.5]

    if 'unsupervised divergence types' in adversarial_setting.keys():
        unsupervised_divergence_types = adversarial_setting["unsupervised divergence types"]
        print('unsupervised divergence types', str(unsupervised_divergence_types))

    else:
        unsupervised_divergence_types = ['mse', 'contour']
    if 'unsupervised divergence weights' in adversarial_setting.keys():
        unsupervised_divergence_weights = adversarial_setting["unsupervised divergence weights"]
        print('unsupervised divergence weights', str(unsupervised_divergence_weights))
    else:
        unsupervised_divergence_weights = [1.0, 0.5]

    entropy_flag = False
    if 'entropy minimization' in experiment_opt['learning'].keys():
        entropy_flag = experiment_opt['learning']["entropy minimization"]
    print('entropy minimization', str(entropy_flag))

    if 'random combination' in adversarial_setting.keys():
        random_combination_flag = adversarial_setting["random combination"]
        print('random_combination_flag', str(random_combination_flag))
    else:
        random_combination_flag = True

    lambda_l = 1.0
    lambda_u = 1.0

    if 'adaptive_cr_weight' in adversarial_setting.keys():
        adaptive_cr_weight = adversarial_setting["adaptive_cr_weight"]
        print('adaptive_cr_weight', str(lambda_u))
    else:
        adaptive_cr_weight = False

   # =========================<<<<<start training>>>>>>>>=============================>
    i_iter = 0
    stop_flag = False
    cl_loss_weight = 0
    score_list = []
    n_epochs = segmentor_opt['n_epochs']
    for i_epoch in range(n_epochs):
        gc.collect()  # collect garbage
        g_count = 0
        total_loss = 0.
        regularization_loss = 0.
        supervised_consistency_loss_record = 0.
        unsupervised_consistency_loss_record = 0.
        std_loss_record = 0.
        unsupervised_entropy_loss_record = 0.
        supervised_entropy_loss_record = 0.
        if SSL_flag or adaptive_cr_weight is True:
            ratio = ((i_epoch + 1) / 200)
            ratio = 1 if ratio > 1 else ratio
            a_lambda_u = ratio * lambda_u
            a_lambda_l = ratio * lambda_l
        else:
            # keep it constant
            a_lambda_u = lambda_u
            a_lambda_l = lambda_l

        print('adjust supervised cr coefficient to:', a_lambda_l)
        print('adjust unsupervised cr coefficient to:', a_lambda_u)

        if stop_flag:
            break
        for b_iter in range(len(labelled_data_loader)):

            gc.collect()  # collect garbage
            # step 1: get a batch of labelled and unlabelled images
            segmentation_model.train()
            segmentation_model.model.zero_grad()

            a_loss_dict = {}

            labelled_batch = get_batch(dataiter=labelled_data_iter, dataloader=labelled_data_loader)

            image_l, label_l = labelled_batch['image'], labelled_batch['label']
            keep_origin = experiment_opt['data']['keep_orig_image_label_pair_for_training']
            if keep_origin:
                image_orig, gt_orig = labelled_batch['origin_image'], labelled_batch['origin_label']
                image_l = torch.cat([image_l, image_orig], dim=0)
                label_l = torch.cat([label_l, gt_orig], dim=0)

            image_l = makeVariable(image_l, type='float', use_gpu=use_gpu, requires_grad=False)
            label_l = makeVariable(label_l, type='long', use_gpu=use_gpu, requires_grad=False)
            # get classwise weights
            try:
                class_weights = experiment_opt['learning']['class_weights']
                if debug:
                    print('class weights', class_weights)
            except:
                print('fail to load weights')
                class_weights = None

            if SSL_flag:
                unlabelled_batch = get_batch(dataiter=unlablled_data_iter, dataloader=unlabelled_loader)
                image_u = unlabelled_batch['image']
                label_u = None
                image_u = makeVariable(image_u, type='float', use_gpu=use_gpu, requires_grad=False)
                un_labelled_keep_origin = False
                if 'unlabelled_keep_orig_image_label_pair_for_training' in experiment_opt['data'].keys():
                    un_labelled_keep_origin = experiment_opt['data']['unlabelled_keep_orig_image_label_pair_for_training']
                if un_labelled_keep_origin:
                    image_u_orig = unlabelled_batch['origin_image']
                    image_u_orig = makeVariable(image_u_orig, type='float', use_gpu=use_gpu, requires_grad=False)
                    image_u = torch.cat([image_u, image_u_orig], dim=0)
            else:
                image_u = None

             # optimize data augmentation
            if adv_consistency_learning:
                augmentor, opt_flags = get_default_augmentor(dataset_name=dataset_name,
                                                             random_select=random_combination_flag,
                                                             data_size=image_l.size(), transformation_type=transformation_type,
                                                             divergence_types=supervised_divergence_types,
                                                             divergence_weights=supervised_divergence_weights)

                augmentor.init_random_transformation()
                augmentor.is_gt = False
                augmentor.disable_adv_noise = disable_adv_noise

                if use_gt_for_consistency:
                    augmentor.is_gt = True
                    init_output_l = construct_input(label_l, num_classes=num_classes,
                                                    apply_softmax=False, is_labelmap=True, use_gpu=True)
                else:
                    if use_mean_teacher:
                        teacher_model = segmentation_model.get_teacher_model()
                        teacher_model.eval()
                        with torch.no_grad():
                            with _disable_tracking_bn_stats(teacher_model):
                                init_output_l = teacher_model(image_l)
                        model = segmentation_model.get_student_model()
                        segmentation_model.train()
                        print('USE MEAN TEACHER')
                    else:
                        init_output_l = None

                supervised_consistency_loss = a_lambda_l * augmentor.adversarial_training(
                    data=image_l, model=segmentation_model.model,
                    init_output=init_output_l,
                    lazy_load=False, power_iteration=power_iteration,
                    n_iter=n_iter,
                    optimization_mode=optimization_mode,
                    optimize_flags=[True] * len(augmentor.chain_of_transforms))

                supervised_consistency_loss_record += supervised_consistency_loss.item()
                a_loss_dict['CR_l'] = supervised_consistency_loss.item()

                if not SSL_flag:
                    regularization_loss = supervised_consistency_loss
                else:
                    du_augmentor, opt_flags = get_default_augmentor(dataset_name=dataset_name, random_select=random_combination_flag,
                                                                    data_size=image_u.size(), debug=debug, transformation_type=transformation_type,
                                                                    divergence_types=unsupervised_divergence_types, divergence_weights=unsupervised_divergence_weights)
                    du_augmentor.disable_adv_noise = disable_adv_noise
                    du_augmentor.if_norm_image = if_norm_image
                    du_augmentor.is_gt = False

                    du_augmentor.init_random_transformation()
                    if use_mean_teacher:
                        teacher_model = segmentation_model.get_teacher_model()
                        teacher_model.eval()
                        with torch.no_grad():
                            with _disable_tracking_bn_stats(teacher_model):
                                init_output_u = teacher_model(image_u)
                        model = segmentation_model.get_student_model()
                        segmentation_model.train()
                    else:
                        init_output_u = None
                    unsupervised_consistency_loss = a_lambda_u * du_augmentor.adversarial_training(data=image_u,
                                                                                                   model=segmentation_model.model,
                                                                                                   init_output=init_output_u,
                                                                                                   lazy_load=False,
                                                                                                   n_iter=n_iter,
                                                                                                   optimization_mode=optimization_mode,
                                                                                                   optimize_flags=[
                                                                                                       True] * len(du_augmentor.chain_of_transforms),
                                                                                                   power_iteration=power_iteration)
                    unsupervised_consistency_loss_record += (unsupervised_consistency_loss).item()
                    a_loss_dict['CR_u'] = unsupervised_consistency_loss.item()
                    regularization_loss = supervised_consistency_loss + unsupervised_consistency_loss
                    segmentation_model.train()
                    with torch.enable_grad():
                        pred_u = segmentation_model.model(image_u.detach().clone())
                    if entropy_flag:
                        entropy_u_loss = EntropyLoss(reduction='mean')(pred_u)
                    else:
                        entropy_u_loss = torch.tensor(0., dtype=image_u.dtype, device=image_u.device)
                    a_loss_dict['Ent_u'] = entropy_u_loss.item()
                    regularization_loss += entropy_u_loss
                    unsupervised_entropy_loss_record += entropy_u_loss.item()

            else:
                # other baseline methods
                if baseline_method != "":
                    if "mixmatch" == baseline_method:
                        assert SSL_flag
                        regularization_loss, labelled_regularization_loss, unlabelled_regularization_loss = segmentation_model.get_mix_loss(method="mixmatch", input_image_l=image_l, label_l=label_l, input_image_u=image_u,
                                                                                                                                            lambda__l=a_lambda_l, lambda_u=a_lambda_u)
                        a_loss_dict[baseline_method] = regularization_loss.item()

                    else:
                        if "mixup" == baseline_method:
                            regularization_loss, labelled_regularization_loss, unlabelled_regularization_loss = segmentation_model.get_mix_loss(
                                method='mixup', input_image_l=image_l, label_l=label_l, lambda_l=a_lambda_l)
                            if SSL_flag:
                                regularization_loss, labelled_regularization_loss, unlabelled_regularization_loss = segmentation_model.get_mix_loss(method="semi_mixup", input_image_l=image_l, label_l=label_l, input_image_u=image_u,
                                                                                                                                                    lambda__l=a_lambda_l, lambda_u=a_lambda_u)

                        else:
                            labelled_regularization_loss = a_lambda_l * \
                                segmentation_model.get_advanced_loss(method=baseline_method, input_image=image_l)
                            if SSL_flag:
                                unlabelled_regularization_loss = a_lambda_u * \
                                    segmentation_model.get_advanced_loss(method=baseline_method, input_image=image_u)
                            else:
                                unlabelled_regularization_loss = 0 * labelled_regularization_loss
                            regularization_loss = labelled_regularization_loss + unlabelled_regularization_loss
                            supervised_consistency_loss_record += labelled_regularization_loss.item()
                            unsupervised_consistency_loss_record += unlabelled_regularization_loss.item()

                    a_loss_dict[baseline_method + ": CR_l"] = labelled_regularization_loss.item()
                    a_loss_dict[baseline_method + ": CR_u"] = unlabelled_regularization_loss.item()
                    a_loss_dict[baseline_method + ": CR"] = regularization_loss.item()
                    supervised_consistency_loss_record += supervised_consistency_loss.item()
                    unsupervised_consistency_loss_record += unsupervised_consistency_loss.item()
                else:
                    regularization_loss = 0.
            segmentation_model.train()
            with torch.enable_grad():
                pred_l = segmentation_model.forward(image_l)
            if 'std_losses' in experiment_opt['learning'].keys() and 'std_weights' in experiment_opt['learning'].keys():
                original_loss = 0.
                for loss_name, loss_weight in zip(experiment_opt['learning']['std_losses'], experiment_opt['learning']['std_weights']):
                    loss = loss_weight * segmentation_model.basic_loss_fn(pred=pred_l, target=label_l.long(),
                                                                          loss_type=loss_name, class_weights=class_weights)
                    a_loss_dict[loss_name] = loss.item()
                    original_loss += loss
            else:
                original_loss = segmentation_model.basic_loss_fn(pred=init_output_l, target=label_l.long(),
                                                                 loss_type='weighted dice', class_weights=None) + segmentation_model.basic_loss_fn(
                    pred=init_output_l, target=label_l.long(), loss_type='weighted cross entropy', class_weights=class_weights)

            std_loss_record += original_loss.item()

            if entropy_flag:
                entropy_l_loss = EntropyLoss(reduction='mean')(init_output_l)
            else:
                entropy_l_loss = torch.tensor(0., dtype=image_l.dtype, device=image_l.device)
            a_loss_dict["Ent l"] = entropy_l_loss.item()
            supervised_entropy_loss_record += entropy_l_loss.item()

            segmentation_loss = original_loss + entropy_l_loss + regularization_loss
            print(a_loss_dict)

            segmentation_model.reset_optimizers()
            segmentation_loss.backward()
            segmentation_model.optimize_params()
            total_loss += segmentation_loss.item()
            segmentation_model.reset_loss()
            g_count += 1
            i_iter += 1
            if i_iter > segmentor_opt["max_iteration"]:
                stop_flag = True

        print('{} network: {} epoch {} training loss iter: {}, total  loss: {}'.
              format(experiment_name, segmentor_opt["network_type"], i_epoch, g_count, str(total_loss / (1.0 * g_count))))

        if log:
            # plt trainning curve
            writer.add_scalar('loss/total', (total_loss / (1.0 * g_count + 1e-6)), i_epoch)
            writer.add_scalar('loss/supervised', (std_loss_record / (1.0 * g_count)), i_epoch)
            writer.add_scalar('loss/supervised_consistency_loss',
                              (supervised_consistency_loss_record / (1.0 * g_count)), i_epoch)
            writer.add_scalar('loss/unsupervised_consistency_loss',
                              (unsupervised_consistency_loss_record / (1.0 * g_count)), i_epoch)
            writer.add_scalar('loss/unsupervised_entropy',
                              (unsupervised_entropy_loss_record / (1.0 * g_count)), i_epoch)
            writer.add_scalar('loss/supervised_entropy', (supervised_entropy_loss_record / (1.0 * g_count)), i_epoch)

        # =========================<<<<<start evaluating>>>>>>>>=============================>
        if use_ema:
            segmentation_model.model = segmentation_model.get_teacher_model()

        segmentation_model.eval()

        def eval_model(segmentation_model, validate_loader, keep_origin=True):
            segmentation_model.running_metric.reset()
            for b_iter, batch in enumerate(validate_loader):
                random_sax_image, random_sax_gt = batch['image'], batch['label']
                if keep_origin:
                    image_orig, gt_orig = labelled_batch['origin_image'], labelled_batch['origin_label']
                    random_sax_image = torch.cat([random_sax_image, image_orig], dim=0)
                    random_sax_gt = torch.cat([random_sax_gt, gt_orig], dim=0)
                random_sax_image_V = makeVariable(random_sax_image, type='float',
                                                  use_gpu=segmentor_opt["use_gpu"], requires_grad=True)
                segmentation_model.evaluate(input=random_sax_image_V,
                                            targets_npy=random_sax_gt.numpy())

            score = print_metric(segmentation_model.running_metric, name=experiment_name)
            # keep the best model
            curr_score = score['Mean IoU : \t']
            curr_acc = score['Mean Acc : \t']
            return curr_score, curr_acc

        # curr_test_score,curr_test_acc = eval_model(segmentation_model,test_loader,keep_origin=False)
        curr_score, curr_acc = eval_model(segmentation_model, validate_loader, keep_origin=True)
        score_list.append(curr_score)
        if log:
            # plt validation curve
            writer.add_scalar('iou/validate iou', curr_score, i_epoch)
            writer.add_scalar('acc/validate acc', curr_acc, i_epoch)

        if best_score < curr_score:
            best_score = curr_score
            segmentation_model.save_model(save_dir, epoch_iter='best', model_prefix=segmentor_opt["network_type"])
            segmentation_model.save_testing_images_results(save_dir, epoch_iter='best', max_slices=5)

        ###########save outputs ####################################################################
        if i_epoch % experiment_opt["output"]["save_epoch_every_num_epochs"] == 0 or i_epoch == 1 or i_epoch == n_epochs - 1:
            segmentation_model.save_model(save_dir, epoch_iter=i_epoch, model_prefix=segmentor_opt["network_type"])
            segmentation_model.save_testing_images_results(save_dir, epoch_iter=i_epoch, max_slices=5)
            gc.collect()  # collect garbage

        if use_ema:
            segmentation_model.model = segmentation_model.get_student_model()
        if segmentation_model.scheduler is not None:
            segmentation_model.scheduler.step()
        if stop_flag:
            break

    if log:
        try:
            writer.export_scalars_to_json(join(save_dir, experiment_name + ".json"))
            writer.close()
        except:
            print('already closed')


# ========================= config==================================================#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cardiac segmentation network training')
    parser.add_argument("--json_config_path", type=str, default='./configs/ACDC/multi_class/mix/four_chain_affine_adv.json',
                        help='path of configurations')

    parser.add_argument("--cval", type=int, default=0,
                        help="cross validation subset")
    parser.add_argument("--data_setting", type=str, default="three_shot",
                        help="data_setting:['one_shot','three_shot']")
    # network setting
    parser.add_argument("--resume_path", type=str, default='./result/ACDC_Segmentation/multi_class_baseline/three_shot/multi_class/finetune/affine_elastic_intensity_opt/0_cval_0/best/checkpoints/UNet_16$SAX$_Segmentation.pth', help='path to resume the models')

    parser.add_argument("--log", action='store_true', default=False,
                        help='use tensorboardX to tracking the training and testing curve')
    parser.add_argument("--save_dir", type=str,
                        default="./result/ACDC_Segmentation/multi_class_baseline/",
                        help='path to resume the models')
    parser.add_argument("--debug", action='store_true', default=False,
                        help='use debug')

    # ========================= initialize training settings==================================================#
    # first load basic settings and then load args, finally load experiment configs

    # enabale this for optimal algorithm tuned for your hardware when your inputs do not vary in size.
    torch.backends.cudnn.benchmark = True

    training_opt = parser.parse_args()
    if exists(training_opt.json_config_path):
        print('load params from {}'.format(training_opt.json_config_path))
        experiment_opt = Params(training_opt.json_config_path).dict
    else:  #
        raise FileNotFoundError

    # input dataset setting
    data_opt = experiment_opt['data']
    data_aug_policy_name = data_opt["data_aug_policy"]

    resume_path = training_opt.resume_path
    save_dir = training_opt.save_dir

    # training setting
    SSL_flag = experiment_opt["learning"]["semi"]

    # ========================= initialize training settings==================================================#
    tr = Transformations(data_aug_policy_name=data_opt["data_aug_policy"], pad_size=data_opt['pad_size'],
                         crop_size=data_opt['crop_size']).get_transformation()
    train_set = CardiacACDCDataset(root_dir=data_opt["root_dir"], num_classes=data_opt["num_classes"],
                                   image_format_name=data_opt["image_format_name"],
                                   label_format_name=data_opt["label_format_name"],
                                   transform=tr['train'], subset_name=data_opt['frame'], split='train',
                                   data_setting_name=training_opt.data_setting,
                                   cval=training_opt.cval,
                                   keep_orig_image_label_pair=data_opt['keep_orig_image_label_pair_for_training'],
                                   use_cache=data_opt['use_cache'],
                                   myocardium_seg=data_opt['myocardium_only']
                                   )

    validate_set = CardiacACDCDataset(root_dir=data_opt["root_dir"], num_classes=data_opt["num_classes"],
                                      image_format_name=data_opt["image_format_name"],
                                      label_format_name=data_opt["label_format_name"],
                                      transform=tr['train'], subset_name=data_opt['frame'],
                                      split='validate',
                                      data_setting_name=training_opt.data_setting,
                                      cval=training_opt.cval,
                                      use_cache=data_opt['use_cache'],
                                      myocardium_seg=data_opt['myocardium_only'],
                                      keep_orig_image_label_pair=True)
    # test_set = CardiacACDCDataset(root_dir=data_opt["root_dir"], num_classes=data_opt["num_classes"],
    #                                 image_format_name=data_opt["image_format_name"],
    #                                 label_format_name=data_opt["label_format_name"],
    #                                 transform=tr['validate'], subset_name=data_opt['frame'], split='test',
    #                                 data_setting_name=training_opt.data_setting,
    #                                 cval=training_opt.cval,
    #                                 keep_orig_image_label_pair=False,
    #                                 use_cache=data_opt['use_cache'],
    #                                 myocardium_seg=data_opt['myocardium_only']
    #                                 )
    test_set = None

    if SSL_flag:
        if 'unlabelled_data_aug_policy' in data_opt.keys():
            unlabelled_data_tr = Transformations(data_aug_policy_name=data_opt["unlabelled_data_aug_policy"], pad_size=data_opt['pad_size'],
                                                 crop_size=data_opt['crop_size']).get_transformation()
        else:
            unlabelled_data_tr = tr

        if 'unlabelled_keep_orig_image_label_pair_for_training' in data_opt.keys():
            keep_origin = data_opt['unlabelled_keep_orig_image_label_pair_for_training']
            print('unlabelled data policy:', data_opt['unlabelled_data_aug_policy'])

        else:
            keep_origin = data_opt['keep_orig_image_label_pair_for_training'],

        unlabelled_train_set = CardiacACDCDataset(root_dir=data_opt["root_dir"], num_classes=data_opt["num_classes"],
                                                  image_format_name=data_opt["image_format_name"],
                                                  label_format_name=data_opt["label_format_name"],
                                                  transform=unlabelled_data_tr['train'], subset_name=data_opt['frame'], split='unlabelled',
                                                  data_setting_name=training_opt.data_setting,
                                                  cval=training_opt.cval,
                                                  keep_orig_image_label_pair=keep_origin,
                                                  use_cache=data_opt['use_cache'],
                                                  myocardium_seg=data_opt['myocardium_only']
                                                  )
    else:
        unlabelled_train_set = None

    #     combined_dataset = CombinedDataSet(source_dataset=train_set, target_dataset=unlabelled_train_set)
    # else:
    #     combined_dataset =train_set

    datasets = [train_set, unlabelled_train_set, validate_set, test_set]

    idx2cls_dict = train_set.idx2cls_dict
    # ========================= start training ==================================================#

    print('train_{}_with_{}_datasize:{}_num_classes:{}'.format(data_opt['dataset_name'], data_aug_policy_name, str(
        train_set.datasize), str(experiment_opt['segmentation_model']["num_classes"])))

    config_name = (training_opt.json_config_path.split('.')[0]).replace('configs/', '')
    print('configure name', config_name)
    experiment_name = config_name + '_' + training_opt.data_setting + '_' + str(training_opt.cval)

    folder_name = "{data_setting}/{exp}/{cval}".format(data_setting=str(
        training_opt.data_setting), exp=config_name, cval=str(training_opt.cval))
    save_dir = os.path.join(save_dir, folder_name)
    check_dir(save_dir, create=True)
    train_network(training_opt, experiment_name=experiment_name,
                  dataset=datasets,
                  segmentor_opt=experiment_opt['segmentation_model'],
                  segmentor_resume_path=resume_path,
                  experiment_opt=experiment_opt,
                  save_dir=save_dir,
                  log=training_opt.log)
