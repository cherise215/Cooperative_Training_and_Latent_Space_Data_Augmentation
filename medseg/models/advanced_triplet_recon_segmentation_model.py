# this segmentation model is composed of 2 subnetworks at least, an encoder and an decoder

import itertools

import random
import os
from os.path import join
import torch.nn as nn
import torch
import torch.optim as optim
import gc
import torch.nn.functional as F

from medseg.models.init_weight import init_weights
from medseg.models.ebm.encoder_decoder import MyEncoder, MyDecoder, Dual_Branch_Encoder
from medseg.models.model_util import _disable_tracking_bn_stats, makeVariable, mask_latent_code_channel_wise, mask_latent_code_spatial_wise
from medseg.models.custom_loss import basic_loss_fn

from medseg.common_utils.metrics import runningScore
from medseg.common_utils.basic_operations import construct_input, set_grad, rescale_intensity
from medseg.common_utils.save import save_testing_images_results


class AdvancedTripletReconSegmentationModel(nn.Module):
    def __init__(self, network_type='FCN_16_standard', image_ch=1,
                 learning_rate=1e-4,
                 encoder_dropout=None,
                 decoder_dropout=None,
                 num_classes=4, n_iter=1,
                 checkpoint_dir=None, use_gpu=True, debug=False
                 ):
        """[summary]

        Args:
            network_type (str): network arch name. Default: FCN_16_standard
            image_ch (int, optional):image channel number. Defaults to 1.
            learning_rate (float, optional): learning rate for network parameter optimization. Defaults to 1e-4.
            encoder_dropout (float, optional): [description]. Defaults to None.
            decoder_dropout (float, optional): [description]. Defaults to None.
            num_classes (int, optional): [description]. Defaults to 4.
            n_iter (int, optional): If set to 1, will use FTN's output as final prediction at test time. If set to 2, will use STN's refinement as the final prediction. Defaults to 1.
            checkpoint_dir (str, optional): path to the checkpoint directory. Defaults to None.
            use_gpu (bool, optional): [description]. Defaults to True.
            debug (bool, optional): [description]. Defaults to False.
        """
        super(AdvancedTripletReconSegmentationModel, self).__init__()
        self.network_type = network_type
        self.image_ch = image_ch
        self.checkpoint_dir = checkpoint_dir
        self.num_classes = num_classes

        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.use_gpu = use_gpu
        self.debug = debug

        # initialization
        self.model = self.get_network(checkpoint_dir=checkpoint_dir)
        self.optimizers = None
        self.reset_all_optimizers()
        self.latent_code = {'image': None,
                            'segmentation': None,
                            'shape': None  # latent code from the denoising autoencoder
                            }

        self.running_metric = self.set_running_metric()  # cal iou score during training

        self.cur_eval_images = None
        self.cur_eval_predicts = None
        self.cur_eval_gts = None  # N*H*W
        self.cur_time_predicts = {}
        self.loss = 0.

    def get_network(self, checkpoint_dir=None):
        '''
        get a network model, if checkpoint dir is not none, load weights from the disk
        return a model
        '''
        ##
        network_type = self.network_type
        print('construct {}'.format(network_type))
        shape_inc_ch = self.num_classes

        if network_type in ['FCN_16_standard', 'FCN_16_standard_w_o_filter', 'FCN_16_standard_share_code']:
            if '16' in network_type:
                reduce_factor = 4
            else:
                raise ValueError

            # FTN
            image_encoder = Dual_Branch_Encoder(input_channel=self.image_ch, z_level_1_channel=512 // reduce_factor, z_level_2_channel=512 //
                                                reduce_factor, feature_reduce=reduce_factor, if_SN=False, encoder_dropout=self.encoder_dropout,
                                                norm=nn.BatchNorm2d)
            segmentation_decoder = MyDecoder(input_channel=512 // reduce_factor, up_type='NN', output_channel=self.num_classes,
                                             feature_reduce=reduce_factor, if_SN=False, decoder_dropout=self.decoder_dropout, norm=nn.BatchNorm2d)

            image_decoder = MyDecoder(input_channel=512 // reduce_factor, up_type='Conv2', output_channel=self.image_ch,
                                      feature_reduce=reduce_factor, if_SN=False, decoder_dropout=self.decoder_dropout, norm=nn.BatchNorm2d, last_act=nn.Sigmoid())

            # STN
            shape_encoder = MyEncoder(input_channel=shape_inc_ch, output_channel=512 // reduce_factor, feature_reduce=reduce_factor,
                                      if_SN=False, encoder_dropout=self.encoder_dropout, norm=nn.BatchNorm2d, act=nn.ReLU())
            shape_decoder = MyDecoder(input_channel=512 // reduce_factor, up_type='NN', output_channel=self.num_classes,
                                      feature_reduce=reduce_factor, if_SN=False, decoder_dropout=self.decoder_dropout, norm=nn.BatchNorm2d)
            # load weights
            image_encoder_path = None
            segmentation_decoder_path = None
            shape_decoder_path = None
            shape_encoder_path = None
            image_decoder_path = None

            if checkpoint_dir is not None and not checkpoint_dir == "":
                image_encoder_path = join(checkpoint_dir, 'image_encoder.pth')
                shape_decoder_path = join(checkpoint_dir, 'shape_decoder.pth')
                shape_encoder_path = join(checkpoint_dir, 'shape_encoder.pth')
                image_decoder_path = join(checkpoint_dir, 'image_decoder.pth')
                segmentation_decoder_path = join(
                    checkpoint_dir, 'segmentation_decoder.pth')

            image_encoder = self.init_model(
                image_encoder, resume_path=image_encoder_path)
            shape_decoder = self.init_model(
                shape_decoder, resume_path=shape_decoder_path)
            shape_encoder = self.init_model(
                shape_encoder, resume_path=shape_encoder_path)
            segmentation_decoder = self.init_model(
                segmentation_decoder, resume_path=segmentation_decoder_path)
            image_decoder = self.init_model(
                image_decoder, resume_path=image_decoder_path)

            if self.use_gpu:
                image_encoder.to('cuda')
                segmentation_decoder.to('cuda')
                shape_decoder.to('cuda')
                shape_encoder.to('cuda')
                image_decoder.to('cuda')

            model = {'image_encoder': image_encoder,
                     'segmentation_decoder': segmentation_decoder,
                     'shape_encoder': shape_encoder,
                     'shape_decoder': shape_decoder,
                     'image_decoder': image_decoder,
                     }
        else:
            raise NotImplementedError

        return model

    def parameters(self):
        return itertools.chain([module.parameters() for module in self.model.values()])

    def named_parameters(self):
        return itertools.chain([module.named_parameters() for module in self.model.values()])

    def init_model(self, model, resume_path=None):
        if not resume_path is None:
            if not resume_path == '':
                assert os.path.exists(
                    resume_path), 'path: {} must exist'.format(resume_path)
                try:
                    model.load_state_dict(torch.load(resume_path))
                    print(f'load saved params from {resume_path}')
                except:
                    try:
                        # dummy code for some historical reason.
                        model.load_state_dict(torch.load(resume_path)[
                                              'model_state'], strict=False)
                    except:
                        print('fail to load checkpoint under {}'.format(resume_path))
            else:
                print('can not find checkpoint under {}'.format(resume_path))
        else:
            try:
                init_weights(model, init_type='kaiming')
                print('init network')
            except:
                print('failed to init model')
        return model

    def run(self, input):
        zi, zs = self.encode_image(input)
        decoder = self.model['segmentation_decoder']

        recon_image = self.decode_image(zi)
        init_predict = decoder(zs)
        refined_predict = self.recon_shape(init_predict)
        return recon_image, init_predict, refined_predict

    def encode_image(self, input, disable_track_bn_stats=False):
        # FTN encoders
        encoder = self.model['image_encoder']
        if disable_track_bn_stats:
            with _disable_tracking_bn_stats(encoder):
                latent_code_i, latent_code_s = encoder(input)
        else:
            latent_code_i, latent_code_s = encoder(input)
        if 'share_code' in self.network_type:
            # z_i and z_s are shared ## for ablation study
            latent_code_i = latent_code_s
        elif 'w_o_filter' in self.network_type:
            latent_code_s = latent_code_i
        self.latent_code['image'] = latent_code_i
        self.latent_code['segmentation'] = latent_code_s
        return latent_code_i, latent_code_s

    def decode_segmentation_from_image_code(self, latent_code_i, disable_track_bn_stats=False):
        # FTN segmentation decoder function: z_s -> S
        decoder = self.model['segmentation_decoder']
        encoder = self.model['image_encoder']

        if disable_track_bn_stats:
            with _disable_tracking_bn_stats(encoder):
                z_s = encoder.filter_code(latent_code_i)
            with _disable_tracking_bn_stats(decoder):
                segmentation = decoder(z_s)
        else:
            z_s = encoder.filter_code(latent_code_i)
            segmentation = decoder(z_s)
        return segmentation

    def decode_image(self, latent_code, disable_track_bn_stats=False):
        # FTN image decoder function: z_i -> I
        image_decoder = self.model['image_decoder']
        if disable_track_bn_stats:
            with _disable_tracking_bn_stats(image_decoder):
                pred = image_decoder(latent_code)
        else:
            pred = image_decoder(latent_code)
        return pred

    def encode_shape(self, segmentation, is_label_map=False, disable_track_bn_stats=False, temperature=2):
        '''
        STN: encoder function: S -> latent_z (STN)
        given a logit from the network or gt labels, encode it to the latent space
        '''
        prediction_map = construct_input(segmentation, image=None, num_classes=self.num_classes, apply_softmax=not is_label_map,
                                         is_labelmap=is_label_map, temperature=temperature, use_gpu=self.use_gpu, smooth_label=False)
        if disable_track_bn_stats:
            with _disable_tracking_bn_stats(self.model['shape_encoder']):
                shape_code = self.model['shape_encoder'](prediction_map)
        else:
            shape_code = self.model['shape_encoder'](prediction_map)
        self.latent_code['shape'] = shape_code
        return shape_code

    def decode_shape(self, latent_code, disable_track_bn_stats=False):
        '''
        STN:
        decoder function: latent_z (STN) -> S
        '''

        shape_decoder = self.model['shape_decoder']
        if disable_track_bn_stats:
            with _disable_tracking_bn_stats(shape_decoder):
                pred = shape_decoder(latent_code)
        else:
            pred = shape_decoder(latent_code)
        return pred

    def recon_shape(self, segmentation_logit, is_label_map=False, disable_track_bn_stats=False):
        '''
        STN: shape refinement/correction: S'-> STN(S)
        return logit of reconstructed shape
        '''
        recon_shape = self.decode_shape(self.encode_shape(segmentation_logit, is_label_map,
                                                          disable_track_bn_stats), disable_track_bn_stats)
        return recon_shape

    def recon_image(self, image, disable_track_bn_stats=False):
        '''
        FTN: image recon, I-> FTN-> I'
        return reconstructed shape
        '''
        z_i, z_s = self.encode_image(
            image, disable_track_bn_stats=disable_track_bn_stats)
        recon_image = self.decode_image(
            z_i, disable_track_bn_stats=disable_track_bn_stats)
        return recon_image

    def forward(self, input):
        '''
        predict fast segmentation (FTN)
        '''
        zs, predict = self.fast_predict(input)
        return predict

    def eval(self):
        self.train(if_testing=True)

    def requires_grad_(self, requires_grad=True):
        for module in self.model.values:
            for p in module.parameters():
                p.requires_grad = requires_grad

    def get_modules(self):
        return self.model.values

    def perturb_latent_code(self, latent_code, decoder_function, label_y=None,
                            perturb_type='random', threshold=0.5,
                            if_soft=False, random_threshold=False,
                            loss_type='mse', if_detach=False):
        """

        Args:
            latent_code (torch tensor): latent code z (a low-dimensional latent representation)
            decoder_function (nn.module): decoder function. a function that maps the latent code to the output space (image/label)
            label_y (torch tensor, optional): target value. Defaults to None. For targeted masking, it requires a target to compute the loss for gradient computation.
            perturb_type (str, optional): Names of mask methods. Defaults to 'random'. If random, will randomly select a method from the pool: ['dropout', 'spatial', 'channel']
            threshold (float, optional): dropout rate for random dropout or threshold for targeted masking:  mask codes with top p% gradients. Defaults to 0.5.
            if_soft (bool, optional): Use annealing factor to produce a soft mask with mask values sampled from [0,0.5] instead of 0. Defaults to False.
            random_threshold (bool, optional): Random sample a threshold from (0,threshold]. Defaults to False.
            loss_type (str, optional): Task-specific loss for targeted masking. Defaults to 'mse'.
            if_detach: If set to ``True``, will return the cloned masked code. Defaults to False
        Raises:
            NotImplementedError: [description]

        Returns:
            [type]: [description]
        """
        assert perturb_type in ['random', 'dropout',
                                'spatial', 'channel'], 'invalid method name'

        if perturb_type == 'random':
            # random select a perturb type from 'dropout', 'spatial', 'channel'
            random_perturb_candidates = ['dropout', 'spatial', 'channel']
            random.shuffle(random_perturb_candidates)
            perturb_type = random_perturb_candidates[0]

        # print(perturb_type)
        if perturb_type == 'dropout':
            masked_latent_code = F.dropout2d(latent_code, p=threshold)
            mask = torch.where(masked_latent_code == latent_code,
                               torch.ones_like(masked_latent_code),
                               torch.zeros_like(masked_latent_code))
        else:
            assert loss_type in ['mse', 'ce', 'corr'], 'not implemented loss'
            if perturb_type == 'spatial':
                masked_latent_code, mask = mask_latent_code_spatial_wise(latent_code, num_classes=self.num_classes, decoder_function=decoder_function,
                                                                         label=label_y, percentile=threshold, random=random_threshold, loss_type=loss_type, if_detach=if_detach, if_soft=if_soft)
            elif perturb_type == 'channel':
                masked_latent_code, mask = mask_latent_code_channel_wise(latent_code, num_classes=self.num_classes, decoder_function=decoder_function,
                                                                         label=label_y, percentile=threshold, random=random_threshold, loss_type=loss_type, if_detach=if_detach, if_soft=if_soft)
            else:
                raise NotImplementedError
        if if_detach:
            masked_latent_code = masked_latent_code.detach().clone()
        torch.cuda.empty_cache()
        return masked_latent_code, mask

    def get_recon_diff(self, input):
        """[summary]
        given an input image, produce a segmentation, corrected segmentation, and reconstructed images
        Args:
            input ([torch tensor]): images in [NCHW] format
        returns:
            abs image diff: |image-image_recon|
            abs segmentation diff: |segmentation-recon_segmentation|
            intial segmentation:
            recfine segmentation:
            recon image:

        """
        self.eval()
        (latent_code_i, latent_code_s), first_prediction = self.fast_predict(input)
        refined_prediction = self.recon_shape(
            first_prediction, is_label_map=False)
        reconstructed_image = self.decode_image(latent_code=latent_code_i)
        abs_image_diff = torch.abs(input - reconstructed_image)
        abs_segmentation_diff = torch.abs(
            refined_prediction - first_prediction)
        return abs_image_diff, abs_segmentation_diff, first_prediction, refined_prediction, reconstructed_image

    def predict(self, input, softmax=False, n_iter=None):
        self.eval()
        if n_iter is None:
            n_iter = self.n_iter
        else:
            n_iter = n_iter
        gc.collect()  # collect garbage
        with torch.no_grad():
            if n_iter <= 1:
                z0, pred = self.fast_predict(input)
            elif n_iter >= 1:
                z0, pred = self.fast_predict(input)
                for i in range(n_iter - 1):
                    pred, internal_predicts = self.slow_refinement(
                        pred_logit=pred, n_steps=n_iter, save_internal_predicts=False)

        if softmax:
            pred = torch.softmax(pred, dim=1)
        torch.cuda.empty_cache()
        return pred

    def decoder_inference(self, decoder, latent_code, eval=False, disable_track_bn_stats=False):
        decoder_state = decoder.training
        if eval:
            decoder.eval()
            with torch.no_grad():
                logit = decoder(latent_code)

        else:
            if disable_track_bn_stats:
                with _disable_tracking_bn_stats(decoder):
                    logit = decoder(latent_code)

            else:
                logit = decoder(latent_code)

        decoder.train(mode=decoder_state)
        return logit

    def standard_training(self, clean_image_l, label_l, perturbed_image, separate_training=False, compute_gt_recon=True, update_latent=True, disable_track_bn_stats=False):
        """
        compute standard training loss
        Args:
            clean_image_l (torch tensor): original images (w/o corruption) NCHW
            label_l (torch tensor): reference segmentation. NHW
            perturbed_image (torch tensor): corrupted/noisy images. NCHW
            separate_training (bool, optional): if true, will block the gradients flow from STN to FTN. Defaults to False.
            compute_gt_recon (bool, optional): compute shape correction loss where input to STN is the ground truth map. Defaults to True.
            update_latent (bool, optional): save the latent codes. Defaults to True.

        Returns:
            standard_supervised_loss (float tensor): task-specific loss (ce loss for segmentation)
            image_recon_loss (float tensor): image reconstruction loss (mse for image recon)
            gt_shape_recon_loss (float tensor): shape correction loss (reconstruct the input label map)
            pred_shape_recon_loss (float tensor): shape correction loss (refine the output from FTN)
        """

        zero = torch.tensor(0., device=clean_image_l.device)

        (z_i, z_s), y_0 = self.fast_predict(perturbed_image,
                                            disable_track_bn_stats=disable_track_bn_stats)
        if update_latent:
            self.z_i = z_i
            self.z_s = z_s
        # seg task loss
        standard_supervised_loss = basic_loss_fn(
            pred=y_0, target=label_l.detach(), loss_type='cross entropy')

        # image recon loss
        image_recon = self.decode_image(z_i)
        image_recon_loss = 0.5 * \
            torch.nn.MSELoss(reduction='mean')(
                input=image_recon, target=clean_image_l)

        # shape recon loss
        if compute_gt_recon:
            gt_recon = self.recon_shape(
                label_l.detach().clone(), is_label_map=True)
            gt_shape_recon_loss = basic_loss_fn(
                pred=gt_recon, target=label_l, loss_type='cross entropy')
        else:
            gt_shape_recon_loss = zero

        if separate_training:
            y_0_new = makeVariable(y_0.detach().clone(
            ), requires_grad=False, type='float', use_gpu=self.use_gpu)
        else:
            y_0_new = y_0
        p_recon = self.recon_shape(
            y_0_new, is_label_map=False, disable_track_bn_stats=disable_track_bn_stats)
        pred_shape_recon_loss = basic_loss_fn(
            pred=p_recon, target=label_l, loss_type='cross entropy')
        return standard_supervised_loss, image_recon_loss, gt_shape_recon_loss, pred_shape_recon_loss

    def hard_example_generation(self,
                                clean_image_l,
                                label_l,
                                gen_corrupted_seg=True,
                                gen_corrupted_image=True,
                                corrupted_image_DA_config={"loss_name": "mse",
                                                           "mask_type": "random",
                                                           "max_threshold": 0.5,
                                                           "random_threshold": True,
                                                           "if_soft": True},
                                corrupted_seg_DA_config={"loss_name": "ce",
                                                         "mask_type": "random",
                                                         "max_threshold": 0.5,
                                                         "random_threshold": True,
                                                         "random_threshold": True,
                                                         "if_soft": True}):
        # fixed segmentation decoder, we perturb the latent space to get corrupted segmentation, and use them to train our denoising shape autoencodeer,
        set_grad(self.model['segmentation_decoder'], requires_grad=False)
        set_grad(self.model['image_decoder'], requires_grad=False)
        perturbed_image_0, perturbed_y_0 = None, None
        if gen_corrupted_image:
            self.reset_all_optimizers()
            perturbed_z_i_0, img_code_mask = self.perturb_latent_code(latent_code=self.z_i,
                                                                      label_y=clean_image_l,
                                                                      perturb_type=corrupted_image_DA_config[
                                                                          "mask_type"],
                                                                      decoder_function=self.model['image_decoder'],
                                                                      loss_type=corrupted_image_DA_config["loss_name"],
                                                                      threshold=corrupted_image_DA_config[
                                                                          "max_threshold"],
                                                                      random_threshold=corrupted_image_DA_config[
                                                                          "random_threshold"],
                                                                      if_detach=True, if_soft=corrupted_image_DA_config["if_soft"])
            perturbed_image_0 = self.decoder_inference(decoder=self.model['image_decoder'],
                                                       latent_code=perturbed_z_i_0, eval=False, disable_track_bn_stats=True)
        if gen_corrupted_seg:
            self.reset_all_optimizers()
            # print ('perform shape code perturbation')
            perturbed_z_0, shape_code_mask = self.perturb_latent_code(latent_code=self.z_s,
                                                                      label_y=label_l,
                                                                      perturb_type=corrupted_seg_DA_config["mask_type"],
                                                                      decoder_function=self.model['segmentation_decoder'],
                                                                      loss_type=corrupted_seg_DA_config["loss_name"],
                                                                      threshold=corrupted_seg_DA_config["max_threshold"],
                                                                      random_threshold=corrupted_seg_DA_config[
                                                                          "random_threshold"],
                                                                      if_detach=True, if_soft=corrupted_seg_DA_config["if_soft"])

            perturbed_y_0 = self.decoder_inference(decoder=self.model['segmentation_decoder'],
                                                   latent_code=perturbed_z_0, eval=False, disable_track_bn_stats=True)

        set_grad(self.model['segmentation_decoder'], requires_grad=True)
        set_grad(self.model['image_decoder'], requires_grad=True)

        return perturbed_image_0, perturbed_y_0

    def hard_example_training(self, perturbed_image, clean_image_l, perturbed_seg, label_l, separate_training=False, use_gpu=True):
        """
        compute hard training loss
        Args:
           perturbed_image (torch tensor): corrupted/noisy images. NCHW
           clean_image_l (torch tensor): original images (w/o corruption) NCHW
           perturbed_seg (torch tensor): corrupted segmentation. NCHW
           label_l (torch tensor): reference segmentation. NHW
           separate_training (bool, optional): if true, will block the gradients flow from STN to FTN. Defaults to False.
           use gpu (bool, optional): use gpu. Defaults to True.
        Returns:
           seg_loss (float tensor):  segmentation loss given the corrupted image
           recon_loss (float tensor): image reconstruction loss (input is the corrupted imaeg)
           shape_loss (float tensor): shape correction loss (input is the FTN's prediction on corrupted images)
           perturbed_p_recon_loss (float tensor): shape correction loss (input is the generated corrupted segmentations by code masking)
        """
        zero = torch.tensor(0., device=perturbed_image.device)
        seg_loss, recon_loss, shape_loss, perturbed_p_recon_loss = zero, zero, zero, zero
        if perturbed_image is not None:
            # w. corrupted image
            perturbed_image = makeVariable(
                perturbed_image.detach().clone(), use_gpu=use_gpu, type='float')
            seg_loss, recon_loss, _, shape_loss = self.standard_training(clean_image_l=clean_image_l, label_l=label_l,
                                                                         perturbed_image=perturbed_image, compute_gt_recon=False, separate_training=separate_training, update_latent=False, disable_track_bn_stats=True)

        if perturbed_seg is not None:
            # w. corrupted segmentation
            if separate_training:
                perturbed_seg = perturbed_seg.detach().clone()
            perturbed_p_recon = self.recon_shape(
                perturbed_seg, is_label_map=False, disable_track_bn_stats=True)
            perturbed_p_recon_loss = basic_loss_fn(
                pred=perturbed_p_recon, target=label_l, loss_type='cross entropy')

        return seg_loss, recon_loss, shape_loss, perturbed_p_recon_loss

    def fast_predict(self, input, disable_track_bn_stats=False):
        """
        given an input image, return its latent code and pixelwise prediction

        Args:
            input ([type]): torch tensor
            disable_track_bn_stats (bool, optional):disable bn stats tracking. Defaults to False.

        Returns:
            z0: latent code tuple
            p0: pixelwise logits from the model
        """
        gc.collect()  # collect garbage
        encoder = self.model['image_encoder']
        decoder = self.model['segmentation_decoder']
        if not self.training:
            with torch.no_grad():
                z_i, z_s = encoder(input)
                if 'share_code' in self.network_type:
                    z_i = z_s
                elif 'w_o_filter' in self.network_type:
                    z_s = z_i
                y_0 = decoder(z_s)

        else:
            if disable_track_bn_stats:
                with _disable_tracking_bn_stats(encoder):
                    z_i, z_s = encoder(input)
            else:
                z_i, z_s = encoder(input)
            if 'share_code' in self.network_type:
                z_i = z_s
            elif 'w_o_filter' in self.network_type:
                z_s = z_i

            if disable_track_bn_stats:
                with _disable_tracking_bn_stats(decoder):
                    y_0 = decoder(z_s)
            else:
                y_0 = decoder(z_s)
        return (z_i, z_s), y_0

    def predict_w_reconstructed_image(self, image):
        image_recon = self.recon_image(image)
        (zi, zs), pred = self.fast_predict(image_recon)
        return pred

    def slow_refinement(self, pred_logit, n_steps=1, auto_stop=False, save_internal_predicts=False):
        """[summary]

        Args:
            pred ([torch tensor]): [initial prediction
            n_steps ([int], optional): [description]. Defaults to 1.
            auto_stop (bool, optional): [stop refinement when latent code is not changing anymore]. Defaults to False.

        Returns:
            s_t [torch tensor]: refined prediction
            internal_predicts: a dict: key: iter,  value: prediction
        """
        if n_steps is None:
            n_steps = self.n_iter

        break_flag = False
        internal_predicts = {}
        internal_predicts[0] = [pred_logit]
        s_t = pred_logit
        for i in range(n_steps):
            prev = s_t.clone()
            s_t = self.recon_shape(pred_logit.detach().clone())
            diff = torch.sqrt(torch.mean((prev - s_t)**2))
            if self.debug:
                print('iter: {} |z_new-z|_2^2: {}'.format(i, diff))
            if auto_stop:
                if diff < 1e-4:
                    s_t = prev
                    break_flag = True
            if save_internal_predicts:
                internal_predicts[i] = [s_t]
            if break_flag:
                break
        return s_t, internal_predicts

    def evaluate(self, input, targets_npy, n_iter=None):
        '''
        evaluate the model performance

        :param input: 4-d tensor input: NCHW
        :param targets_npy: numpy ndarray: N*H*W
        :param running_metric: runnning metric for evaluatation
        :return:
        '''
        if n_iter is None:
            n_iter = self.n_iter
        gc.collect()  # collect garbage
        self.train(if_testing=True)
        pred = self.predict(input, n_iter=n_iter)
        pred_npy = pred.max(1)[1].cpu().numpy()
        self.running_metric.update(
            label_trues=targets_npy, label_preds=pred_npy)
        self.cur_eval_images = input.data.cpu().numpy()[:, 0, :, :]
        del input
        self.cur_eval_predicts = pred_npy
        self.cur_eval_gts = targets_npy  # N*H*W
        return pred

    def save_model(self, save_dir, epoch_iter, model_prefix=None, save_optimizers=False):
        if model_prefix is None:
            model_prefix = self.network_type
        epoch_path = join(save_dir, *[str(epoch_iter), 'checkpoints'])
        if not os.path.exists(epoch_path):
            os.makedirs(epoch_path)
        for model_name, model in self.model.items():
            torch.save(model.state_dict(),
                       join(epoch_path, '{}.pth'.format(model_name)))
        if save_optimizers:
            for model_name, optimizer in self.optimizers.items():
                torch.save(optimizer.state_dict(),
                           join(epoch_path, '{}_optim.pth'.format(model_name)))

    def save_snapshots(self, save_dir, epoch, model_prefix='interrupted'):
        epoch_path = join(save_dir, *['interrupted', 'checkpoints'])
        if not os.path.exists(epoch_path):
            os.makedirs(epoch_path)
        if model_prefix is None:
            model_prefix = self.network_type
        save_path = join(epoch_path, self.network_type + '.pkl')
        model_states_dict = {}
        for model_name, model in self.model.items():
            state_dict = model.module.state_dict() if isinstance(
                self.model, torch.nn.DataParallel) else model.state_dict()
            model_states_dict[model_name] = state_dict
        optimizers_dict = {}
        for optimizer_name, optimizer in self.optimizers.items():
            optimizers_dict[optimizer_name] = optimizer.state_dict()
        state = {'network_type': self.network_type,
                 'epoch': epoch,
                 'model_state': model_states_dict,
                 'optimizer_state': optimizers_dict
                 }
        torch.save(state, save_path)
        return save_path

    def load_snapshots(self, file_path):
        """
        load checkpoints from the pkl file
        Args:
            file_path (str): path-to-checkpoint.pkl

        Returns:
            the epoch when saved (int):
        """
        start_epoch = 0
        if file_path is None:
            return start_epoch
        if file_path == '' or (not os.path.exists(file_path)):
            print(f'warning: {file_path} does not exists')
            return start_epoch
        try:
            checkpoint = torch.load(file_path)
        except:
            print('error in opening {}'.format(file_path))
        try:
            if self.model is None:
                self.get_network(network_type=checkpoint['network_type'])
            state_dicts = checkpoint['model_state']
            optimizer_states = checkpoint['optimizer_state']
            assert self.model, 'must initialize model first'
            assert self.optimizers, 'must initialize optimizer first'
            for k, v in self.model.items():
                v.load_state_dict(state_dicts[k])
            for k, v in self.optimizers.items():
                v.load_state_dict(optimizer_states[k])
            start_epoch = checkpoint['epoch']
            print("Loaded checkpoint '{}' (epoch {})".format(
                file_path, checkpoint['epoch']))
        except Exception as e:
            print('error: {} in loading {}'.format(e, file_path))
        return start_epoch

    def train(self, if_testing=False):
        self.training = True
        assert self.model, 'no model exists'
        for k, v in self.model.items():
            if not if_testing:
                v.train()
                set_grad(v, requires_grad=True)
            else:
                v.eval()

    def eval(self):
        self.training = False
        self.train(if_testing=True)

    def reset_all_optimizers(self):
        if self.optimizers is None:
            self.set_optimizers()
        for k, v in self.optimizers.items():
            v.zero_grad()

    def zero_grad(self):
        for model in self.model.values:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def get_optimizer(self, model_name=None):
        assert self.optimizers, 'please set optimizers first before fetching'
        if model_name is None:
            return self.optimizers
        else:
            return self.optimizers[model_name]

    def set_optimizers(self):
        assert self.model
        optimizers_dict = {}
        for model_name, model in self.model.items():
            print('set optimizer for:', model_name)
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            optimizers_dict[model_name] = optimizer
        self.optimizers = optimizers_dict

    def optimize_all_params(self):
        for k, v in self.optimizers.items():
            v.step()

    def optimize_params(self, model_name):
        self.optimizers[model_name].step()

    def reset_optimizer(self, model_name):
        self.optimizers[model_name].zero_grad()

    def set_running_metric(self):
        running_metric = runningScore(n_classes=self.num_classes)
        return running_metric

    def save_testing_images_results(self, save_dir, epoch_iter, max_slices=10, file_name='Seg_plots.png'):
        gts = self.cur_eval_gts
        predicts = self.cur_eval_predicts
        images = self.cur_eval_images
        save_testing_images_results(
            images, gts, predicts, save_dir, epoch_iter, max_slices, file_name)


if __name__ == '__main__':

    solver = AdvancedTripletReconSegmentationModel(
        network_type='FCN_16_standard', num_classes=4, n_iter=3, use_gpu=True)
    model = solver.model
    images = torch.randn(2, 1, 224, 224).to('cuda')
    pred = solver.predict(images)
    # print ('output',pred.size())
    # print ('latent',solver.latent_code)
