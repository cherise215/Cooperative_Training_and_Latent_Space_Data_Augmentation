# Created by cc215 at 27/1/20
# this dataset is available at '/vol/medic01/users/cc215/Dropbox/projects/DeformADA/Data/ACDC'
# from ACDC challenge dataset, containing ED/ES whole stack (3D).
# Note: All images have been preprocessed and cropped to 224 by 224, following the preprocessing steps in
# "Semi-Supervised and Task-Driven Data Augmentation"
# https://arxiv.org/abs/1902.05396
# contains 100 patients in total
# Data structure:
# each patient has a nrrd file for each phase ED/ES.
# path:
# ED images : root_dir/ED/{patient_id}_img.nrrd,root_dir/ED/{patient_id}_seg.nrrd, all [n_slices*224*224]s
# ES images : root_dir/ES/{patient_id}_img.nrrd,root_dir/ES/{patient_id}_seg.nrrd, all [n_slices*224*224]s

import numpy as np
import os
import random
import SimpleITK as sitk
import torch

from medseg.dataset_loader.base_segmentation_dataset import BaseSegDataset
from medseg.dataset_loader.ACDC_few_shot_cv_settings import get_ACDC_split_policy
from medseg.common_utils.basic_operations import load_img_label_from_path, crop_or_pad, rescale_intensity
DATASET_NAME = 'ACDC'
IDX2CLASS_DICT = {
    0: 'BG',
    1: 'LV',
    2: 'MYO',
    3: 'RV',
}
IMAGE_FORMAT_NAME = '{p_id}/{frame}_img.nii.gz'
LABEL_FORMAT_NAME = '{p_id}/{frame}_seg.nii.gz'
IMAGE_SIZE = (224, 224, 1)
LABEL_SIZE = (224, 224)

# images are stored like
# image: root_dir/subset_name/{p_id}_img.nrrd
# label: root_dir/subset_name/{p_id}_seg.nrrd


class CardiacACDCDataset(BaseSegDataset):
    def __init__(self,
                 transform, dataset_name=DATASET_NAME,
                 root_dir='/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized',
                 frame='ES', num_classes=4,
                 debug=False,
                 image_size=IMAGE_SIZE,
                 label_size=LABEL_SIZE,
                 idx2cls_dict=IDX2CLASS_DICT,
                 use_cache=True,
                 data_setting_name='three_shot',
                 split='train',
                 cval=0,  # int, cross validation id
                 formalized_label_dict=None,
                 keep_orig_image_label_pair=True,
                 image_format_name=IMAGE_FORMAT_NAME,
                 label_format_name=LABEL_FORMAT_NAME,
                 myocardium_seg=False,
                 right_ventricle_seg=False,
                 new_spacing=[1.36719, 1.36719, -1], normalize=False
                 ):
        # predefined variables
        # initialization

        self.debug = debug
        self.data_setting_name = data_setting_name
        self.split = split  # can be validation or test or all
        self.cval = cval
        if myocardium_seg:
            formalized_label_dict = {0: 'BG', 1: 'MYO'}
        if right_ventricle_seg:
            formalized_label_dict = {0: 'BG', 1: 'RV'}
        super(CardiacACDCDataset, self).__init__(dataset_name=dataset_name, transform=transform, num_classes=num_classes,
                                                 image_size=image_size, label_size=label_size, idx2cls_dict=idx2cls_dict,
                                                 use_cache=use_cache, formalized_label_dict=formalized_label_dict, keep_orig_image_label_pair=keep_orig_image_label_pair)
        # specific paramters in this dataset
        self.root_dir = os.path.join(root_dir)
        self.frame = frame
        self.image_format_name = image_format_name
        self.label_format_name = label_format_name
        self.normalize = normalize  # normalize 3D data if data are raw. default: false
        self.new_spacing = new_spacing

        self.datasize, self.patient_id_list, self.index2pid_dict, self.index2slice_dict = self.scan_dataset(
            root_dir=self.root_dir, image_format_name=image_format_name, data_setting_name=data_setting_name, cval=cval, split=split)

        self.temp_data_dict = None  # temporary data during loading
        self.p_id = 0  # current pid
        self.patient_number = len(self.patient_id_list)
        self.slice_id = 0
        self.index = 0  # index for selecting which slices
        self.dataset_name = DATASET_NAME + \
            '_{}_{}_{}'.format(frame, data_setting_name, split)
        if self.split == 'train':
            self.dataset_name += str(cval)

        print('load {},  containing {}, found {} slices'.format(
            self.dataset_name + self.frame, len(self.patient_id_list), self.datasize))
        self.voxelspacing = [1.36719, 1.36719, -1]

        if self.new_spacing is not None:
            self.voxelspacing = self.new_spacing

        self.myocardium_seg = myocardium_seg
        self.right_ventricle_seg = right_ventricle_seg

    def find_pid_slice_id(self, index):
        '''
        given an index, find the patient id and slice id
        return the current id
        :return:
        '''
        self.p_id = self.index2pid_dict[index]
        self.slice_id = self.index2slice_dict[index]

        return self.p_id, self.slice_id

    def load_data(self, index):
        '''
        give a index to fetch a data package for one patient
        :return:
        data from a patient.
        class dict: {
        'image': ndarray,H*W*CH, CH =1, for gray images
        'label': ndaray, H*W
        '''
        assert len(self.patient_id_list) > 0, "no data found in the disk at {}".format(
            self.root_dir)

        patient_id, slice_id = self.find_pid_slice_id(index)

        if self.debug:
            print(patient_id)
        image_3d, label_3d, sitkImage, sitkLabel = self.load_patientImage_from_nrrd(
            patient_id, new_spacing=self.new_spacing, normalize=self.normalize)
        image = image_3d[slice_id]
        label = label_3d[slice_id]

        max_id = image_3d.shape[0]
        id_list = list(np.arange(max_id))
        # remove slice w.o objects
        while True:
            image = image_3d[slice_id]
            label = label_3d[slice_id]
            if abs(np.sum(label) - 0) > 1e-4:
                break
            else:
                id_list.remove(slice_id)
                random.shuffle(id_list)
                slice_id = id_list[0]

        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        if self.debug:
            print(image.shape)
            print(label.shape)

        cur_data_dict = {'image': image,
                         'label': label,
                         'pid': patient_id}
        self.temp_data_dict = cur_data_dict
        return cur_data_dict

    def load_patientImage_from_nrrd(self, patient_id, new_spacing=None, normalize=False):
        img_name = self.image_format_name.format(
            p_id=patient_id, frame=self.frame)
        label_name = self.label_format_name.format(
            p_id=patient_id, frame=self.frame)
        img_path = os.path.join(self.root_dir, img_name)
        label_path = os.path.join(self.root_dir, label_name)
        # load data
        img_arr, label_arr, sitkImage, sitkLabel = load_img_label_from_path(
            img_path, label_path, new_spacing=new_spacing, normalize=normalize)

        return img_arr, label_arr, sitkImage, sitkLabel

    def scan_dataset(self, root_dir, image_format_name, data_setting_name, cval, split):
        '''
        given the data setting names and split, cross validation id
        :return: dataset size, a list of pids for training/testing/validation, and a dict for retrieving patient id and slice id.
        '''

        patient_id_list = get_ACDC_split_policy(
            identifier=data_setting_name, cval=cval)[split]
        # print ('{} set has {} patients'.format(self.split,len(patient_id_list)))
        index2pid_dict = {}
        index2slice_dict = {}
        cur_ind = 0
        for pid in patient_id_list:
            img_path = os.path.join(
                root_dir, image_format_name.format(p_id=pid, frame=self.frame))

            if not os.path.exists(img_path):
                print (f'{img_path} not found')
                continue
            ndarray = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
            num_slices = ndarray.shape[0]
            for cnt in range(num_slices):
                index2pid_dict[cur_ind] = pid
                index2slice_dict[cur_ind] = cnt
                cur_ind += 1
        datasize = cur_ind
        return datasize, patient_id_list, index2pid_dict, index2slice_dict

    def get_patient_data_for_testing(self, pid_index, crop_size=None, normalize_2D=True):
        '''
        prepare test volumetric data
        :param pad_size:[H',W']
        :param crop_size: [H',W']
        :return:
        data dict:
        {'image':torch tensor data N*1*H'*W'
        'label': torch tensor data: N*H'*W'
        }
        '''
        # print('here')
        self.p_id = self.patient_id_list[pid_index]
        image, label, sitkImage, sitkLabel = self.load_patientImage_from_nrrd(
            self.p_id, new_spacing=self.new_spacing, normalize=self.normalize)
        if crop_size is not None:
            crop_size = crop_size.copy()
            image, label, h_s, w_s, h, w = crop_or_pad(
                image, crop_size, label=label)
        image_tensor = torch.from_numpy(image[:, np.newaxis, :, :]).float()
        label_tensor = torch.from_numpy(label[:, :, :]).long()
        image_tensor = image_tensor.contiguous()
        if normalize_2D:
            image_tensor = rescale_intensity(image_tensor, 0, 1)
        dict = {
            'image': image_tensor,
            'label': label_tensor
        }
        return dict

    def __len__(self):
        return self.datasize

    def get_id(self):
        '''
        return the current patient id
        :return:
        '''
        return self.p_id + "_" + self.frame


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from medseg.dataset_loader.transform import Transformations  #
    from torch.utils.data import DataLoader

    image_size = [224, 224, 1]
    label_size = [224, 224]
    crop_size = [192, 192, 1]
    tr = Transformations(data_aug_policy_name='ACDC_affine_elastic_intensity_v2',
                         pad_size=image_size, crop_size=crop_size).get_transformation()
    dataset = CardiacACDCDataset(data_setting_name='standard', split='validate',
                                 transform=tr['train'], num_classes=4, right_ventricle_seg=True,
                                 frame='ED',
                                 root_dir='/vol/biomedic3/cc215/data/MICCAI2021_multi_domain_robustness_datasets/ACDC')
    train_loader = DataLoader(
        dataset=dataset, num_workers=0, batch_size=1, shuffle=False, drop_last=True)

    for i, item in enumerate(dataset):
        img = item['origin_image']
        label = item['origin_label']
        print(img.numpy().shape)
        print(label.numpy().shape)
        plt.subplot(141)
        plt.imshow(img.numpy()[0], cmap='gray')
        plt.subplot(142)
        plt.imshow(label.numpy())
        plt.colorbar()

        img = item['image']
        label = item['label']
        print(img.numpy().shape)
        print(label.numpy().shape)
        plt.subplot(143)
        plt.imshow(img.numpy()[0], cmap='gray')
        plt.subplot(144)
        plt.imshow(label.numpy())
        plt.colorbar()
        plt.clf()
        plt.show()
        if i >= 10:
            break
