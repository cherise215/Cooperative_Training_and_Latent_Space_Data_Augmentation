# Created by cc215 at 27/1/20
# this dataset is available at '/vol/medic01/users/cc215/data/MedicalDecathlon/Task05_Prostate/preprocessed'
# from Medical Decathlon challenge dataset, we use T2 as input for the task.
# Note: All images have been preprocessed (resampled to the 0.625 x 0.625 x 3.6 mm, the median value of the voxel spacings), following the preprocessing steps in
# "nn-Unet"
# contains 32 patients in total
# Data structure:
# each patient has a nrrd file
# path:
### image : root_dir/ES/{patient_id}/t2_img.nrrd
### label : root_dir/ES/{patient_id}/label.nrrd

import logging
import numpy as np
import os
import SimpleITK as sitk
import torch
from sklearn.model_selection import train_test_split

from medseg.dataset_loader.base_segmentation_dataset import BaseSegDataset

DATASET_NAME = 'Prostate'
IDX2CLASS_DICT = {
    0: 'BG',
    1: 'PZ',
    2: 'CZ',
}
IMAGE_FORMAT_NAME = '{p_id}/t2_img.nrrd'
LABEL_FORMAT_NAME = '{p_id}/label.nrrd'
IMAGE_SIZE = (320, 320, 1)
LABEL_SIZE = (320, 320)


class ProstateDataset(BaseSegDataset):
    def __init__(self,
                 transform, dataset_name=DATASET_NAME,
                 root_dir='/vol/biomedic3/cc215/data/MedicalDecathlon/Task05_Prostate/preprocessed',
                 num_classes=3,
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
                 lazy_load=False,
                 binary_segmentation=False,
                 smooth_label=False,
                 debug=False,

                 ):
        # predefined variables
        # initialization
        self.debug = debug
        self.data_setting_name = data_setting_name
        self.split = split  # can be validation or test or all
        self.cval = cval
        super(ProstateDataset, self).__init__(dataset_name=dataset_name, transform=transform, num_classes=num_classes,
                                              image_size=image_size, label_size=label_size, idx2cls_dict=idx2cls_dict,
                                              use_cache=use_cache, formalized_label_dict=formalized_label_dict, keep_orig_image_label_pair=keep_orig_image_label_pair)
        # specific paramters in this dataset
        self.root_dir = root_dir
        self.image_format_name = image_format_name
        self.label_format_name = label_format_name
        self.binary_segmentation = binary_segmentation
        self.smooth_label = smooth_label
        if lazy_load is True:
            self.datasize = 0
            self.patient_id_list = []
            self.index2pid_dict = {}
            self.index2slice_dict = {}
        else:
            self.datasize, self.patient_id_list, self.index2pid_dict, self.index2slice_dict = self.scan_dataset()

        self.temp_data_dict = None  # temporary data during loading
        self.p_id = 0  # current pid
        self.patient_number = len(self.patient_id_list)
        self.slice_id = 0
        self.index = 0  # index for selecting which slices
        self.dataset_name = DATASET_NAME + '_{}_{}'.format(str(data_setting_name), split)
        if self.split == 'train':
            self.dataset_name += str(cval)

        print('load {},  containing {}, found {} slices'.format(
            self.dataset_name, len(self.patient_id_list), self.datasize))
        self.voxelspacing = [0.625, 0.625, 3.6]

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
        assert len(self.patient_id_list) > 0, "no data found in the disk at {}".format(self.root_dir)
        index = index % self.datasize
        patient_id, slice_id = self.find_pid_slice_id(index)

        if self.debug:
            print(patient_id)
        sitkImage, sitkLabel = self.load_patientImage_from_nrrd(patient_id)

        if self.smooth_label:
            pass

        image = sitk.GetArrayFromImage(sitkImage)[slice_id]
        label = sitk.GetArrayFromImage(sitkLabel)[slice_id]
        if self.binary_segmentation:
            label[label > 0] = 1
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

    def load_patientImage_from_nrrd(self, patient_id):
        img_name = self.image_format_name.format(p_id=patient_id)
        label_name = self.label_format_name.format(p_id=patient_id)
        img_path = os.path.join(self.root_dir, img_name)
        label_path = os.path.join(self.root_dir, label_name)
        # load data
        sitkImage = sitk.ReadImage(img_path)
        sitkImage = sitk.Cast(sitkImage, sitk.sitkFloat32)
        sitkLabel = sitk.ReadImage(label_path)
        sitkLabel = sitk.Cast(sitkLabel, sitk.sitkInt16)
        return sitkImage, sitkLabel

    def scan_dataset(self):
        '''
        given the data setting names and split, cross validation id
        :return: dataset size, a list of pids for training/testing/validation, and a dict for retrieving patient id and slice id.
        '''

        patient_id_list = self.get_pid_list(identifier=self.data_setting_name, cval=self.cval)[self.split]
        # print ('{} set has {} patients'.format(self.split,len(patient_id_list)))
        index2pid_dict = {}
        index2slice_dict = {}
        cur_ind = 0
        for pid in patient_id_list:
            img_path = os.path.join(self.root_dir, self.image_format_name.format(p_id=pid))
            ndarray = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
            num_slices = ndarray.shape[0]
            for cnt in range(num_slices):
                index2pid_dict[cur_ind] = pid
                index2slice_dict[cur_ind] = cnt
                cur_ind += 1
        datasize = cur_ind
        return datasize, patient_id_list, index2pid_dict, index2slice_dict

    def get_pid_list(self, identifier, cval):
        assert cval >= 1, 'cval must be >1'
        all_p_id_list = sorted(os.listdir(self.root_dir))
        # leave the test set out
        test_ids = ['patient_37',
                    'patient_35',
                    'patient_40',
                    'patient_43',
                    'patient_13',
                    'patient_29',
                    'patient_04', ]
        train_val_ids = [pid for pid in all_p_id_list if pid not in test_ids]
        # train_val_ids, test_ids = train_test_split(all_p_id_list,test_size=0.2,random_state=0)
        train_ids, val_ids = train_test_split(train_val_ids, test_size=0.1, random_state=cval - 1)
        # test_ids = [ tid for tid in test_ids if  '43' not in tid ] ## remove bad labels
        # print (test_ids)
        size = len(train_ids)
        # print ('train size',size)
        labelled_ids = train_ids[:(size // 2)]
        unlabelled_ids = train_ids[(size // 2):]
        if identifier == 'all':
            # use all training data as labelled data
            labelled_ids_split = train_ids
        elif identifier == 'three_shot':
            labelled_ids_split, _ = train_test_split(labelled_ids, train_size=3, random_state=cval)
        elif identifier == 'three_shot_upperbound':
            labelled_ids_split, _ = train_test_split(labelled_ids, train_size=3, random_state=cval)
            labelled_ids_split = labelled_ids_split + unlabelled_ids
        elif identifier == 'full':
            labelled_ids_split = labelled_ids

        elif isinstance(float(identifier), float):
            identifier = float(identifier)
            if 0 < identifier < 1:
                labelled_ids_split, _ = train_test_split(labelled_ids, train_size=identifier, random_state=cval)
            elif identifier > 1:
                identifier = int(identifier)
                if 0 < identifier < len(labelled_ids):
                    labelled_ids_split, _ = train_test_split(labelled_ids, train_size=identifier, random_state=cval)
                elif abs(identifier + 1) < 1e-6:
                    labelled_ids_split = labelled_ids
                else:
                    raise ValueError
            else:
                raise NotImplementedError
        else:
            print('use all training subjects')
            labelled_ids_split = labelled_ids
        return {
            'name': str(identifier) + '_cv_' + str(cval),
            'train': labelled_ids_split,
            'validate': val_ids,
            'test': test_ids,
            'test+unlabelled': test_ids + unlabelled_ids,
            'unlabelled': unlabelled_ids,
        }

    def get_patient_data_for_testing(self, pid_index, crop_size=None):
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
        self.p_id = self.patient_id_list[pid_index]
        sitkImage, sitkLabel = self.load_patientImage_from_nrrd(self.p_id)

        image = sitk.GetArrayFromImage(sitkImage)
        label = sitk.GetArrayFromImage(sitkLabel)
        if self.binary_segmentation:
            label[label > 0] = 1
        if crop_size is not None:
            h, w = image.shape[1], image.shape[2]
            assert crop_size[0] <= h and crop_size[1] <= w, 'crop sizes must be smaller than image sizes'
            h_s = (h - crop_size[0]) // 2
            w_s = (w - crop_size[1]) // 2
            image = image[:, h_s:h_s + crop_size[0], w_s:w_s + crop_size[1]]
            label = label[:, h_s:h_s + crop_size[0], w_s:w_s + crop_size[1]]
        label = self.formulate_labels(label)

        # 0-1 scaling
        logging.info('min max rescaling')
        eps = 1e-20
        if image.shape[0] == 1:
            min_val, max_val = np.percentile(image, (0, 100))
            image[image > max_val] = max_val
            image[image < min_val] = min_val
            if not abs(max_val - min_val) < 1e-12:
                image = (image - min_val) / (max_val - min_val)

        else:
            for i in range(image.shape[0]):
                a_slice = image[i]
                min_val, max_val = np.percentile(a_slice, (0, 100))
                a_slice[a_slice > max_val] = max_val
                a_slice[a_slice < min_val] = min_val
                a_slice = (a_slice - min_val) / (max_val - min_val + eps)
                image[i] = a_slice

        image_tensor = torch.from_numpy(image[:, np.newaxis, :, :]).float()
        label_tensor = torch.from_numpy(label[:, :, :]).long()
        dict = {
            'image': image_tensor,
            'label': label_tensor
        }
        return dict

    def __len__(self):
        return self.datasize

    @staticmethod
    def get_all_image_array_from_datastet(dataset):
        image_arrays = np.array([data['image'].numpy().reshape(1, -1).squeeze() for i, data in enumerate(dataset)])
        return image_arrays

    @staticmethod
    def get_mean_image(dataset):
        image_arrays = np.array([data['image'].numpy().reshape(1, -1).squeeze() for i, data in enumerate(dataset)])
        return np.mean(image_arrays, axis=0)

    def get_id(self):
        '''
        return the current patient id
        :return:
        '''
        return self.p_id


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from medseg.dataset_loader.transform import Transformations  #
    from torch.utils.data import DataLoader

    image_size = [288, 288, 1]
    label_size = [288, 288]
    crop_size = [288, 288, 1]
    tr = Transformations(data_aug_policy_name='Prostate_affine_elastic_intensity',
                         pad_size=image_size, crop_size=crop_size).get_transformation()
    dataset = ProstateDataset(split='train', data_setting_name=0.3, transform=tr['train'], binary_segmentation=True,
                              num_classes=3)
    train_loader = DataLoader(dataset=dataset, num_workers=0, batch_size=1, shuffle=False, drop_last=True)
    print(len(train_loader))
    for i, item in enumerate(train_loader):
        img = item['origin_image']
        label = item['origin_label']
        print(i, dataset.get_id())
        print(img.numpy().shape)
        # print(label.numpy().shape)
        # plt.subplot(141)
        # plt.imshow(img.numpy()[0], cmap='gray')
        # plt.subplot(142)
        # plt.imshow(label.numpy())
        # plt.colorbar()

        # img = item['image']
        # label = item['label']
        # print(img.numpy().shape)
        # print(label.numpy().shape)
        # plt.subplot(143)
        # plt.imshow(img.numpy()[0], cmap='gray')
        # plt.subplot(144)
        # plt.imshow(label.numpy())
        # plt.colorbar()
        # plt.show()
        # if i >= 10:
        #     break
