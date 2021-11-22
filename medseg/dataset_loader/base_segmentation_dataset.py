# Created by cc215 at 11/12/19
# Enter feature description here
# Enter scenario name here
# Enter steps here

import torch.utils.data as data
import torch
from torch.utils.data import Dataset

import numpy as np
from medseg.common_utils.basic_operations import switch_kv_in_dict


class BaseSegDataset(Dataset):
    def __init__(self, dataset_name, transform, image_size, label_size, idx2cls_dict=None, num_classes=2,
                 use_cache=False, formalized_label_dict=None, keep_orig_image_label_pair=False):
        '''

        :param dataset_name:
        :param transform:
        :param image_size:
        :param label_size:
        :param idx2cls_dict:
        :param num_classes:
        :param use_cache:
        :param formalized_label_dict:
        :param keep_orig_image_label_pair:  if true, then each time will produce image-label pairs before and/after data augmentation
        '''
        super(BaseSegDataset).__init__()
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.image_size = image_size
        self.label_size = label_size
        self.transform = transform
        self.idx2cls_dict = idx2cls_dict
        if idx2cls_dict is None:
            self.idx2cls_dict = {}
            for i in range(num_classes):
                self.idx2cls_dict[i] = str(i)
        self.formalized_label_dict = self.idx2cls_dict if formalized_label_dict is None else formalized_label_dict
        self.use_cache = use_cache
        self.cache_dict = {}
        self.index = 0
        self.voxelspacing = [1., 1., 1.]
        self.keep_orig_image_label_pair = keep_orig_image_label_pair
        self.patient_number = 0

    def get_id(self):
        '''
        return the current id
        :return:
        '''
        return self.index

    def get_voxel_spacing(self):
        '''
        return the current id
        :return:
        '''
        return self.voxelspacing

    def set_id(self, index):
        '''
        set the current id with semantic information (e.g. patient id)
        :return:
        '''
        return self.index

    def __getitem__(self, index):
        self.set_id(index)
        if self.use_cache:
            # load data from RAM to save IO time
            if index in self.cache_dict.keys():
                data_dict = self.cache_dict[index]
            else:
                data_dict = self.load_data(index)
                self.cache_dict[index] = data_dict

        else:
            data_dict = self.load_data(index)

        data_dict = self.preprocess_data_to_tensors(
            data_dict['image'], data_dict['label'])

        return data_dict

    def load_data(self, index):
        '''
        generate dummy data for sanity check, need to reimplement it in child classes
        :param index, the iterator index
        :return:
        image: nd array H*W*CH
        label: nd array H*W
        '''
        image = np.random.rand(*self.image_size)
        label = np.random.rand(*self.label_size)
        label[label > 0.5] = 1
        label[label <= 0.5] = 0
        label = np.uint8(label)
        return {'image': image,
                'label': label
                }

    def __len__(self):
        return 30

    def preprocess_data_to_tensors(self, image, label):
        '''
        use predefined data preprocessing pipeline to transform data
        :param image: ndarray: H*W*CH
        :param label: ndarray: H*W
        :return:
        dict{
        'image': torch tensor: ch*H*W
        'label': torch tensor: H*W
        }
        '''
        assert len(image.shape) == 3 and len(
            label.shape) <= 3, 'input image and label dim should be 3 and 2 respectively, but got {} and {}'.format(
            len(image.shape),
            len(label.shape))
        # safe check, the channel should be in the last dimension
        assert image.shape[2] < image.shape[1] and image.shape[2] < image.shape[
            0], ' input image should be of the HWC format'

        # reassign label:
        new_labels = self.formulate_labels(label)

        new_labels = np.uint8(new_labels)
        orig_image = image
        orig_label = new_labels.copy()

        # expand label to be 3D for transformation
        if_slice_data = True if len(label.shape) == 2 else False
        if if_slice_data:
            new_labels = new_labels[:, :, np.newaxis]
        new_labels = np.uint8(new_labels)
        if image.shape[2] > 1:  # RGB channel
            new_labels = np.repeat(new_labels, axis=2, repeats=image.shape[2])
        transformed_image, transformed_label = self.transform(
            image, new_labels)
        if if_slice_data:
            transformed_label = transformed_label[0, :, :]

        result_dict = {
            'image': transformed_image,
            'label': transformed_label
        }
        if self.keep_orig_image_label_pair:
            h, w = orig_image.shape[0], orig_image.shape[1]

            new_h, new_w = transformed_image.size(1), transformed_image.size(2)
            # pad images AND labels
            h_s = (h - new_h) // 2
            w_s = (w - new_w) // 2
            if h < new_h:
                pad_result = np.zeros(
                    (new_h, orig_image.shape[1], orig_image.shape[2]), dtype=orig_image.dtype)
                pad_result[-h_s:-h_s + h] = orig_image
                orig_image = pad_result
                pad_result = np.zeros(
                    (new_h, orig_image.shape[1]), dtype=orig_label.dtype)
                pad_result[-h_s:-h_s + h] = orig_label
                orig_label = pad_result
            if w < new_w:
                pad_result = np.zeros(
                    (orig_image.shape[0], new_w, orig_image.shape[2]), dtype=orig_image.dtype)
                pad_result[:, -w_s:-w_s + w] = orig_image
                orig_image = pad_result
                pad_result = np.zeros(
                    (orig_image.shape[0], new_w), dtype=orig_label.dtype)
                pad_result[:, -w_s:-w_s + w] = orig_label
                orig_label = pad_result

            h, w = orig_image.shape[0], orig_image.shape[1]
            h_s = (h - new_h) // 2
            w_s = (w - new_w) // 2
            assert h_s >= 0 and w_s >= 0, 'crop image should be smaller than original image'
            if h_s > 0 or w_s > 0:
                orig_image = orig_image[h_s:h_s + new_h, w_s:w_s + new_w]
                orig_label = orig_label[h_s:h_s + new_h, w_s:w_s + new_w]
            orig_image_tensor = torch.from_numpy(
                orig_image).float().permute(2, 0, 1)  # ch*H*W
            orig_label_tensor = torch.from_numpy(orig_label).long()
            result_dict['origin_image'] = orig_image_tensor
            result_dict['origin_label'] = orig_label_tensor

        return result_dict

    def formulate_labels(self, label, foreground_only=False):
        origin_labels = label.copy()
        if foreground_only:
            origin_labels[origin_labels > 0] = 1
            return origin_labels
        old_cls_to_idx_dict = switch_kv_in_dict(self.idx2cls_dict)
        new_cls_to_idx_dict = switch_kv_in_dict(self.formalized_label_dict)
        new_labels = np.zeros_like(label, dtype=np.uint8)
        for key in new_cls_to_idx_dict.keys():
            old_label_value = old_cls_to_idx_dict[key]
            new_label_value = new_cls_to_idx_dict[key]
            new_labels[origin_labels == old_label_value] = new_label_value
        return new_labels

    def get_patient_data_for_testing(self, pid_index, crop_size=None, normalize_2D=False):
        '''
        image
        :param pad_size:[H',W']
        :param crop_size: [H',W']
        :return:
        torch tensor data N*1*H'*W'
        torch tensor data: N*H'*W'
        '''
        raise NotImplementedError

    def get_info(self):
        print('{} contains {} images with size of {}, num_classes: {} '.format(self.dataset_name, str(self.datasize),
                                                                               str(self.image_size),
                                                                               str(self.num_classes)))

    def save_cache(a_dict, cache_path):
        '''
        given a dict, save it to disk pth
        '''
        pass

    def load_cache(a_dict, cache_path):
        pass


class CombinedDataSet(data.Dataset):
    """
    source_dataset and augmented_source_dataset must be aligned
    """

    def __init__(self, source_dataset, target_dataset):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def __getitem__(self, index):
        source_index = index % len(self.source_dataset)
        target_index = (index + np.random.randint(0,
                                                  len(self.target_dataset) - 1)) % len(self.target_dataset)

        return self.source_dataset[source_index], self.target_dataset[target_index]

    def __len__(self):
        return min(len(self.source_dataset), len(self.target_dataset))


class ConcatDataSet(data.Dataset):
    """
    concat a list of datasets together
    """

    def __init__(self, dataset_list):
        self.dataset_list = dataset_list
        a_sum = 0
        self.patient_number = 0
        self.formalized_label_dict = self.dataset_list[0].formalized_label_dict
        self.pid2datasetid = {}
        self.slice2datasetid = {}
        for dataset_id, dset in enumerate(self.dataset_list):
            for id in range(self.patient_number, self.patient_number + dset.patient_number):
                self.pid2datasetid[id] = dataset_id
            for sid in range(a_sum, a_sum + len(dset)):
                self.slice2datasetid[sid] = dataset_id
            a_sum += len(dset)
            self.patient_number += dset.patient_number
        self.datasize = a_sum
        print(
            f'total patient number: {self.patient_number}, 2D slice number:{self.datasize}')

    def __getitem__(self, index):
        dataset_id = self.slice2datasetid[index]
        if dataset_id >= 1:
            start_index = 0
            for ds in self.dataset_list[:dataset_id]:
                start_index += len(ds)
            index = index - start_index
        # print(f'index {index} dataset id {dataset_id}')
        self.cur_dataset = self.dataset_list[dataset_id]
        return self.cur_dataset[index]

    def __len__(self):
        return self.datasize

    def get_id(self):
        '''
        return the current patient id
        :return:
        '''

        return self.cur_dataset.get_id()

    def get_voxel_spacing(self):
        return self.cur_dataset.get_voxel_spacing()

    def get_patient_data_for_testing(self, pid_index, crop_size=None, normalize_2D=False):
        # if normalize_2D:
        #     print('call 2D normalize')
        self.p_id = pid_index
        dataset_id = self.pid2datasetid[pid_index]
        self.cur_dataset = self.dataset_list[dataset_id]
        index = pid_index % self.cur_dataset.patient_number
        data_pack = self.cur_dataset.get_patient_data_for_testing(
            index, crop_size, normalize_2D)
        return data_pack


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from medseg.dataset_loader.transform import Transformations  #
    from torch.utils.data import DataLoader

    image_size = (5, 5, 1)
    label_size = (5, 5)
    crop_size = (5, 5, 1)
    # class_dict={
    #   0: 'BG',  1: 'FG'}
    tr = Transformations(data_aug_policy_name='affine',
                         crop_size=crop_size).get_transformation()
    dataset = BaseSegDataset(dataset_name='dummy', image_size=image_size, label_size=label_size, transform=tr['train'],
                             use_cache=True)
    dataset_2 = BaseSegDataset(dataset_name='dummy', image_size=image_size, label_size=label_size, transform=tr['train'],
                               use_cache=True)
    combined_train_loader = CombinedDataSet(
        source_dataset=dataset, target_dataset=dataset_2)
    train_loader = DataLoader(dataset=combined_train_loader,
                              num_workers=0, batch_size=1, shuffle=True, drop_last=True)

    for i, item in enumerate(train_loader):
        source_input, target_input = item
        # print (source_input)
        img = source_input['image']
        label = target_input['image']
        print(img.numpy().shape)
        print(label.numpy().shape)
        plt.subplot(121)
        plt.imshow(img.numpy()[0, 0])
        plt.subplot(122)
        plt.imshow(label.numpy()[0, 0])
        plt.colorbar()
        plt.show()
        break
