'''
train a model on various test datasets
'''
import os
from os.path import join
import pandas as pd
from scipy import stats


from medseg.models.advanced_triplet_recon_segmentation_model import AdvancedTripletReconSegmentationModel
from medseg.test_basic_segmentation_solver import TestSegmentationNetwork

from medseg.dataset_loader.cardiac_ACDC_dataset import CardiacACDCDataset
from medseg.dataset_loader.cardiac_MM_dataset import Cardiac_MM_Dataset
from medseg.dataset_loader.transform import Transformations
from medseg.dataset_loader.base_segmentation_dataset import ConcatDataSet
from medseg.common_utils.basic_operations import check_dir

## change your data root dir here!
PROJECT_DATA_ROOT_DIR = '/vol/biomedic3/cc215/data/MICCAI2021_multi_domain_robustness_datasets'

## test data cropping
pad_size = [224, 224, 1]
crop_size = [192, 192, 1]
IDX2CLASS_DICT = {
        0: 'BG',
        1: 'LV',
        2: 'MYO',
        3: 'RV',
    }
def get_testset(test_dataset_name, frames=['ED', 'ES']):
    data_aug_policy_name = 'no_aug'
    tr = Transformations(data_aug_policy_name=data_aug_policy_name, pad_size=pad_size,
                         crop_size=crop_size).get_transformation()

    formalized_label_dict = IDX2CLASS_DICT
    right_ventricle_seg = False
    testset_list = []
    for frame in frames:
        IMAGE_FORMAT_NAME = '{p_id}/' + frame + '_img.nii.gz'
        LABEL_FORMAT_NAME = '{p_id}/' + frame + '_seg.nii.gz'
        if test_dataset_name == 'ACDC':
            root_dir = f'{PROJECT_DATA_ROOT_DIR}/ACDC'
            test_dataset = CardiacACDCDataset(root_dir=root_dir, transform=tr['validate'], idx2cls_dict=IDX2CLASS_DICT, num_classes=4,
                                              data_setting_name='10', formalized_label_dict=formalized_label_dict,
                                              frame=frame, split='test', myocardium_seg=False,
                                              image_format_name=IMAGE_FORMAT_NAME,
                                              label_format_name=LABEL_FORMAT_NAME,
                                              right_ventricle_seg=right_ventricle_seg,
                                              new_spacing=None)
        elif test_dataset_name == 'MM':
            root_dir = f'{PROJECT_DATA_ROOT_DIR}/MM'
            test_dataset = Cardiac_MM_Dataset(root_dir=root_dir,
                                              transform=tr['validate'], num_classes=4, formalized_label_dict=formalized_label_dict,
                                              idx2cls_dict=IDX2CLASS_DICT,
                                              image_format_name=IMAGE_FORMAT_NAME,
                                              label_format_name=LABEL_FORMAT_NAME,
                                              new_spacing=None)

        elif test_dataset_name in ['RandomGhosting', 'RandomBias', 'RandomSpike', 'RandomMotion']:
            root_folder = f'{PROJECT_DATA_ROOT_DIR}/ACDC-C/{test_dataset_name}'
            test_dataset = Cardiac_MM_Dataset(root_dir=root_folder,
                                              dataset_name=test_dataset_name,
                                              transform=tr['validate'], num_classes=4, formalized_label_dict=formalized_label_dict,
                                              idx2cls_dict=IDX2CLASS_DICT,
                                              image_format_name=IMAGE_FORMAT_NAME,
                                              label_format_name=LABEL_FORMAT_NAME,
                                              new_spacing=None)

        else:
            raise NotImplementedError
        testset_list.append(test_dataset)
    if len(testset_list) >= 2:
        concatdataset = ConcatDataSet(testset_list)
    else:
        concatdataset = testset_list[0]
    return concatdataset


def evaluate(method_name, segmentation_model, test_dataset_name, frames=['ED', 'ES'], metrics_list=['Dice'],
             save_report_dir=None,
             save_predict=False, save_soft_prediction=False, foreground_only=False):
    n_iter = segmentation_model.n_iter
    # evaluation settings
    save_path = checkpoint_dir.replace(
        'checkpoints', f'report/{test_dataset_name}')
    check_dir(save_path, create=True)

    summary_report_file_name = 'iter_{}_summary.csv'.format(n_iter)
    detailed_report_file_name = 'iter_{}_detailed.csv'.format(n_iter)
    test_dataset = get_testset(test_dataset_name, frames=frames)
    tester = TestSegmentationNetwork(test_dataset=test_dataset,
                                     crop_size=[192, 192, 1], segmentation_model=segmentation_model, use_gpu=True,
                                     save_path=save_path, summary_report_file_name=summary_report_file_name,
                                     detailed_report_file_name=detailed_report_file_name, patient_wise=True, metrics_list=metrics_list,
                                     foreground_only=foreground_only,
                                     save_prediction=save_predict, save_soft_prediction=save_soft_prediction)

    tester.run()

    print('<Summary> {} on dataset {} across {}'.format(
        method_name, test_dataset_name, str(frames)))
    print(tester.df.describe())
    # save each method's result summary/details on each test dataset
    tester.df.describe().to_csv(join(save_path + f'/{test_dataset_name}' + '{}_iter_{}_summary.csv'.format(
        str(frames), str(n_iter))))
    tester.df.to_csv(join(save_path + f'/{test_dataset_name}' + '{}_iter_{}_detailed.csv'.format(
        str(frames), str(n_iter))))

    means = [round(v, 4) for k, v in tester.df.mean(axis=0).items()]
    stds = [round(v, 4) for k, v in tester.df.std(axis=0).items()]
    return means, stds, tester.df


if __name__ == '__main__':
    use_gpu = True
    # model config
    num_classes = 4
    network_type = 'FCN_16_standard'
    n_iter = 2  # 1 for FTN's prediction, 2 for FTN+STN's refinements
    cval_id_list = [0, 1, 2]

    test_dataset_name_list = [
        'ACDC', 'RandomBias', 'RandomSpike','RandomGhosting','RandomMotion', 'MM']
    frames=['ED', 'ES']
    for cval_id in cval_id_list:
        # change your path here
        segmentor_resume_dir_dict = {
            'standard_training': f'./saved/train_ACDC_10_n_cls_4/ACDC/standard_training_test/{cval_id}/model/best/checkpoints',
            'cooperative_training': f'./saved/train_ACDC_10_n_cls_4/ACDC/cooperative_training/{cval_id}/model/best/checkpoints',
         
        }

        # load model
        model_dict = {}
        for method, checkpoint_dir in segmentor_resume_dir_dict.items():
            if not os.path.exists(checkpoint_dir):
                print(f'{method}:{checkpoint_dir} not found. ')
                continue
            model_dict[method] = AdvancedTripletReconSegmentationModel(network_type=network_type, decoder_dropout=None,
                                                                       checkpoint_dir=checkpoint_dir,
                                                                       num_classes=num_classes, n_iter=n_iter, use_gpu=True)
        df_dict = {}
        for test_dataset_name in test_dataset_name_list:
            result_summary = []

            for method_name, model in model_dict.items():
                save_report_dir = join(
                    segmentor_resume_dir_dict[method_name], 'report')
                check_dir(save_report_dir, create=True)
                means, stds, concatenated_df = evaluate(
                    segmentation_model=model, test_dataset_name=test_dataset_name, frames=frames, method_name=method_name, save_report_dir=save_report_dir)
                result_summary.append(
                    [test_dataset_name, method_name, means, stds])
                df_dict[method_name] = concatenated_df
            aggregated_df = pd.DataFrame(data=result_summary, columns=[
                'dataset', 'method', 'Dice mean', 'Dice std'])
            print(aggregated_df)
