'''
train a model on various test datasets
'''
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


pad_size = [224, 224, 1]
crop_size = [192, 192, 1]
new_spacing = [1.36719, 1.36719, -1]


def get_testset(test_dataset_name, frames=['ED', 'ES']):
    data_aug_policy_name = 'no_aug'
    tr = Transformations(data_aug_policy_name=data_aug_policy_name, pad_size=pad_size,
                         crop_size=crop_size).get_transformation()
    IDX2CLASS_DICT = {
        0: 'BG',
        1: 'LV',
        2: 'MYO',
        3: 'RV',
    }
    formalized_label_dict = IDX2CLASS_DICT
    right_ventricle_seg = False
    testset_list = []
    for frame in frames:
        if test_dataset_name == 'ACDC':
            root_dir = '/vol/biomedic3/cc215/Project/DeformADA/Data/bias_corrected_and_normalized'
            test_dataset = CardiacACDCDataset(root_dir=root_dir, transform=tr['validate'], idx2cls_dict=IDX2CLASS_DICT, num_classes=4,
                                              data_setting_name='10', formalized_label_dict=formalized_label_dict,
                                              subset_name=frame, split='test', myocardium_seg=False,
                                              right_ventricle_seg=right_ventricle_seg,
                                              new_spacing=None, normalize=False)
        elif test_dataset_name == 'MM':
            root_dir = '/vol/biomedic3/cc215/data/cardiac_MMSeg_challenge/Training-corrected/Labeled'
            IMAGE_FORMAT_NAME = '{p_id}/' + frame + '_img.nii.gz'
            LABEL_FORMAT_NAME = '{p_id}/' + frame + '_seg.nii.gz'
            test_dataset = Cardiac_MM_Dataset(root_dir=root_dir,
                                              transform=tr['validate'], num_classes=4, formalized_label_dict=formalized_label_dict,
                                              idx2cls_dict=IDX2CLASS_DICT,
                                              image_format_name=IMAGE_FORMAT_NAME,
                                              label_format_name=LABEL_FORMAT_NAME,
                                              new_spacing=new_spacing, normalize=True)

        elif test_dataset_name in ['RandomGhosting', 'RandomBias', 'RandomSpike', 'RandomMotion']:
            root_folder = '/vol/biomedic3/cc215/data/ACDC/ACDC_artefacted/{}'.format(test_dataset_name)
            IMAGE_FORMAT_NAME = '{p_id}/' + frame + '_img.nrrd'
            LABEL_FORMAT_NAME = '{p_id}/' + frame + '_label.nrrd'
            test_dataset = Cardiac_MM_Dataset(root_dir=root_folder,
                                              dataset_name=test_dataset_name,
                                              transform=tr['validate'], num_classes=4, formalized_label_dict=formalized_label_dict,
                                              idx2cls_dict=IDX2CLASS_DICT,
                                              image_format_name=IMAGE_FORMAT_NAME,
                                              label_format_name=LABEL_FORMAT_NAME,
                                              new_spacing=None, normalize=False)

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
    save_path = join(checkpoint_dir, 'report')

    frame_int = 0
    summary_report_file_name = 'iter_{}_summary.csv'.format(n_iter)
    detailed_report_file_name = 'iter_{}_detailed.csv'.format(n_iter)
    test_dataset = get_testset(test_dataset_name, frames=frames)
    tester = TestSegmentationNetwork(test_dataset=test_dataset,
                                     crop_size=[192, 192, 1], segmentation_model=segmentation_model, use_gpu=True,
                                     save_path=save_report_dir + f'/{test_dataset_name}', summary_report_file_name=summary_report_file_name,
                                     detailed_report_file_name=detailed_report_file_name, patient_wise=True, metrics_list=metrics_list,
                                     foreground_only=foreground_only,
                                     save_prediction=save_predict, save_soft_prediction=save_soft_prediction)

    tester.run()

    print('<Summary> {} on dataset {} across {}'.format(method_name, test_dataset_name, str(frames)))
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
    cval_id = 2

    test_dataset_name_list = ['ACDC', 'MM', 'RandomGhosting', 'RandomBias', 'RandomSpike', 'RandomMotion']
    # change your path here
    segmentor_resume_dir_dict = {
        'standard': f'saved/train_ACDC_10_n_cls_4/standard_training/{cval_id}/model/best/checkpoints',
        'cooperative': f'saved/train_ACDC_10_n_cls_4/cooperative_training/{cval_id}/model/best/checkpoints'
    }

    # load model
    model_dict = {}
    for method, checkpoint_dir in segmentor_resume_dir_dict.items():
        model_dict[method] = AdvancedTripletReconSegmentationModel(network_type=network_type, decoder_dropout=None,
                                                                   checkpoint_dir=checkpoint_dir,
                                                                   num_classes=num_classes, n_iter=n_iter, use_gpu=True)
    df_dict = {}
    for test_dataset_name in test_dataset_name_list:
        result_summary = []

        for method_name, model in model_dict.items():
            save_report_dir = join(segmentor_resume_dir_dict[method_name], 'report')
            check_dir(save_report_dir, create=True)
            means, stds, concatenated_df = evaluate(
                segmentation_model=model, test_dataset_name=test_dataset_name, method_name=method_name, save_report_dir=save_report_dir)
            result_summary.append([test_dataset_name, method_name, means, stds])
            df_dict[method_name] = concatenated_df
        aggregated_df = pd.DataFrame(data=result_summary, columns=['dataset', 'method', 'Dice mean', 'Dice std'])
        print(aggregated_df)

        # # conduct student t test between two pandas dataframe and then compute the p value to vis the difference
        # reference_method = "standard"
        # reference_df = df_dict[reference_method]
        # for method_name, df in df_dict.items():
        #     if method_name != reference_method:
        #         p_value_dict = {}
        #         for aclass in ['LV_Dice', 'MYO_Dice', 'RV_Dice']:
        #             ttest, lv_pval = stats.ttest_rel(df[aclass], reference_df[aclass])
        #             p_value_dict[aclass] = '{0:.4f}'.format(lv_pval)
        #         print('--------------------------------P value------------------------')
        #         print(str(p_value_dict))
