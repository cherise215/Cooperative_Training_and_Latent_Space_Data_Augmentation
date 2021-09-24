# cval_id = 0 
# CUDA_VISIBLE_DEVICES=0 python medseg/train_adv_supervised_segmentation_triplet.py --json_config_path './config/ACDC/standard_training.json' --log --seed 40
# CUDA_VISIBLE_DEVICES=1 python medseg/train_adv_supervised_segmentation_triplet.py --json_config_path './config/ACDC/cooperative_training.json' --log --seed 40 --resume_pkl_path "/vol/biomedic3/cc215/Project/MedSeg/saved/train_ACDC_10_n_cls_4/cooperative_training/0/model/interrupted/checkpoints/FCN_16_standard.pkl"
CUDA_VISIBLE_DEVICES=1 python medseg/train_adv_supervised_segmentation_triplet.py --json_config_path './config/ACDC/cooperative_training.json' --log --seed 40
# cval_id = 1 
# CUDA_VISIBLE_DEVICES=0 python medseg/train_adv_supervised_segmentation_triplet.py --json_config_path './config/standard_training.json' --log --cval 1 --seed 20
# CUDA_VISIBLE_DEVICES=1 python medseg/train_adv_supervised_segmentation_triplet.py --json_config_path './config/cooperative_training.json' --log --cval 1 --seed 20

# cval_id = 2
# CUDA_VISIBLE_DEVICES=0 python medseg/train_adv_supervised_segmentation_triplet.py --json_config_path './config/standard_training.json' --log --cval 2 --seed 10
# CUDA_VISIBLE_DEVICES=1 python medseg/train_adv_supervised_segmentation_triplet.py --json_config_path './config/cooperative_training.json' --log --cval 2 --seed 10

