export nnUNet_raw_data_base="/mnt/internal_storage/sdc2/luuk/stoic/public_train_for_uc/universal_classifier_format/raw"
export nnUNet_preprocessed="/mnt/internal_storage/sdc2/luuk/stoic/public_train_for_uc/universal_classifier_format/preprocessed"
export RESULTS_FOLDER="/mnt/internal_storage/sdc2/luuk/stoic/public_train_for_uc/universal_classifier_format/trained_models"

python3 train.py \
 3d_fullres ClassifierTrainer 1 all --use_compressed_data