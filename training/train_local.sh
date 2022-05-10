export nnUNet_raw_data_base="/home/lhboulogne/Documents/phd/data/dummy_data/raw"
export nnUNet_preprocessed="/home/lhboulogne/Documents/phd/data/dummy_data/preprocessed"
export RESULTS_FOLDER="/home/lhboulogne/Documents/phd/data/dummy_data/trained_models"

python3 train.py \
 3d_fullres ClassifierTrainer 1 all