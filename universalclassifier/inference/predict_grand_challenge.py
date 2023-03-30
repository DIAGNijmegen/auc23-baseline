import torch
from batchgenerators.utilities.file_and_folder_operations import *
from typing import Tuple, Union, List
from nnunet.paths import network_training_output_dir
from universalclassifier.paths import default_plans_identifier, default_trainer
from universalclassifier.training.model_restore import load_model_and_checkpoint_files


def predict_grand_challenge(model: str,
                            task_name: str,
                            ordered_image_files: List[str],
                            folds: Union[Tuple[int], List[int]],
                            roi_segmentation_file: str = None,
                            trainer_class_name: str = default_trainer,
                            plans_identifier: str = default_plans_identifier,
                            disable_mixed_precision: bool = True,
                            checkpoint_name: str = "model_final_checkpoint"):
    mixed_precision = not disable_mixed_precision

    if not task_name.startswith("Task"):
        raise RuntimeError("task_name should start with 'Task'")

    assert model in ["3d_fullres"], "-m must be 3d_fullres"

    if isinstance(folds, list):
        if folds[0] == 'all' and len(folds) == 1:
            pass
        else:
            folds = [int(i) for i in folds]
    elif folds == "None":
        folds = None
    else:
        raise ValueError("Unexpected value for argument folds")

    model_folder_name = join(network_training_output_dir, model, task_name, trainer_class_name + "__" +
                             plans_identifier)
    print("using model stored in ", model_folder_name)
    assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name

    expected_num_modalities = load_pickle(join(model_folder_name, "plans.pkl"))['num_modalities']
    assert len(ordered_image_files) == expected_num_modalities, \
        f"Expected {expected_num_modalities} input modalities (excluding the optional roi segmentation), but len(ordered_image_files)=={len(ordered_image_files)}"

    print("emptying cuda cache")
    torch.cuda.empty_cache()

    print("loading parameters for folds,", folds)
    trainer, params = load_model_and_checkpoint_files(model, folds, mixed_precision=mixed_precision,
                                                      checkpoint_name=checkpoint_name)

    print(f"=== Processing {ordered_image_files}, {roi_segmentation_file}:")
    print("preprocessing...")
    d, s, properties = trainer.preprocess_patient(ordered_image_files, roi_segmentation_file)
    d = trainer.combine_data_and_seg(d, s)

    print("predicting...")
    trainer.load_checkpoint_ram(params[0], False)
    pred = trainer.predict_preprocessed_data_return_pred_and_logits(d[None], mixed_precision=mixed_precision)[1]

    for p in params[1:]:
        trainer.load_checkpoint_ram(p, False)
        new_pred = trainer.predict_preprocessed_data_return_pred_and_logits(d[None], mixed_precision=mixed_precision)[1]
        for it in range(len(pred)):
            pred = [p + n_p for p, n_p in zip(pred, new_pred)]

    if len(params) > 1:
        pred = [p / len(params) for p in pred]

    return pred
