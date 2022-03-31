from pathlib import Path
from nnunet.experiment_planning.utils import create_lists_from_splitted_dataset_folder
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

import SimpleITK as sitk
import numpy as np
import json


def add_segmentations_to_task_folder(folder):
    folder = Path(folder)
    labelstr_folder = (folder / 'labelsTr')
    imagesstr_folder = (folder / 'imagesTr')

    maybe_mkdir_p(str(labelstr_folder))

    image_files = create_lists_from_splitted_dataset_folder(imagesstr_folder)

    add_empty_segmentation_images_if_missing(image_files)

    filename = folder/'dataset.json'
    print(filename)
    with open(filename, 'r') as fp:
        dataset_json = json.load(fp)
    with open(filename, 'w') as fp:
        datset_json = add_segmentations_to_dataset_json(dataset_json)
        json.dump(datset_json, fp, indent=2)


def add_segmentations_to_dataset_json(dataset_json):
    tr = dataset_json["training"]
    for it in range(len(tr)):
        if "label" not in tr[it].keys():
            tr[it].update({
                "label": to_labelsdir_without_modality_identifier(tr[it]['image'])
            })
    if "labels" not in dataset_json.keys():
        dataset_json["labels"] = {
            "0": "background",
            "1": "foreground",
        }
    print("Done.")
    return dataset_json


def to_labelsdir_without_modality_identifier(filename):
    filename = Path(filename)
    assert str(filename)[-7:] == '.nii.gz', 'images dir should only contain nii.gz image file'
    suffix = '.nii.gz'
    imagedir = filename.parent.name
    assert imagedir in ('imagesTr', 'imagesTs'), 'in_labelstr_without_modality_identifier() is only for converting ' \
                                                 'files in an imagesXX dir to the corresponding files in labelsXX dir '
    labelsdir = imagedir.replace('image', 'label')
    taskdir = filename.parent.parent
    stem = str(Path(filename).stem.rsplit('_', 1)[0])
    return str(taskdir / labelsdir / (stem+suffix))


def corresponding_segmentation_filenames(files):
    return [to_labelsdir_without_modality_identifier(f[0]) for f in files]


def add_empty_segmentation_images_if_missing(image_files):
    segmentation_files = corresponding_segmentation_filenames(image_files)
    for image_file_list, segmentation_file in zip(image_files, segmentation_files):
        image_file = image_file_list[0]
        if not Path(segmentation_file).is_file():
            print(f"{segmentation_file} is missing. Automatically generating it with only foreground voxels...")
            image_itk = sitk.ReadImage(image_file)
            original_size_of_raw_data = np.array(image_itk.GetSize())[[2, 1, 0]]
            segm = np.ones(original_size_of_raw_data, dtype=np.short)
            segm_itk = sitk.GetImageFromArray(segm)

            segm_itk.SetOrigin(image_itk.GetOrigin())
            segm_itk.SetSpacing(image_itk.GetSpacing())
            segm_itk.SetDirection(image_itk.GetDirection())

            sitk.WriteImage(segm_itk, str(segmentation_file), True)

