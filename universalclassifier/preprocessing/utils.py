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

def add_dummy_modality(filename):
    return filename.replace('.nii.gz', '_0000.nii.gz')

def add_segmentations_to_dataset_json(dataset_json):
    tr = dataset_json["training"]
    for it in range(len(tr)):
        if "label" not in tr[it].keys():
            tr[it].update({
                "label": to_corresponding_label_filename(add_dummy_modality(tr[it]['image']))
            })
    if "labels" not in dataset_json.keys():
        dataset_json["labels"] = {
            "0": "background",
            "1": "foreground",
        }
    return dataset_json


def to_corresponding_label_filename(image_filename):
    image_filename = Path(image_filename)
    suffix = '.nii.gz'
    assert str(image_filename)[-7:] == suffix, f'images dir should only contain {suffix} image file'
    imagedir = image_filename.parent.name

    assert imagedir in ('imagesTr', 'imagesTs'), 'in_labelstr_without_modality_identifier() is only for converting ' \
                                                 'files in an imagesXX dir to the corresponding files in labelsXX dir '
    splitted_image_filename = image_filename.stem[-7:].rsplit('_', 2)
    assert len(splitted_image_filename) == 3, "all image filenames in dataset.json must have format 'NAME_PATIENTID_MODALITYID'"
    task_identifier, patient_id, modality_id = splitted_image_filename
    assert patient_id.isnumeric(), "all image filenames in dataset.json must have format 'NAME_PATIENTID_MODALITYID', where PATIENTID is numeric"
    assert modality_id.isnumeric(), "all image filenames in dataset.json must have format 'NAME_PATIENTID_MODALITYID', where MODALITYID is numeric"

    labelsdir = imagedir.replace('image', 'label')
    taskdir = image_filename.parent.parent
    return str(taskdir / labelsdir / (task_identifier + '_' + patient_id + suffix))


def corresponding_segmentation_filenames(files):
    return [to_corresponding_label_filename(f[0]) for f in files]


def add_empty_segmentation_images_if_missing(image_files):
    segmentation_files = corresponding_segmentation_filenames(image_files)
    seg_files_exist = [Path(s).is_file() for s in segmentation_files]
    if sum(seg_files_exist) == 0:  # generate dummy roi files
        for image_file_list, segmentation_file in zip(image_files, segmentation_files):
            print(f"{segmentation_file} is missing. Automatically generating it with only foreground voxels...")
            image_file = image_file_list[0]
            image_itk = sitk.ReadImage(image_file)
            original_size_of_raw_data = np.array(image_itk.GetSize())[[2, 1, 0]]
            segm = np.ones(original_size_of_raw_data, dtype=np.short)
            segm_itk = sitk.GetImageFromArray(segm)

            segm_itk.SetOrigin(image_itk.GetOrigin())
            segm_itk.SetSpacing(image_itk.GetSpacing())
            segm_itk.SetDirection(image_itk.GetDirection())

            sitk.WriteImage(segm_itk, str(segmentation_file), True)
    elif sum(seg_files_exist) != len(segmentation_files):
        raise RuntimeError(
            "Some segmentation masks are missing. I will generate empty masks only if all are missing. I "
            "will use the masks only if all are present.")
    else:
        pass  # roi files already in place

