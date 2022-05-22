
import shutil

import nnunet
from batchgenerators.utilities.file_and_folder_operations import *

from nnunet.configuration import default_num_threads
from nnunet.training.model_restore import recursive_find_python_class
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21
import numpy as np

import universalclassifier

class ClassificationExperimentPlanner3D(ExperimentPlanner3D_v21):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super().__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "universal_classifier_plans_v1.0"
        self.plans_fname = join(self.preprocessed_output_folder, "UniversalClassifierPlansv1.0_plans_3D.pkl")
        self.preprocessor_name = "UniversalClassifierPreprocessor"
        self.max_shape_limit = [240, 240, 240]  # hard coded for now
        self.minimum_batch_size = 2  # Works for I3dr model with 240^3 img size

    def get_properties_for_stage(self, original_spacing, max_shape):
        # TODO implement smart way to find image size and batch size by seeing what fits in GPU memory

        # TODO: work out how memory scales with image size. Naively assuming the same nr of voxels results in the same
        #  memory consumption now
        voxels = np.product(max_shape)
        voxels_limit = np.product(self.max_shape_limit)
        ratio = np.cbrt(voxels_limit / voxels)  # cube root because of scaling in each dimension

        if ratio < 1:  # so voxels_limit < voxels
            max_shape = [np.ceil(s*ratio).astype(np.int) for s in max_shape]
            current_spacing = [s/ratio for s in original_spacing]
            voxels = np.product(max_shape)
            ratio = voxels_limit / voxels
        else:
            current_spacing = original_spacing

        # TODO: work out how memory scales with image size. Naively scale batchsize by fraction of voxels wrt an image
        #  with a size of max_shape_limit for now. May not work.
        batch_size = np.max([
            self.minimum_batch_size,
            np.floor(self.minimum_batch_size * ratio).astype(int)
        ])

        do_dummy_2D_data_aug = (max(max_shape) / max_shape[0]) > self.anisotropy_threshold

        plan = {
            'batch_size': batch_size,
            'image_size': max_shape,
            'current_spacing': current_spacing,
            'original_spacing': original_spacing,
            'do_dummy_2D_data_aug': do_dummy_2D_data_aug,
        }
        return plan

    def plan_experiment(self):
        use_nonzero_mask_for_normalization = self.determine_whether_to_use_mask_for_norm()
        print("Are we using the nonzero mask for normalizaion?", use_nonzero_mask_for_normalization)
        spacings = self.dataset_properties['all_spacings']
        sizes = self.dataset_properties['all_sizes']

        all_classes = self.dataset_properties['all_classes']
        all_classification_labels = self.dataset_properties['all_classification_labels']
        modalities = self.dataset_properties['modalities']
        num_modalities = len(list(modalities.keys()))

        target_spacing = self.get_target_spacing()
        new_shapes = [np.array(i) / target_spacing * np.array(j) for i, j in zip(spacings, sizes)]

        max_spacing_axis = np.argmax(target_spacing)
        remaining_axes = [i for i in list(range(3)) if i != max_spacing_axis]
        self.transpose_forward = [max_spacing_axis] + remaining_axes
        self.transpose_backward = [np.argwhere(np.array(self.transpose_forward) == i)[0][0] for i in range(3)]

        # we base our calculations on the median shape of the datasets
        median_shape = np.median(np.vstack(new_shapes), 0)
        print("the median shape of the dataset is ", median_shape)

        max_shape = np.max(np.vstack(new_shapes), 0)
        print("the max shape in the dataset is ", max_shape)
        min_shape = np.min(np.vstack(new_shapes), 0)
        print("the min shape in the dataset is ", min_shape)

        # how many stages will the image pyramid have?
        self.plans_per_stage = list()

        target_spacing_transposed = np.array(target_spacing)[self.transpose_forward]
        # median_shape_transposed = np.array(median_shape)[self.transpose_forward]
        # We instead want to use max shape here for classification:
        max_shape_transposed = np.array(max_shape)[self.transpose_forward]
        print("the transposed max shape of the dataset is ", max_shape_transposed)

        print("generating configuration for 3d_fullres")
        # current_spacing and original_spacing are the same here
        self.plans_per_stage.append(self.get_properties_for_stage(target_spacing_transposed,
                                                                  max_shape_transposed))

        # thanks Zakiyi (https://github.com/MIC-DKFZ/nnUNet/issues/61) for spotting this bug :-)
        # if np.prod(self.plans_per_stage[-1]['median_patient_size_in_voxels'], dtype=np.int64) / \
        #        architecture_input_voxels < HOW_MUCH_OF_A_PATIENT_MUST_THE_NETWORK_SEE_AT_STAGE0:

        # Removed code here, because only allowing one stage for classifier.

        self.plans_per_stage = self.plans_per_stage[::-1]
        self.plans_per_stage = {i: self.plans_per_stage[i] for i in range(len(self.plans_per_stage))}  # convert to dict

        print(self.plans_per_stage)
        print("transpose forward", self.transpose_forward)
        print("transpose backward", self.transpose_backward)

        normalization_schemes = self.determine_normalization_scheme()

        # these are independent of the stage
        plans = {'num_stages': len(list(self.plans_per_stage.keys())), 'num_modalities': num_modalities,
                 'modalities': modalities, 'normalization_schemes': normalization_schemes,
                 'dataset_properties': self.dataset_properties, 'list_of_npz_files': self.list_of_cropped_npz_files,
                 'original_spacings': spacings, 'original_sizes': sizes,
                 'preprocessed_data_folder': self.preprocessed_output_folder,
                 'num_classes': len(all_classes),
                 'all_classes': all_classes,
                 'num_classification_classes': [len(labels) for labels in all_classification_labels],
                 'all_classification_labels': all_classification_labels,
                 'use_mask_for_norm': use_nonzero_mask_for_normalization,
                 'transpose_forward': self.transpose_forward, 'transpose_backward': self.transpose_backward,
                 'data_identifier': self.data_identifier, 'plans_per_stage': self.plans_per_stage,
                 'preprocessor_name': self.preprocessor_name,
                 }

        self.plans = plans
        self.save_my_plans()

    def run_preprocessing(self, num_threads):
        if os.path.isdir(join(self.preprocessed_output_folder, "gt_segmentations")):
            shutil.rmtree(join(self.preprocessed_output_folder, "gt_segmentations"))
        shutil.copytree(join(self.folder_with_cropped_data, "gt_segmentations"),
                        join(self.preprocessed_output_folder, "gt_segmentations"))
        normalization_schemes = self.plans['normalization_schemes']
        use_nonzero_mask_for_normalization = self.plans['use_mask_for_norm']
        intensityproperties = self.plans['dataset_properties']['intensityproperties']
        preprocessor_class = recursive_find_python_class([join(universalclassifier.__path__[0], "preprocessing")],
                                                         self.preprocessor_name,
                                                         current_module="universalclassifier.preprocessing")
        assert preprocessor_class is not None
        preprocessor = preprocessor_class(normalization_schemes, use_nonzero_mask_for_normalization,
                                         self.transpose_forward,
                                          intensityproperties)
        target_spacings = [i["current_spacing"] for i in self.plans_per_stage.values()]
        target_sizes = [i["image_size"] for i in self.plans_per_stage.values()]
        if self.plans['num_stages'] > 1 and not isinstance(num_threads, (list, tuple)):
            num_threads = (default_num_threads, num_threads)
        elif self.plans['num_stages'] == 1 and isinstance(num_threads, (list, tuple)):
            num_threads = num_threads[-1]
        preprocessor.run(target_spacings, target_sizes, self.folder_with_cropped_data, self.preprocessed_output_folder,
                         self.plans['data_identifier'], num_threads)