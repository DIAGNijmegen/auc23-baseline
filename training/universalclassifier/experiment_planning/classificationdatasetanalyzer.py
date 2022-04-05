from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np


class ClassificationDatasetAnalyzer(DatasetAnalyzer):

    # Taking every 100 voxels instead of every 10
    def _get_voxels_in_foreground(self, patient_identifier, modality_id):
        print(f"Extracting foreground voxels for {patient_identifier}...")
        all_data = np.load(join(self.folder_with_cropped_data, patient_identifier) + ".npz")['data']
        modality = all_data[modality_id]
        mask = all_data[-1] > 0
        voxels = list(modality[mask][::100])  # no need to take every voxel ## taking every 100 instead of every 10
        return voxels

    def analyze_dataset(self, collect_intensityproperties=True):
        dataset_properties = super().analyze_dataset(collect_intensityproperties)
        print(dataset_properties.keys())
        all_sizes = dataset_properties['all_sizes']
        all_spacings = dataset_properties['all_spacings']
        dimensions = [[si*sp for si, sp in zip(size, spacing)] for size, spacing in zip(all_sizes, all_spacings)]
        dataset_properties['max_dimensions'] = [np.max(d) for d in zip(*dimensions)]
        save_pickle(dataset_properties, join(self.folder_with_cropped_data, "dataset_properties.pkl"))
        return dataset_properties
