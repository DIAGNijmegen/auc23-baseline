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
        voxels = list(modality[mask][::100]) # no need to take every voxel ## taking every 100 instead of every 10
        return voxels
