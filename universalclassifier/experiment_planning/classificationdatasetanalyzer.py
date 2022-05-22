from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np

from collections import OrderedDict
from multiprocessing import Pool

class ClassificationDatasetAnalyzer(DatasetAnalyzer):

    # Added upper bound of 1e9 voxels to compute statistics from. This should be enough to get a good statistic.
    # ROI might be smaller, but probably not a lot smaller since we are cropping to it
    def _get_voxels_in_foreground(self, patient_identifier, modality_id, nr_voxels_upper_bound=1e9):
        all_data = np.load(join(self.folder_with_cropped_data, patient_identifier) + ".npz")['data']
        modality = all_data[modality_id]
        mask = all_data[-1] > 0

        voxels_in_dataset = np.sum([np.product(s) for s in self.sizes])
        stride = int(np.ceil(voxels_in_dataset / nr_voxels_upper_bound))
        voxels = list(modality[mask][::stride])  # no need to take every voxel. edit: Changed 10 to stride
        return voxels

    # added max_dimensions for padding later, _get_voxels_in_foreground is not used currently
    def analyze_dataset(self, collect_intensityproperties=True):
        # get all spacings and sizes
        self.sizes, self.spacings = self.get_sizes_and_spacings_after_cropping()

        # get all classes and what classes are in what patients
        # class min size
        # region size per class
        classes = self.get_classes()
        all_classes = [int(i) for i in classes.keys() if int(i) > 0]

        classification_labels = self.get_classification_labels()
        all_classification_labels = [[int(i) for i in label["values"].keys()] for label in classification_labels]

        # modalities
        modalities = self.get_modalities()

        # collect intensity information
        if collect_intensityproperties:
            #self._set_total_nr_voxels_in_foreground()
            intensityproperties = self.collect_intensity_properties(len(modalities))
        else:
            intensityproperties = None

        # size reduction by cropping
        size_reductions = self.get_size_reduction_by_cropping()

        dataset_properties = dict()
        dataset_properties['all_sizes'] = self.sizes
        dataset_properties['all_spacings'] = self.spacings
        dataset_properties['all_classes'] = all_classes
        dataset_properties['all_classification_labels'] = all_classification_labels
        dataset_properties['modalities'] = modalities  # {idx: modality name}
        dataset_properties['intensityproperties'] = intensityproperties
        dataset_properties['size_reductions'] = size_reductions  # {patient_id: size_reduction}

        dimensions = [[si*sp for si, sp in zip(size, spacing)] for size, spacing in zip(self.sizes, self.spacings)]
        dataset_properties['max_dimensions'] = [np.max(d) for d in zip(*dimensions)]  # added

        save_pickle(dataset_properties, join(self.folder_with_cropped_data, "dataset_properties.pkl"))
        return dataset_properties

    def get_classification_labels(self):
        datasetjson = load_json(join(self.folder_with_cropped_data, "dataset.json"))
        return datasetjson['classification_labels']