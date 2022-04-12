from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np


class ClassificationDatasetAnalyzer(DatasetAnalyzer):
    # added max_dimensions for padding later, _get_voxels_in_foreground is not used currently
    def analyze_dataset(self, collect_intensityproperties=True):
        # get all spacings and sizes
        sizes, spacings = self.get_sizes_and_spacings_after_cropping()

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
        dataset_properties['all_sizes'] = sizes
        dataset_properties['all_spacings'] = spacings
        dataset_properties['all_classes'] = all_classes
        dataset_properties['all_classification_labels'] = all_classification_labels
        dataset_properties['modalities'] = modalities  # {idx: modality name}
        dataset_properties['intensityproperties'] = intensityproperties
        dataset_properties['size_reductions'] = size_reductions  # {patient_id: size_reduction}

        dimensions = [[si*sp for si, sp in zip(size, spacing)] for size, spacing in zip(sizes, spacings)]
        dataset_properties['max_dimensions'] = [np.max(d) for d in zip(*dimensions)]  # added

        save_pickle(dataset_properties, join(self.folder_with_cropped_data, "dataset_properties.pkl"))
        return dataset_properties

    def get_classification_labels(self):
        datasetjson = load_json(join(self.folder_with_cropped_data, "dataset.json"))
        return datasetjson['classification_labels']