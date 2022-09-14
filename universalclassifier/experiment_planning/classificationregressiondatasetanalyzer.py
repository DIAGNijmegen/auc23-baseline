from universalclassifier.experiment_planning.classificationdatasetanalyzer import ClassificationDatasetAnalyzer
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np

from collections import OrderedDict
from multiprocessing import Pool


class ClassificationRegressionDatasetAnalyzer(ClassificationDatasetAnalyzer):
    def analyze_dataset(self, collect_intensityproperties=True, nr_voxels_upper_bound=1e9):
        # get all spacings and sizes
        self.sizes, self.spacings = self.get_sizes_and_spacings_after_cropping()
        self.set_foreground_voxel_stride(nr_voxels_upper_bound)

        # get all classes and what classes are in what patients
        # class min size
        # region size per class
        classes = self.get_classes()
        all_classes = [int(i) for i in classes.keys() if int(i) > 0]

        classification_labels = self.get_classification_labels()
        all_classification_labels = [[int(i) for i in label["values"].keys()] for label in classification_labels]

        regression_label_stats = self.get_regression_labels_with_stats()

        # modalities
        modalities = self.get_modalities()

        # collect intensity information
        if collect_intensityproperties:
            # self._set_total_nr_voxels_in_foreground()
            intensityproperties = self.collect_intensity_properties(len(modalities))
        else:
            intensityproperties = None

        print("intensity properties collected.")

        # size reduction by cropping
        size_reductions = self.get_size_reduction_by_cropping()

        dataset_properties = dict()
        dataset_properties['all_sizes'] = self.sizes
        dataset_properties['all_spacings'] = self.spacings
        dataset_properties['all_classes'] = all_classes
        dataset_properties['all_classification_labels'] = all_classification_labels
        dataset_properties['regression_label_stats'] = regression_label_stats
        dataset_properties['modalities'] = modalities  # {idx: modality name}
        dataset_properties['intensityproperties'] = intensityproperties
        dataset_properties['size_reductions'] = size_reductions  # {patient_id: size_reduction}

        dimensions = [[si * sp for si, sp in zip(size, spacing)] for size, spacing in zip(self.sizes, self.spacings)]
        dataset_properties['max_dimensions'] = [np.max(d) for d in zip(*dimensions)]  # added

        save_pickle(dataset_properties, join(self.folder_with_cropped_data, "dataset_properties.pkl"))
        return dataset_properties

    def get_regression_labels_with_stats(self):
        datasetjson = load_json(join(self.folder_with_cropped_data, "dataset.json"))
        training_items = datasetjson['training']
        labels = datasetjson['regression_labels']
        for label in labels:
            name = label["name"]
            labeled_items = [item for item in training_items
                             if not (item[name] is None or np.isnan(name))]
            num_labeled = len(labeled_items)
            label["fraction_labeled"] = num_labeled / len(training_items)
            label["std"] = np.std(labeled_items)
            label["mean"] = np.mean(labeled_items)
        return labels

