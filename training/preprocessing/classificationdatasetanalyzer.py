from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from batchgenerators.utilities.file_and_folder_operations import *


class ClassificationDatasetAnalyzer(DatasetAnalyzer):
    def get_classification_classes(self):
        datasetjson = load_json(join(self.folder_with_cropped_data, "dataset.json"))
        return datasetjson['classification_labels']

    def analyze_dataset(self, **kwargs):
        dataset_properties = super().analyze_dataset(**kwargs)
        dataset_properties['classification_labels'] = self.get_classification_classes()
        save_pickle(dataset_properties, join(self.folder_with_cropped_data, "dataset_properties.pkl"))
        return dataset_properties
