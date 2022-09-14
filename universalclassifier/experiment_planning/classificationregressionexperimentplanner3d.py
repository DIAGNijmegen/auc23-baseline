from batchgenerators.utilities.file_and_folder_operations import *
from universalclassifier.experiment_planning.classificationexperimentplanner3d import ClassificationExperimentPlanner3D


class ClassificationRegressionExperimentPlanner3D(ClassificationExperimentPlanner3D):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super().__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "universal_regressor_and_classifier_plans_v1.0"
        self.plans_fname = join(self.preprocessed_output_folder, "UniversalRegressorAndClassifierPlansv1.0_plans_3D.pkl")
        self.preprocessor_name = "UniversalClassifierPreprocessor"

    def plan_experiment(self, original_spacing, max_shape):
        super().plan_experiment(original_spacing, max_shape)
        self.load_my_plans()
        self.plans['regression_label_stats'] = self.dataset_properties['regression_label_stats']
        self.save_my_plans()
