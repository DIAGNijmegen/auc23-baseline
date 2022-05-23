# Altered from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/experiment_planning/nnUNet_plan_and_preprocess.py



#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from batchgenerators.utilities.file_and_folder_operations import *
from universalclassifier.experiment_planning.utils import crop

import shutil
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet.preprocessing.sanity_checks import verify_dataset_integrity as verify_dataset_integrity_original_function
from nnunet.training.model_restore import recursive_find_python_class
from nnunet.paths import *

from universalclassifier.preprocessing import utils
from universalclassifier.experiment_planning.classificationdatasetanalyzer import ClassificationDatasetAnalyzer
import universalclassifier
import nnunet


def find_planner(search_in, planner_name, current_module):
    if planner_name is not None:
        planner = recursive_find_python_class([search_in], planner_name,
                                                 current_module=current_module)
        if planner is None:
            raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in "
                               "%s" % (planner_name, current_module))
    else:
        planner = None

    return planner


def main(args):
    task_ids = args.task_ids
    dont_run_preprocessing = args.no_pp
    tl = args.tl
    tf = args.tf
    planner_name3d = args.planner3d
    planner_name2d = args.planner2d
    overwrite_plans = args.overwrite_plans
    overwrite_plans_identifier = args.overwrite_plans_identifier
    verify_dataset_integrity = args.verify_dataset_integrity

    if planner_name3d == "None":
        planner_name3d = None
    if planner_name2d == "None":
        planner_name2d = None

    if overwrite_plans is not None:
        if planner_name2d is not None:
            print("Overwriting plans only works for the 3d planner. I am setting '--planner2d' to None. This will "
                  "skip 2d planning and preprocessing.")
        assert planner_name3d == 'ExperimentPlanner3D_v21_Pretrained', "When using --overwrite_plans you need to use " \
                                                                       "'-pl3d ExperimentPlanner3D_v21_Pretrained'"

    # we need raw data
    tasks = []
    for i in task_ids:
        i = int(i)

        task_name = convert_id_to_task_name(i)

        raw_folder = join(nnUNet_raw_data, task_name)
        print("Adding empty segmentations to task folder if no segmentations are present...", flush=True)
        utils.add_segmentations_to_task_folder(raw_folder)
        if verify_dataset_integrity:
            verify_dataset_integrity_original_function(raw_folder)

        crop(task_name, False, tf)

        tasks.append(task_name)

    planner_3d = find_planner(search_in=join(universalclassifier.__path__[0], 'experiment_planning'),
                              planner_name=planner_name3d,
                              current_module="universalclassifier.experiment_planning")
    planner_2d = find_planner(search_in=join(nnunet.__path__[0], 'experiment_planning'),
                              planner_name=planner_name2d,
                              current_module="nnunet.experiment_planning")

    for t in tasks:
        cropped_out_dir = os.path.join(nnUNet_cropped_data, t)
        preprocessing_output_dir_this_task = os.path.join(preprocessing_output_dir, t)
        #splitted_4d_output_dir_task = os.path.join(nnUNet_raw_data, t)
        #lists, modalities = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

        # we need to figure out if we need the intensity propoerties. We collect them only if one of the modalities is CT
        dataset_json = load_json(join(cropped_out_dir, 'dataset.json'))
        modalities = list(dataset_json["modality"].values())
        collect_intensityproperties = True if (("CT" in modalities) or ("ct" in modalities)) else False
        dataset_analyzer = ClassificationDatasetAnalyzer(cropped_out_dir, overwrite=False, num_processes=tf)  # this class creates the fingerprint
        print("Analyzing data...", flush=True)
        _ = dataset_analyzer.analyze_dataset(collect_intensityproperties)  # this will write output files that will be used by the ExperimentPlanner

        print("Copying...", flush=True)
        maybe_mkdir_p(preprocessing_output_dir_this_task)
        shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task)
        shutil.copy(join(nnUNet_raw_data, t, "dataset.json"), preprocessing_output_dir_this_task)

        threads = (tl, tf)

        print("number of threads: ", threads, "\n")
        if planner_3d is not None:
            if overwrite_plans is not None:
                assert overwrite_plans_identifier is not None, "You need to specify -overwrite_plans_identifier"
                exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_this_task, overwrite_plans,
                                         overwrite_plans_identifier)
            else:
                exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_this_task)
            exp_planner.plan_experiment()
            if not dont_run_preprocessing:  # double negative, yooo
                exp_planner.run_preprocessing(threads)
        if planner_2d is not None:
            exp_planner = planner_2d(cropped_out_dir, preprocessing_output_dir_this_task)
            exp_planner.plan_experiment()
            if not dont_run_preprocessing:  # double negative, yooo
                exp_planner.run_preprocessing(threads)


if __name__ == "__main__":
    main()
