from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np

from multiprocessing import Pool
from nnunet.configuration import default_num_threads

from collections import OrderedDict


class ClassificationDatasetAnalyzer(DatasetAnalyzer):
    """ This was not necessary so far.
    def __init__(self, folder_with_cropped_data, overwrite=True, num_processes=default_num_threads):
        super().__init__(folder_with_cropped_data, overwrite, num_processes)
        self.max_voxels_used_for_intensity_computation = 1e8

    # new
    def _set_total_nr_voxels_in_foreground(self):
        if self.overwrite or not isfile(self.intensityproperties_file):
            print(f"Counting total number of voxels in foreground...")
            p = Pool(self.num_processes)
            v = p.starmap(self._get_nr_voxels_in_foreground, zip(self.patient_identifiers))
            total = 0
            for iv in v:
                total += iv
            print(f"Total voxels in foreground: {total}")
            self.intensity_computation_voxel_stride = max(10,  # originally 1 every 10 voxels
                                                          int(total // self.max_voxels_used_for_intensity_computation))
            print(f"Only using one voxel every {self.intensity_computation_voxel_stride} voxels for computing "
                  f"intensity properties.")

    # new
    def _get_nr_voxels_in_foreground(self, patient_identifier):
        all_data = np.load(join(self.folder_with_cropped_data, patient_identifier) + ".npz")['data']
        mask = all_data[-1] > 0
        return np.sum(mask)

    # adjusted to allow for faster processing of larger datasets
    def _get_voxels_in_foreground(self, patient_identifier, modality_id):
        all_data = np.load(join(self.folder_with_cropped_data, patient_identifier) + ".npz")['data']
        modality = all_data[modality_id]
        mask = all_data[-1] > 0
        ## taking every self.intensity_computation_voxel_stride instead of every 10
        voxels = list(modality[mask][::self.intensity_computation_voxel_stride])  # no need to take every voxel
        return voxels
    """

    # added max_dimensions for padding later, _get_voxels_in_foreground is not used currently
    def analyze_dataset(self, collect_intensityproperties=True):
        # get all spacings and sizes
        sizes, spacings = self.get_sizes_and_spacings_after_cropping()

        # get all classes and what classes are in what patients
        # class min size
        # region size per class
        classes = self.get_classes()
        all_classes = [int(i) for i in classes.keys() if int(i) > 0]

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
        dataset_properties['modalities'] = modalities  # {idx: modality name}
        dataset_properties['intensityproperties'] = intensityproperties
        dataset_properties['size_reductions'] = size_reductions  # {patient_id: size_reduction}

        dimensions = [[si*sp for si, sp in zip(size, spacing)] for size, spacing in zip(sizes, spacings)]
        dataset_properties['max_dimensions'] = [np.max(d) for d in zip(*dimensions)]  # added

        save_pickle(dataset_properties, join(self.folder_with_cropped_data, "dataset_properties.pkl"))
        return dataset_properties
