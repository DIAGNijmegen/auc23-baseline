import numpy as np
from nnunet.configuration import default_num_threads
from nnunet.preprocessing.cropping import get_case_identifier_from_npz
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing.pool import Pool
from nnunet.preprocessing.preprocessing import GenericPreprocessor


class UniversalClassifierPreprocessor(GenericPreprocessor):

    # only changed all_classes to max_dimensions
    def run(self, target_spacings, target_sizes, input_folder_with_cropped_npz, output_folder, data_identifier,
            num_threads=default_num_threads, force_separate_z=None):
        """

        :param target_spacings: list of lists [[1.25, 1.25, 5]]
        :param input_folder_with_cropped_npz: dim: c, x, y, z | npz_file['data'] np.savez_compressed(fname.npz, data=arr)
        :param output_folder:
        :param num_threads:
        :param force_separate_z: None
        :return:
        """
        print("Initializing to run preprocessing")
        print("npz folder:", input_folder_with_cropped_npz)
        print("output_folder:", output_folder)
        list_of_cropped_npz_files = subfiles(input_folder_with_cropped_npz, True, None, ".npz", True)
        maybe_mkdir_p(output_folder)
        num_stages = len(target_spacings)
        if not isinstance(num_threads, (list, tuple, np.ndarray)):
            num_threads = [num_threads] * num_stages

        assert len(num_threads) == num_stages

        # we need to know which classes are present in this dataset so that we can precompute where these classes are
        # located. This is needed for oversampling foreground <- not needed for classifier
        #all_classes = load_pickle(join(input_folder_with_cropped_npz, 'dataset_properties.pkl'))['all_classes']

        for i in range(num_stages):
            all_args = []
            output_folder_stage = os.path.join(output_folder, data_identifier + "_stage%d" % i)
            maybe_mkdir_p(output_folder_stage)
            spacing = target_spacings[i]
            target_size = target_sizes[i]
            for j, case in enumerate(list_of_cropped_npz_files):
                case_identifier = get_case_identifier_from_npz(case)
                args = spacing, target_size, case_identifier, output_folder_stage, input_folder_with_cropped_npz, force_separate_z
                all_args.append(args)
            p = Pool(num_threads[i])
            p.starmap(self._run_internal, all_args)
            p.close()
            p.join()

    # Add padding to max_size
    def _run_internal(self, target_spacing, target_size, case_identifier, output_folder_stage, cropped_output_dir,
                      force_separate_z):
        data, seg, properties = self.load_cropped(cropped_output_dir, case_identifier)

        data = data.transpose((0, *[i + 1 for i in self.transpose_forward]))
        seg = seg.transpose((0, *[i + 1 for i in self.transpose_forward]))

        data, seg, properties = self.resample_and_normalize(data, target_spacing,
                                                            properties, seg, force_separate_z)

        #add padding so that all images have the same size

        all_data = np.vstack((data, seg)).astype(np.float32)
        print("saving: ", os.path.join(output_folder_stage, "%s.npz" % case_identifier))
        np.savez_compressed(os.path.join(output_folder_stage, "%s.npz" % case_identifier),
                            data=all_data.astype(np.float32))
