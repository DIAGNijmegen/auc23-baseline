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
        data, seg, properties = self.central_pad(data, target_size, properties, seg)

        all_data = np.vstack((data, seg)).astype(np.float32)
        print("saving: ", os.path.join(output_folder_stage, "%s.npz" % case_identifier))
        np.savez_compressed(os.path.join(output_folder_stage, "%s.npz" % case_identifier),
                            data=all_data.astype(np.float32))

    def central_pad(self, data, target_size, properties, seg):
        assert not ((data is None) and (seg is None))
        if data is not None:
            assert len(data.shape) == 4, "data must be c x y z"
        if seg is not None:
            assert len(seg.shape) == 4, "seg must be c x y z"

        if data is not None:
            shape = np.array(data[0].shape)
        else:
            shape = np.array(seg[0].shape)


        print(f"Applying uniform padding from {shape} to {target_size}...")
        if data is not None:
            data = self.central_pad_data_or_seg(data, target_size)
        if data is not None:
            seg = self.central_pad_data_or_seg(seg, target_size)

        properties['size after central pad'] = target_size
        return data, seg, properties

    def central_pad_data_or_seg(self, np_image, target_size, outside_val=0):
        target_size = np.asarray([np_image.shape[0]] + list(target_size), dtype=np.int)

        assert len(np_image.shape) == 4, "data must be (c, x, y, z)"
        assert all([s1 <= s2 for s1, s2 in zip(np_image.shape, target_size)]) # only padding, no cropping

        output_image = np.full(target_size, outside_val, np_image.dtype)

        offsets = tuple()
        for it, sh in enumerate(target_size):
            if it == 0:
                offset = slice(None)  # Keep all channels
            else:
                size = sh // 2
                center = (np_image.shape[it] // 2)

                start = center - size

                # computing offset to pad if the crop is partly outside of the scan
                offset = slice(-min(0, start), 2 * size - max(0, (start + 2 * size) - np_image.shape[it]))
            offsets += (offset,)

        print("end of central pad:")
        from pprint import pprint
        pprint(offsets)
        print(target_size)
        output_image[offsets] = np_image
        return output_image
