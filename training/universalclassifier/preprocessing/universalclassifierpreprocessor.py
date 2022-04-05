import numpy as np
import os
import pickle

from nnunet.preprocessing.preprocessing import GenericPreprocessor


class UniversalClassifierPreprocessor(GenericPreprocessor):
    def _run_internal(self, target_spacing, case_identifier, output_folder_stage, cropped_output_dir, force_separate_z,
                      all_classes):
        data, seg, properties = self.load_cropped(cropped_output_dir, case_identifier)

        print("hiero.")
        data = data.transpose((0, *[i + 1 for i in self.transpose_forward]))
        seg = seg.transpose((0, *[i + 1 for i in self.transpose_forward]))


        print("hieroow.")
        data, seg, properties = self.resample_and_normalize(data, target_spacing,
                                                            properties, seg, force_separate_z)

        print("Should add padding here.")
        #add padding so that all images have the same size

        all_data = np.vstack((data, seg)).astype(np.float32)
        print("saving: ", os.path.join(output_folder_stage, "%s.npz" % case_identifier))
        np.savez_compressed(os.path.join(output_folder_stage, "%s.npz" % case_identifier),
                            data=all_data.astype(np.float32))
