import numpy as np
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.utilities.file_and_folder_operations import *


class DataLoader3D(SlimDataLoaderBase):
    def __init__(self, data, image_size, batch_size, memmap_mode="r"):
        super(DataLoader3D, self).__init__(data, batch_size, None)
        self.memmap_mode = memmap_mode
        self.num_channels = None
        self.image_size = image_size
        self.list_of_keys = list(self._data.keys())
        self.data_shape, self.seg_shape, self.target_shape = self.determine_shapes()

    def determine_shapes(self):
        k = list(self._data.keys())[0]

        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']

        if 'properties' in self._data[k].keys():
            properties = self._data[k]['properties']
        else:
            properties = load_pickle(self._data[k]['properties_file'])

        num_color_channels = case_all_data.shape[0] - 1
        data_shape = (self.batch_size, num_color_channels, *self.image_size)
        seg_shape = (self.batch_size, 1, *self.image_size)
        target_shape = (self.batch_size, len(properties['target']))
        return data_shape, seg_shape, target_shape

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)
        target = np.zeros(self.target_shape, dtype=np.int64)
        case_properties = []
        for j, i in enumerate(selected_keys):

            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                properties = load_pickle(self._data[i]['properties_file'])
            case_properties.append(properties)

            # cases are stored as npz, but we require unpack_dataset to be run. This will decompress them into npy
            # which is much faster to access
            if isfile(self._data[i]['data_file'][:-4] + ".npy"):
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)
            else:
                case_all_data = np.load(self._data[i]['data_file'])['data']

            data[j] = case_all_data[:-1]
            seg[j, 0] = case_all_data[-1:]
            target[j] = properties['target']

        return {'data': data, 'seg': seg, 'target': target, 'properties': case_properties, 'keys': selected_keys}