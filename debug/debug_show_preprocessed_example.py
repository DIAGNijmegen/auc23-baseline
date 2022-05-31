import numpy as np
import matplotlib.pyplot as plt
import os
import glob


def show_image_channels(image):
    fix, ax = plt.subplots(len(image), 3)
    for channel in range(len(image)):
        sh = image.shape
        ax[channel, 0].imshow(image[channel, sh[1] // 2, :, :])
        ax[channel, 1].imshow(image[channel, :, sh[2] // 2, :])
        ax[channel, 2].imshow(image[channel, :, :, sh[3] // 2])
    plt.show()


def show_image_channels_for_folder(folder):
    print(folder)
    for filename in glob.glob(os.path.join(folder, "*.npz")):
        print(filename)
        image = np.load(filename)['data']
        show_image_channels(image)


if __name__ == "__main__":
    preprocessed_root = "/mnt/internal_storage/sdc2/luuk/stoic/public_train_for_uc/universal_classifier_format/preprocessed/"
    task_folder = "Task001_COVID19Severity"
    plans_folder = "universal_classifier_plans_v1.0_stage0"
    folder = os.path.join(preprocessed_root, task_folder, plans_folder)
    folder = "/home/lhboulogne/Downloads/test/npzs/"
    show_image_channels_for_folder(folder)

