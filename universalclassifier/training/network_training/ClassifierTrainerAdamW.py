# Copied and edited from https://github.com/MIC-DKFZ/nnUNet/blob/8e0ad8ebe0b24165419d087d4451b4631f5b37f2/nnunet/training/network_training/nnUNetTrainer.py#L432

from universalclassifier.training.network_training.ClassifierTrainer import ClassifierTrainer
import torch.optim


class ClassifierTrainerAdamW(ClassifierTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, stage=None, unpack_data=True,
                 deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, stage, unpack_data, deterministic, fp16)

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        self.lr_scheduler = None
