from universalclassifier.training.network_training.ClassifierTrainerBestCheckpoint import ClassifierTrainerBestCheckpoint
import torch
from universalclassifier.network_architecture.bagnet.bagnet import bagnet33

class ClassifierTrainerBestCheckpointBagNet(ClassifierTrainerBestCheckpoint):
    def initialize_network(self):
        if not self.threeD:
            raise RuntimeError("2D network not implemented")

        self.network = bagnet33(self.num_input_channels, self.num_classification_classes, base_channels=16)

        if torch.cuda.is_available():
            self.network.cuda()

