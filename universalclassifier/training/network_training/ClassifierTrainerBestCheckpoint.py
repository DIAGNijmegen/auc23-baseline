from universalclassifier.training.network_training.ClassifierTrainer import ClassifierTrainer
import torch
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.metrics import roc_auc_score
import numpy as np


class ClassifierTrainerBestCheckpoint(ClassifierTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, stage=None, unpack_data=True,
                 deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, stage, unpack_data, deterministic, fp16)
        self.online_eval_outputs_softmax = []
        self.online_eval_targets = []

    def run_online_evaluation(self, outputs, targets):
        with torch.no_grad():
            outputs_softmax = [softmax_helper(output) for output in outputs]
            if len(self.online_eval_outputs_softmax) == 0:
                assert len(self.online_eval_targets) == 0
                self.online_eval_outputs_softmax = [output_softmax.detach().cpu().numpy() for output_softmax in
                                                    outputs_softmax]
                self.online_eval_targets = [target.detach().cpu().numpy() for target in targets]
            else:
                num_outputs = len(outputs)
                for it in range(num_outputs):
                    self.online_eval_outputs_softmax[it] = np.append(self.online_eval_outputs_softmax[it],
                                                                     outputs_softmax[it].detach().cpu().numpy())
                    self.online_eval_targets[it] = np.append(self.online_eval_targets[it],
                                                             targets[it].detach().cpu().numpy())

    def finish_online_evaluation(self):
        aucs_per_output = []
        for output_softmax, target in zip(self.online_eval_outputs_softmax, self.online_eval_targets):
            num_classes = output_softmax.shape[1]
            aucs_per_class = []
            for c in range(1, num_classes):
                out = output_softmax[:, c]
                tgt = target == c
                auc = roc_auc_score(tgt, out)
                aucs_per_class.append(auc)
            aucs_per_output.append(np.mean(aucs_per_class))

        self.all_val_eval_metrics.append(np.mean(aucs_per_output))

        self.online_eval_outputs_softmax = []
        self.online_eval_targets = []
