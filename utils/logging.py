"""추후 wandb로 교체할 예정"""
import random

import torch
from torch.utils.tensorboard import SummaryWriter
from utils.plotting import plot_alignment_to_numpy, plot_spectrogram_to_numpy, plot_gate_outputs_to_numpy


class Logger(SummaryWriter):
    def __init__(self, logdir):
        super().__init__()

    def log_training(self, loss, grad_norm, iteration):
        self.add_scalar("training/loss", loss, iteration)
        self.add_scalar("training/grad_norm", grad_norm, iteration)

    def log_validation(self, loss, model, y, y_pred, iteration):
        self.add_scalar("validation/loss", loss, iteration)

        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        idx = random.randint(0, alignments.size(0), -1)
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration, dataformats='HWC')


# hyperparameter information save
def hparam_save(log_dir, hp):
    with open(f"{log_dir}.txt", 'w') as f:
        for key, value in hp.items():
            f.write('%s:%s\n' % (key, value))
