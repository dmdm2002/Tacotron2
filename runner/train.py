import os
import time
import argparse
import math

from numpy import finfo

import torch
from torch.utils.data import DataLoader

from data.dataset import TextMelDataset, TextMelCollate
from losses.loss import Tacotron2Loss
from utils import hparams


class Train(hparams):
    def __init__(self):
        super().__init__()
