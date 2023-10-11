import os
import time
import argparse
import math
import torchaudio

from tqdm import tqdm
from numpy import finfo

import torch
from torch.utils.data import DataLoader

from data.dataset import TextMelDataset, TextMelCollate
from losses.loss import Tacotron2Loss
from utils.hparams import HyperParams
from utils.logging import Logger, hparam_save


class Train(HyperParams):
    def __init__(self):
        super().__init__()

        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.ckp_path, exist_ok=True)
        os.makedirs(self.train_output_path, exist_ok=True)
        os.makedirs(self.valid_sample_path, exist_ok=True)
        hparam_save(self.log_path, HyperParams().__dict__)

    def to_gpu(self, x):
        x = x.contiguous()
        if torch.cuda.is_available():
            x = x.cuda(non_blocking=True)
        return x

    def batch_to_gpu(self, batch):
        text_padded, text_lengths, mel_specgram_padded, mel_specgram_lengths, gate_padded = batch
        text_padded = self.to_gpu(text_padded).long()
        text_lengths = self.to_gpu(text_lengths).long()
        mel_specgram_padded = self.to_gpu(mel_specgram_padded).float()
        gate_padded = self.to_gpu(gate_padded).float()
        mel_specgram_lengths = self.to_gpu(mel_specgram_lengths).long()
        x = (text_padded, text_lengths, mel_specgram_padded, mel_specgram_lengths)
        y = (mel_specgram_padded, gate_padded)
        return x, y
    def prepare_data_loader(self):
        hp = HyperParams()
        trainset = TextMelDataset(self.training_files, hp)
        valset = TextMelDataset(self.validation_files, hp)
        collate_fn = TextMelCollate(self.n_frames_per_step)

        train_loader = DataLoader(trainset,
                                  shuffle=True,
                                  batch_size=self.batch_size,
                                  pin_memory=False,
                                  collate_fn=collate_fn,
                                  )

        return train_loader, valset, collate_fn

    def run(self):
        bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
        model = bundle.get_tacotron2().to(self.device)
        # waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
        # waveglow.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # if self.fp16_run:
        #     from apex import amp
        #     model, optimizer = amp.initialize(
        #         model, optimizer, opt_level='O2')

        criterion = Tacotron2Loss()
        logger = Logger(self.log_path)

        train_loader, valset, collate_fn = self.prepare_data_loader()

        iteration = 0
        epoch_offset = 0

        for epoch in range(epoch_offset, self.epochs):
            model.train()
            for idx, item in enumerate(tqdm(train_loader, desc=f'[Train Epoch ==> {epoch}/{self.epochs}]: ')):
                (text_padded, text_lengths, mel_specgram_padded, mel_specgram_lengths), y = self.batch_to_gpu(item)
                y_pred = model(text_padded, text_lengths, mel_specgram_padded, mel_specgram_lengths)

                loss = criterion(y_pred, y)

                optimizer.zero_grad()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.grad_clip_thresh)
                loss.backward()
                optimizer.step()

                logger.log_training(loss.item(), grad_norm, iteration)


if __name__ == '__main__':
    a = Train()
    a.run()


