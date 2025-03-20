# -*- coding: utf-8 -*-
import os
import sys

import torch
from torch import nn
from models.base import LambdaLayer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class AutoEncoderModel(nn.Module):
    def __init__(self, input_channels, n_pred_steps):
        super().__init__()
        self.input_channels = input_channels
        self.n_pred_steps = n_pred_steps
        self.encoder = nn.Sequential(
            nn.Linear(input_channels, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
        )
        self.predictor = nn.Sequential(
            nn.LSTM(
                input_size=64,
                hidden_size=64 // 2,
                batch_first=True,
                bidirectional=True,
                num_layers=2
            ),
            LambdaLayer(lambda x: x[0]),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),

        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        latent = self.encode_step(x)
        if self.training:
            latent_history, latent_future = torch.split(latent, [latent.size(1) - self.n_pred_steps, self.n_pred_steps], 1)
        else:
            latent_history, latent_future = latent, torch.zeros(latent.size(0), self.n_pred_steps, latent.size(2))
        latent_future_pred = self.predict_step(latent_history)
        decoder_inputs = torch.cat([latent_history, latent_future_pred], dim=1)
        x_recover = self.decode_step(decoder_inputs)
        return x_recover, (latent_future_pred, latent_future)

    def encode_step(self, x):
        x = self.encoder(x)
        return x

    def predict_step(self, x):
        states = None
        x_pred_list = []
        for i in range(self.n_pred_steps):
            x_pred = self.predictor(x)
            x_pred = x_pred[:, [-1], :]
            x_pred_list.append(x_pred)
            x = torch.cat([x, x_pred], dim=1)
        return torch.cat(x_pred_list, dim=1)

    def decode_step(self, x):
        x = self.decoder(x)
        return x


class AutoEncoderWithClassifier(AutoEncoderModel):
    def __init__(self, input_channels, n_pred_steps, num_classes=1):
        super().__init__(input_channels, n_pred_steps)
        self.classifier = nn.Linear(64 * n_pred_steps, num_classes)

    def forward(self, x):
        x_recover, (latent_future_pred, latent_future) = super().forward(x)
        logits = self.classifier(latent_future_pred.reshape(latent_future_pred.shape[0], -1))
        return x_recover, logits, (latent_future_pred, latent_future)
